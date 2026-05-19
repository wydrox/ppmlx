# ppmlx/decomposed_engine.py — integration of all v2 memory components
"""
DecomposedMemoryEngine: full v2 extraction pipeline for small models.

Wires: DenseChunker → ContrastiveRetriever → SlotClassifier →
       SlotExtractor → SelfConsistency → Validator → Graph

All components are optional — if any stage fails, the pipeline degrades gracefully
to the next available stage or falls back to the v1 single-pass extraction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import time
import numpy as np

from ppmlx.memory_store import MemoryStore, get_memory_store
from ppmlx.memory_engine import (
    MemoryEngine, MemoryValidator, ShadowMemoryCandidate,
    STATUS_ACTIVE, STATUS_QUARANTINED, STATUS_REJECTED,
)
from ppmlx.dense_chunker import (
    DenseChunker, TextSegment, build_indicator_embeddings,
)
from ppmlx.contrastive_retriever import (
    ContrastiveRetriever, MemorySnapshot, RelevantSegment,
    build_contrastive_pipeline,
)
from ppmlx.slot_classifier import SlotClassifier, ClassifiedSegment
from ppmlx.slot_extractor import SlotExtractor, ExtractedCandidate
from ppmlx.self_consistency import SelfConsistencyExtractor, ConsensusCandidate


@dataclass
class ExtractionReport:
    """Full report from a decomposed extraction run."""
    event_id: str = ""
    # Stage counts
    messages_total: int = 0
    segments_dense: int = 0
    segments_relevant: int = 0
    segments_classified: int = 0
    candidates_extracted: int = 0
    candidates_active: int = 0
    candidates_rejected: int = 0
    # Timing (ms)
    time_dense_chunk: float = 0.0
    time_contrastive: float = 0.0
    time_classify: float = 0.0
    time_extract: float = 0.0
    time_consistency: float = 0.0
    time_validate: float = 0.0
    time_inference: float = 0.0
    time_total: float = 0.0
    # Errors
    errors: list[str] = field(default_factory=list)
    # Fallback
    used_fallback: bool = False


class DecomposedMemoryEngine:
    """
    Memory extraction with task decomposition for small local models.
    
    Usage:
        engine = DecomposedMemoryEngine(store, embedding_model="nomic-embed-text")
        report = engine.extract_from_session(messages, project_id="ppmlx", session_id="s1")
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        extraction_model: str = "gemma-4-e4b-it-optiq",
        embedding_model: str = "qwen3-embedding:0.6b-4bit-dwq",
        # Stage toggles — disable any stage for testing/debug
        enable_dense_chunk: bool = True,
        enable_contrastive: bool = True,
        enable_classify: bool = True,
        enable_consistency: bool = True,
        # Fallback: if decomposed pipeline fails, use v1 single-pass
        fallback_to_v1: bool = True,
    ):
        self.store = store or get_memory_store()
        self.extraction_model = extraction_model
        self.embedding_model = embedding_model
        self.enable_dense_chunk = enable_dense_chunk
        self.enable_contrastive = enable_contrastive
        self.enable_classify = enable_classify
        self.enable_consistency = enable_consistency
        self.fallback_to_v1 = fallback_to_v1

        # Components (lazy init)
        self._dense_chunker: DenseChunker | None = None
        self._retriever: ContrastiveRetriever | None = None
        self._classifier: SlotClassifier | None = None
        self._extractor: SlotExtractor | None = None
        self._consistency: SelfConsistencyExtractor | None = None
        self._validator: MemoryValidator | None = None
        
        # Cached indicator embeddings
        self._indicator_embeddings: list[np.ndarray] | None = None
        self._snapshot: MemorySnapshot | None = None

        # Generation function (lazy — requires loaded model)
        self._generation_fn: Callable | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_session(
        self,
        messages: list[dict],
        project_id: str,
        session_id: str,
        *,
        app_id: str | None = None,
    ) -> ExtractionReport:
        """
        Run full decomposed extraction on a session transcript.
        """
        t0 = time.perf_counter()
        report = ExtractionReport(event_id=session_id)
        report.messages_total = len(messages)

        try:
            # Stage 0: Load current memory state + embeddings
            self._ensure_components(project_id, session_id)

            # Stage 1: Dense chunking
            t1 = time.perf_counter()
            segments = self._run_dense_chunk(messages, report)
            report.time_dense_chunk = (time.perf_counter() - t1) * 1000

            if not segments:
                report.errors.append("dense_chunker: no segments found")
                return self._maybe_fallback(messages, project_id, session_id, app_id, report)

            # Stage 2: Contrastive retrieval
            t2 = time.perf_counter()
            relevant = self._run_contrastive(segments, report)
            report.time_contrastive = (time.perf_counter() - t2) * 1000

            if not relevant:
                report.errors.append("contrastive: all segments filtered out")
                # Not an error — session may genuinely have no new facts
                return report

            # Stage 3-5: Per-segment classification + extraction + consistency
            all_candidates: list[ShadowMemoryCandidate] = []
            event_id = f"{session_id}-decomp"
            for seg in relevant:
                t3 = time.perf_counter()
                classified = self._run_classify(seg, report)
                report.time_classify += (time.perf_counter() - t3) * 1000

                if not classified or "none" in classified.types:
                    continue

                t4 = time.perf_counter()
                if self.enable_consistency:
                    consensus = self._run_consistency(classified, report)
                    for cc in consensus:
                        shadow = self._to_shadow_candidate(cc.candidate)
                        if shadow:
                            all_candidates.append(shadow.with_event(event_id))
                else:
                    candidates = self._run_extract(classified, report)
                    for c in candidates:
                        shadow = self._to_shadow_candidate(c)
                        if shadow:
                            all_candidates.append(shadow.with_event(event_id))
                report.time_extract += (time.perf_counter() - t4) * 1000

            report.segments_classified = len(relevant)
            report.candidates_extracted = len(all_candidates)

            # Stage 6: Validation + graph projection
            t5 = time.perf_counter()
            # Record a synthetic event so the graph engine can resolve namespaces.
            self.store.record_event({
                "event_id": f"{session_id}-decomp",
                "endpoint": "/v1/chat/completions#decomposed",
                "app_id": app_id,
                "project_id": project_id,
                "session_id": session_id,
                "model_alias": self.extraction_model,
                "model_repo": f"mlx-community/{self.extraction_model}",
                "request": {"messages": messages},
                "response_text": "",
                "metadata": {"pipeline": "decomposed_v2"},
            })
            for candidate in all_candidates:
                # Create a minimal event for the validator
                event = {
                    "event_id": f"{session_id}-decomp",
                    "project_id": project_id,
                    "session_id": session_id,
                    "messages": messages,
                    "response_text": "",
                }
                validation = self._validator.validate(event, candidate)
                self.store.store_candidate(candidate.to_record(), validation)
                if validation.get("status") == STATUS_ACTIVE:
                    self.store.upsert_memory_edge(candidate.to_record())
                    report.candidates_active += 1
                else:
                    report.candidates_rejected += 1
            report.time_validate = (time.perf_counter() - t5) * 1000

            # Stage 7: Graph inference
            t6 = time.perf_counter()
            self.store.run_inference()
            report.time_inference = (time.perf_counter() - t6) * 1000

        except Exception as exc:
            report.errors.append(f"pipeline error: {exc}")
            return self._maybe_fallback(messages, project_id, session_id, app_id, report)

        report.time_total = (time.perf_counter() - t0) * 1000
        return report

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_dense_chunk(
        self, messages: list[dict], report: ExtractionReport
    ) -> list[TextSegment]:
        if not self.enable_dense_chunk:
            full_text = "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}"
                for m in messages
            )
            segments = [TextSegment(text=full_text, start_idx=0, end_idx=len(full_text), density_score=0.5)]
            report.segments_dense = len(segments)
            return segments

        embed_fn = self._make_embed_fn()
        segments = self._dense_chunker.chunk(
            messages,
            self._indicator_embeddings,
            embed_fn,
        )
        report.segments_dense = len(segments)
        return segments

    def _run_contrastive(
        self, segments: list[TextSegment], report: ExtractionReport
    ) -> list[RelevantSegment]:
        if not self.enable_contrastive or self._retriever is None:
            relevant = [
                RelevantSegment(
                    text=s.text,
                    novelty_score=0.5,
                    contradiction_flag=False,
                    related_candidate_ids=[],
                    segment_embedding=np.zeros(1024, dtype=np.float32),
                )
                for s in segments
            ]
            report.segments_relevant = len(relevant)
            return relevant

        embed_fn = self._make_batch_embed_fn()
        relevant = self._retriever.retrieve(segments, self._snapshot, embed_fn)
        report.segments_relevant = len(relevant)
        return relevant

    def _run_classify(
        self, seg: RelevantSegment, report: ExtractionReport
    ) -> ClassifiedSegment | None:
        if not self.enable_classify:
            # Without classification, assume all types possible
            return ClassifiedSegment(
                text=seg.text,
                types=["fact", "decision", "preference"],
                spans=[(0, 100)] * 3,
                confidence=0.5,
                raw_response="classification disabled",
            )

        return self._classifier.classify(seg.text)

    def _run_extract(
        self, classified: ClassifiedSegment, report: ExtractionReport
    ) -> list[ExtractedCandidate]:
        return self._extractor.extract(classified.text, classified.types)

    def _run_consistency(
        self, classified: ClassifiedSegment, report: ExtractionReport
    ) -> list[ConsensusCandidate]:
        return self._consistency.extract(classified.text, classified.types)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _ensure_components(self, project_id: str, session_id: str) -> None:
        """Lazy-init all pipeline components."""
        gen = self._get_generation_fn()
        embed_batch = self._make_batch_embed_fn()

        # Dense Chunker
        if self._dense_chunker is None:
            self._dense_chunker = DenseChunker()

        # Indicator embeddings (computed once)
        if self._indicator_embeddings is None:
            self._indicator_embeddings = build_indicator_embeddings(embed_batch)

        # Contrastive Retriever + snapshot
        if self._retriever is None and self.enable_contrastive:
            self._snapshot, self._retriever = build_contrastive_pipeline(
                self.store, project_id, session_id, self.embedding_model
            )

        # Classifier
        if self._classifier is None:
            self._classifier = SlotClassifier(
                model_name=self.extraction_model,
                generation_fn=gen,
            )

        # Extractor
        if self._extractor is None:
            self._extractor = SlotExtractor(
                model_name=self.extraction_model,
                generation_fn=gen,
            )

        # Consistency
        if self._consistency is None:
            self._consistency = SelfConsistencyExtractor(
                model_name=self.extraction_model,
                generation_fn=gen,
            )

        # Validator
        if self._validator is None:
            self._validator = MemoryValidator(self.store)

    def _get_generation_fn(self) -> Callable:
        """Lazy load the MLX generation function."""
        if self._generation_fn is None:
            from ppmlx.engine import get_engine
            engine = get_engine()
            def gen(model, messages, max_tokens, temperature):
                result = engine.generate(
                    model, messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=False,
                )
                return result.text if hasattr(result, "text") else str(result[0])
            self._generation_fn = gen
        return self._generation_fn

    def _make_embed_fn(self) -> Callable[[str], np.ndarray]:
        """Make a single-text embedding function with error handling."""
        batch_fn = self._make_batch_embed_fn()
        def embed_one(text: str) -> np.ndarray:
            try:
                results = batch_fn([text])
                return results[0] if results else np.zeros(128, dtype=np.float32)
            except Exception:
                return np.zeros(128, dtype=np.float32)
        return embed_one

    def _make_batch_embed_fn(self) -> Callable[[list[str]], list[np.ndarray]]:
        """Make a batch embedding function using ppmlx EmbedEngine."""
        from ppmlx.engine_embed import get_embed_engine
        embed_engine = get_embed_engine()
        model = self.embedding_model
        def embed_batch(texts: list[str]) -> list[np.ndarray]:
            if not texts:
                return []
            try:
                vectors = embed_engine.encode(model, texts, normalize=True)
                return [np.array(v, dtype=np.float32) for v in vectors]
            except Exception:
                # Fallback: return zero vectors on embedding failure
                return [np.zeros(128, dtype=np.float32) for _ in texts]
        return embed_batch

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _maybe_fallback(
        self,
        messages: list[dict],
        project_id: str,
        session_id: str,
        app_id: str | None,
        report: ExtractionReport,
    ) -> ExtractionReport:
        """If decomposed pipeline fails, try v1 single-pass extraction."""
        if not self.fallback_to_v1:
            return report

        try:
            engine = MemoryEngine(store=self.store)
            result = engine.capture_chat(
                request_id=f"{session_id}-fallback",
                endpoint="/v1/chat/completions",
                model_alias=self.extraction_model,
                model_repo=f"mlx-community/{self.extraction_model}",
                messages=messages,
                response_text="",
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            )
            report.candidates_extracted = result.get("candidates", 0)
            report.candidates_active = result.get("active", 0)
            report.used_fallback = True
        except Exception as exc:
            report.errors.append(f"fallback also failed: {exc}")

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_shadow_candidate(ec: ExtractedCandidate) -> ShadowMemoryCandidate | None:
        """Convert ExtractedCandidate to ShadowMemoryCandidate."""
        if not ec.subject or not ec.predicate or not ec.object:
            return None
        return ShadowMemoryCandidate(
            type=ec.type,
            subject=ec.subject,
            predicate=ec.predicate,
            object=ec.object,
            text=ec.text,
            scope=ec.scope,
            confidence=ec.confidence,
            source_quote=ec.source_quote,
            salience=ec.salience,
            metadata=ec.metadata,
        )
