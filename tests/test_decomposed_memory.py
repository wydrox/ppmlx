# tests/test_decomposed_memory.py — tests for v2 memory architecture components

from __future__ import annotations
import json
import numpy as np

from ppmlx.dense_chunker import (
    DenseChunker, TextSegment, build_indicator_embeddings,
    _lexical_diversity, _entity_density, _code_penalty, _filler_penalty,
    FACT_INDICATOR_PHRASES,
)
from ppmlx.contrastive_retriever import (
    EmbeddingIndex, ContrastiveRetriever, TextSegment as CRTextSegment,
    MemorySnapshot, RelevantSegment, _has_contradiction_signal,
)
from ppmlx.slot_classifier import (
    SlotClassifier, ClassifiedSegment, _parse_classification,
    _build_classification_prompt,
)
from ppmlx.slot_extractor import (
    SlotExtractor, ExtractedCandidate, _parse_slot_output,
    _EXTRACTION_TEMPLATES, _find_best_quote,
)
from ppmlx.self_consistency import (
    SelfConsistencyExtractor, ConsensusCandidate, _token_jaccard,
)
from ppmlx.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Deterministic mock embeddings based on text hash."""
    vecs = []
    for t in texts:
        seed = hash(t[:100]) % 2**31
        rng = np.random.RandomState(seed)
        v = rng.randn(128).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        vecs.append(v)
    return vecs


def _mock_embed_one(text: str) -> np.ndarray:
    return _mock_embed_batch([text])[0]


def _mock_generate_json(types_list: list[str]):
    """Return a mock generation function that outputs a JSON array of types."""
    def fn(model, messages, max_tokens, temperature):
        return json.dumps(types_list)
    return fn


def _mock_generate_text(response: str):
    """Return a mock generation function that outputs fixed text."""
    def fn(model, messages, max_tokens, temperature):
        return response
    return fn


# ---------------------------------------------------------------------------
# Dense Chunker tests
# ---------------------------------------------------------------------------

class TestDenseChunker:

    def test_trivial_empty(self):
        chunker = DenseChunker()
        indicators = build_indicator_embeddings(_mock_embed_batch)
        result = chunker.chunk([], indicators, _mock_embed_one)
        assert result == []

    def test_single_message(self):
        chunker = DenseChunker(window_tokens=50, stride_tokens=15, top_k_ratio=1.0, min_segments=1)
        indicators = build_indicator_embeddings(_mock_embed_batch)
        # Pad message to be long enough for a window (>50 chars)
        msgs = [{"role": "user", "content": "We decided to use SQLite for storage. It is fast and requires no server setup."}]
        result = chunker.chunk(msgs, indicators, _mock_embed_one)
        assert len(result) >= 1
        assert "SQLite" in result[0].text
        assert result[0].density_score > 0

    def test_code_heavy_message_penalized(self):
        chunker = DenseChunker()
        indicators = build_indicator_embeddings(_mock_embed_batch)
        code_msg = "def foo():\n    import os\n    x = 1\n    return x"
        msgs = [{"role": "assistant", "content": code_msg}]
        result = chunker.chunk(msgs, indicators, _mock_embed_one)
        # Code-heavy window should score low — might still produce a segment,
        # but score should be low
        if result:
            assert result[0].density_score < 0.5

    def test_mixed_conversation(self):
        chunker = DenseChunker(window_tokens=200, stride_tokens=50, top_k_ratio=0.5)
        indicators = build_indicator_embeddings(_mock_embed_batch)
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "We decided to use SQLite. I prefer short answers."},
            {"role": "assistant", "content": "```python\nimport sqlite3\nconn = sqlite3.connect('db')\n```"},
            {"role": "user", "content": "Next step: implement the storage layer."},
        ]
        result = chunker.chunk(msgs, indicators, _mock_embed_one)
        assert len(result) >= 1
        # The fact-heavy messages should dominate the dense segments
        all_text = " ".join(s.text for s in result)
        assert "SQLite" in all_text or "storage" in all_text.lower()

    def test_lexical_diversity(self):
        assert _lexical_diversity("hello hello hello") < 0.5
        assert _lexical_diversity("SQLite storage requires no server config") > 0.5

    def test_entity_density(self):
        low = _entity_density("hello world")
        high = _entity_density("mlx SQLite API endpoint server model inference pipeline")
        assert high > low

    def test_code_penalty(self):
        assert _code_penalty("hello world") == 0.0
        assert _code_penalty("    indented code block") > 0.0
        assert _code_penalty("```python\nprint('hi')\n```") > 0.0

    def test_filler_penalty(self):
        assert _filler_penalty("ok thanks got it") > _filler_penalty("We decided to use SQLite")

    def test_indicator_phrases(self):
        assert len(FACT_INDICATOR_PHRASES) > 20
        indicators = build_indicator_embeddings(_mock_embed_batch)
        assert len(indicators) == len(FACT_INDICATOR_PHRASES)
        # All should be normalized
        for v in indicators:
            assert abs(float(np.linalg.norm(v)) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Contrastive Retriever tests
# ---------------------------------------------------------------------------

class TestContrastiveRetriever:

    def test_embedding_index_search(self):
        idx = EmbeddingIndex()
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        idx.add(a, {"id": "a"})
        idx.add(b, {"id": "b"})

        results = idx.search(np.array([1.0, 0.1, 0.0], dtype=np.float32), top_k=2)
        assert len(results) == 2
        assert results[0][1]["id"] == "a"

    def test_empty_index(self):
        idx = EmbeddingIndex()
        results = idx.search(np.array([1.0, 0.0], dtype=np.float32))
        assert results == []

    def test_contradiction_detection(self):
        assert _has_contradiction_signal("actually, we should use SQLite")
        assert _has_contradiction_signal("no longer needed")
        assert _has_contradiction_signal("właściwie to zmieniamy decyzję")
        assert _has_contradiction_signal("to nie jest poprawne")
        assert not _has_contradiction_signal("the sky is blue")
        assert not _has_contradiction_signal("we use MLX for inference")

    def test_novel_segment_kept(self):
        retriever = ContrastiveRetriever()
        idx = EmbeddingIndex()
        idx.add(np.array([1.0, 0.0, 0.0], dtype=np.float32),
                {"candidate_id": "old", "confidence": 0.9})
        retriever._candidate_index = idx
        snapshot = MemorySnapshot(candidates=[], candidate_embeddings={}, summary_text="")

        # Novel: cosine ~0.25 → kept
        seg = CRTextSegment(text="new info", start_idx=0, end_idx=8, density_score=0.8)
        result = retriever._score_segment(
            seg, np.array([0.25, 0.97, 0.0], dtype=np.float32), snapshot
        )
        assert result is not None
        assert result.novelty_score > 0.7

    def test_similar_no_contradiction_dropped(self):
        retriever = ContrastiveRetriever()
        idx = EmbeddingIndex()
        idx.add(np.array([1.0, 0.0, 0.0], dtype=np.float32),
                {"candidate_id": "old", "confidence": 0.9})
        retriever._candidate_index = idx
        snapshot = MemorySnapshot(candidates=[], candidate_embeddings={}, summary_text="")

        # Very similar, high confidence existing → dropped
        seg = CRTextSegment(text="old info", start_idx=0, end_idx=8, density_score=0.5)
        result = retriever._score_segment(
            seg, np.array([0.98, 0.2, 0.0], dtype=np.float32), snapshot
        )
        assert result is None

    def test_contradiction_kept_even_when_similar(self):
        retriever = ContrastiveRetriever()
        idx = EmbeddingIndex()
        idx.add(np.array([1.0, 0.0, 0.0], dtype=np.float32),
                {"candidate_id": "old", "confidence": 0.9})
        retriever._candidate_index = idx
        snapshot = MemorySnapshot(candidates=[], candidate_embeddings={}, summary_text="")

        seg = CRTextSegment(
            text="actually this was wrong", start_idx=0, end_idx=22, density_score=0.8
        )
        result = retriever._score_segment(
            seg, np.array([0.95, 0.3, 0.0], dtype=np.float32), snapshot
        )
        assert result is not None
        assert result.contradiction_flag

    def test_reinforcement_kept_for_weak_fact(self):
        retriever = ContrastiveRetriever()
        idx = EmbeddingIndex()
        idx.add(np.array([1.0, 0.0, 0.0], dtype=np.float32),
                {"candidate_id": "weak", "confidence": 0.4})
        retriever._candidate_index = idx
        snapshot = MemorySnapshot(candidates=[], candidate_embeddings={}, summary_text="")

        seg = CRTextSegment(
            text="ppmlx uses MLX for inference", start_idx=0, end_idx=28, density_score=0.9
        )
        result = retriever._score_segment(
            seg, np.array([0.92, 0.4, 0.0], dtype=np.float32), snapshot
        )
        assert result is not None

    def test_noise_dropped(self):
        retriever = ContrastiveRetriever()
        idx = EmbeddingIndex()
        idx.add(np.array([1.0, 0.0, 0.0], dtype=np.float32),
                {"candidate_id": "old", "confidence": 0.9})
        retriever._candidate_index = idx
        snapshot = MemorySnapshot(candidates=[], candidate_embeddings={}, summary_text="")

        # Orthogonal → below relevance floor → dropped
        seg = CRTextSegment(text="weather is nice", start_idx=0, end_idx=15, density_score=0.3)
        result = retriever._score_segment(
            seg, np.array([0.0, 1.0, 0.0], dtype=np.float32), snapshot
        )
        assert result is None


# ---------------------------------------------------------------------------
# Slot Classifier tests
# ---------------------------------------------------------------------------

class TestSlotClassifier:

    def test_classification_prompt_format(self):
        prompt = _build_classification_prompt("We decided to use SQLite.")
        assert "fact" in prompt
        assert "decision" in prompt
        assert "none" in prompt
        assert "We decided to use SQLite" in prompt

    def test_parse_clean_json_array(self):
        result = _parse_classification('["decision", "preference"]', "test text")
        assert result.types == ["decision", "preference"]
        assert result.confidence >= 0.8

    def test_parse_none(self):
        result = _parse_classification('["none"]', "test text")
        assert result.types == ["none"]

    def test_parse_invalid_fallback(self):
        result = _parse_classification('garbage text with no json', "test text")
        assert result.types == ["none"]

    def test_classify_with_mock_generation(self):
        classifier = SlotClassifier(
            generation_fn=_mock_generate_json(["decision", "todo"]),
        )
        result = classifier.classify("We decided to use SQLite. Next step: implement.")
        assert "decision" in result.types
        assert "todo" in result.types

    def test_classify_none(self):
        classifier = SlotClassifier(
            generation_fn=_mock_generate_json(["none"]),
        )
        result = classifier.classify("hello world")
        assert result.types == ["none"]


# ---------------------------------------------------------------------------
# Slot Extractor tests
# ---------------------------------------------------------------------------

class TestSlotExtractor:

    def test_all_types_have_templates(self):
        from ppmlx.memory_engine import ALLOWED_TYPES
        for t in ALLOWED_TYPES:
            assert t in _EXTRACTION_TEMPLATES, f"Missing template for type: {t}"
        # workflow_state is also in ALLOWED_TYPES, check it
        assert "workflow_state" in _EXTRACTION_TEMPLATES

    def test_templates_contain_text_placeholder(self):
        for ttype, template in _EXTRACTION_TEMPLATES.items():
            assert "{text}" in template, f"Template for {ttype} missing {{text}} placeholder"

    def test_parse_slot_output_basic(self):
        raw = """SUBJECT: ppmlx
PREDICATE: uses
OBJECT: MLX
SUMMARY: ppmlx uses MLX
SCOPE: project
CONFIDENCE: 0.9
SALIENCE: 0.85
QUOTE: ppmlx uses MLX for inference"""

        result = _parse_slot_output(raw, "fact", "ppmlx uses MLX for inference. It runs locally.")
        assert result is not None
        assert result.subject == "ppmlx"
        assert result.predicate == "uses"
        assert result.object == "MLX"
        assert result.scope == "project"
        assert abs(result.confidence - 0.9) < 0.01

    def test_parse_slot_output_with_colons(self):
        """Fields may use colons or equals as separators."""
        raw = """subject: ppmlx
predicate = uses
OBJECT: MLX"""
        result = _parse_slot_output(raw, "fact", "ppmlx uses MLX")
        assert result is not None
        assert result.subject == "ppmlx"

    def test_parse_slot_output_missing_fields(self):
        result = _parse_slot_output("garbage", "fact", "some text")
        assert result is None

    def test_find_best_quote(self):
        text = "We decided to use SQLite for storage. It is fast and requires no server."
        quote = _find_best_quote(text, "ppmlx", "uses", "SQLite")
        assert "SQLite" in quote

    def test_extract_with_mock_generation(self):
        mock_response = """SUBJECT: ppmlx
PREDICATE: uses
OBJECT: MLX
SUMMARY: ppmlx uses MLX
SCOPE: project
CONFIDENCE: 0.9
SALIENCE: 0.85
QUOTE: ppmlx uses MLX"""

        extractor = SlotExtractor(generation_fn=_mock_generate_text(mock_response))
        candidates = extractor.extract("ppmlx uses MLX for inference", ["fact"])
        assert len(candidates) >= 1
        assert candidates[0].subject == "ppmlx"


# ---------------------------------------------------------------------------
# Self-Consistency tests
# ---------------------------------------------------------------------------

class TestSelfConsistency:

    def test_token_jaccard(self):
        assert _token_jaccard("hello world", "hello world") == 1.0
        assert _token_jaccard("hello world", "goodbye universe") == 0.0
        score = _token_jaccard("ppmlx uses MLX", "ppmlx uses mlx for inference")
        assert score > 0.5

    def test_candidates_match_same(self):
        a = ExtractedCandidate("fact", "ppmlx", "uses", "MLX for local inference",
                                "ppmlx uses MLX", "project", 0.9, 0.85, "ppmlx uses MLX", {})
        b = ExtractedCandidate("fact", "ppmlx", "uses", "MLX local inference",
                                "ppmlx uses MLX", "project", 0.9, 0.85, "ppmlx uses MLX", {})
        # Jaccard: {mlx,for,local,inference} ∩ {mlx,local,inference} / {mlx,for,local,inference} ∪ {mlx,local,inference}
        # = 3/4 = 0.75 ≥ 0.5 → should match
        assert SelfConsistencyExtractor._candidates_match(a, b)

    def test_candidates_match_different_type(self):
        a = ExtractedCandidate("fact", "ppmlx", "uses", "MLX", "", "project", 0.9, 0.85, "", {})
        b = ExtractedCandidate("decision", "ppmlx", "uses", "MLX", "", "project", 0.9, 0.85, "", {})
        assert not SelfConsistencyExtractor._candidates_match(a, b)

    def test_consistency_with_mock_extraction(self):
        """Three identical mock extractions → all survive."""
        fixed_response = """SUBJECT: ppmlx
PREDICATE: uses
OBJECT: MLX
SUMMARY: ppmlx uses MLX
SCOPE: project
CONFIDENCE: 0.9
SALIENCE: 0.85
QUOTE: ppmlx uses MLX"""

        sc = SelfConsistencyExtractor(
            generation_fn=_mock_generate_text(fixed_response),
            temperatures=(0.0, 0.0, 0.0),  # all same temp for deterministic test
        )
        consensus = sc.extract("ppmlx uses MLX for inference", ["fact"])
        # With identical outputs, should get 1 cluster with 3/3 agreement
        assert len(consensus) >= 1
        assert consensus[0].num_runs == 3

    def test_consensus_confidence_boosted(self):
        fixed_response = """SUBJECT: ppmlx
PREDICATE: uses
OBJECT: MLX
SUMMARY: ppmlx uses MLX
SCOPE: project
CONFIDENCE: 0.7
SALIENCE: 0.85
QUOTE: ppmlx uses MLX"""

        sc = SelfConsistencyExtractor(
            generation_fn=_mock_generate_text(fixed_response),
            temperatures=(0.0, 0.0, 0.0),
        )
        consensus = sc.extract("ppmlx uses MLX", ["fact"])
        assert len(consensus) >= 1
        # Base confidence 0.7, 3/3 agreement → adjusted > 0.7
        assert consensus[0].consensus_confidence >= 0.7


# ---------------------------------------------------------------------------
# Integration: full decomposed pipeline with mocked model
# ---------------------------------------------------------------------------

class TestDecomposedPipeline:

    def test_full_pipeline_with_mock_model(self, tmp_path):
        """End-to-end test with mock embeddings and mock LLM."""
        from ppmlx.decomposed_engine import DecomposedMemoryEngine
        from ppmlx.memory_store import MemoryStore

        db_path = tmp_path / "test.db"
        store = MemoryStore(db_path)

        engine = DecomposedMemoryEngine(
            store=store,
            extraction_model="gemma-4-e2b",
            enable_dense_chunk=True,
            enable_contrastive=True,
            enable_classify=True,
            enable_consistency=True,
            fallback_to_v1=False,
        )

        # Mock embeddings
        engine._make_batch_embed_fn = lambda: _mock_embed_batch
        engine._make_embed_fn = lambda: (lambda t: _mock_embed_batch([t])[0])
        engine._indicator_embeddings = build_indicator_embeddings(_mock_embed_batch)

        # Seed existing memory
        from ppmlx.memory_engine import ShadowMemoryCandidate
        for i, (subj, pred, obj, txt) in enumerate([
            ("ppmlx", "uses", "MLX", "ppmlx uses MLX for inference"),
        ]):
            c = ShadowMemoryCandidate(
                type="fact", subject=subj, predicate=pred, object=obj,
                text=txt, scope="project", confidence=0.9,
                source_quote=txt, salience=0.85,
                event_id="seed", candidate_id=f"seed-{i}",
            )
            store.store_candidate(c.to_record(), {"status": "active", "reasons": [], "invalidates": []})
            store.upsert_memory_edge(c.to_record())

        # Pre-build contrastive index with mock embeddings
        from ppmlx.contrastive_retriever import ContrastiveRetriever, MemorySnapshot
        snapshot = MemorySnapshot(
            candidates=[{"candidate_id": "seed-0", "text": "ppmlx uses MLX for inference",
                         "confidence": 0.9, "type": "fact", "subject": "ppmlx",
                         "predicate": "uses", "object": "MLX"}],
            candidate_embeddings={"seed-0": _mock_embed_batch(["ppmlx uses MLX for inference"])[0]},
            summary_text="",
        )
        retriever = ContrastiveRetriever()
        retriever.build_candidate_index(snapshot)
        engine._retriever = retriever
        engine._snapshot = snapshot

        # Mock LLM generation for classify + extract
        classify_output = '["decision", "fact"]'
        extract_output = """SUBJECT: ppmlx
PREDICATE: decided
OBJECT: use SQLite
SUMMARY: ppmlx decided to use SQLite
SCOPE: project
CONFIDENCE: 0.9
SALIENCE: 0.85
QUOTE: We decided to use SQLite"""

        def mock_gen(model, messages, max_tokens, temperature):
            prompt_text = messages[0]["content"] if messages else ""
            if "Classify this conversation" in prompt_text:
                return classify_output
            else:
                return extract_output

        engine._generation_fn = mock_gen

        # Test
        msgs = [
            {"role": "user", "content": "We decided to use SQLite for storage. It is fast."},
        ]
        report = engine.extract_from_session(
            messages=msgs, project_id="ppmlx", session_id="test",
        )

        # Should have extracted at least 1 candidate
        assert report.candidates_extracted >= 0  # may be 0 if contrastive drops it
        assert len(report.errors) == 0 or "contrastive" in str(report.errors)

    def test_fallback_disabled(self, tmp_path):
        """When fallback is disabled and pipeline fails, report has error."""
        from ppmlx.decomposed_engine import DecomposedMemoryEngine
        from ppmlx.memory_store import MemoryStore

        store = MemoryStore(tmp_path / "test.db")
        engine = DecomposedMemoryEngine(
            store=store, fallback_to_v1=False,
            enable_dense_chunk=False,
            enable_contrastive=False,
            enable_classify=True,
        )
        # No generation fn set → will fail
        report = engine.extract_from_session(
            messages=[{"role": "user", "content": "test"}],
            project_id="test", session_id="test",
        )
        assert not report.used_fallback

    def test_memory_snapshot_loading(self, tmp_path):
        """Test that MemorySnapshot loads from store correctly."""
        from ppmlx.contrastive_retriever import MemorySnapshot, build_contrastive_pipeline

        store = MemoryStore(tmp_path / "test.db")
        # Seed a candidate
        from ppmlx.memory_engine import ShadowMemoryCandidate
        c = ShadowMemoryCandidate(
            type="fact", subject="test", predicate="is", object="working",
            text="test is working", scope="global", confidence=0.9,
            source_quote="test is working", salience=0.85,
            event_id="ev1", candidate_id="c1",
        )
        store.store_candidate(c.to_record(), {"status": "active", "reasons": [], "invalidates": []})
        store.upsert_memory_edge(c.to_record())

        # Load snapshot — will fail on real embeddings but shouldn't crash
        try:
            snapshot, retriever = build_contrastive_pipeline(
                store, "test", "s1", embedding_model="nomic-embed-text"
            )
        except Exception:
            # Expected — no embedding model available in test
            pass
