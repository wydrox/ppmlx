# ppmlx/contrastive_retriever.py — implementation spec

"""
Contrastive Retriever: filters conversation segments to keep only novel or
contradictory content relative to the current memory state.

All computation is local: numpy for vector ops, ppmlx EmbedEngine for embeddings.
Zero external services. Zero additional model loads (shares embedding model with
the dense chunker).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import re
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TextSegment:
    """A dense chunk from the conversation."""
    text: str
    start_idx: int          # character offset in original transcript
    end_idx: int
    density_score: float    # from DenseChunker


@dataclass
class MemorySnapshot:
    """Current state of the memory graph, loaded once per session."""
    candidates: list[dict]              # active candidates with text, type, subject, object, etc.
    candidate_embeddings: dict[str, np.ndarray]  # candidate_id → normalized embedding
    summary_text: str                   # compact rendering for diagnostic use


@dataclass
class RelevantSegment:
    """A segment that survived contrastive filtering."""
    text: str
    novelty_score: float                # 0.0 (identical to known) → 1.0 (completely new)
    contradiction_flag: bool            # True if segment signals contradiction of known fact
    related_candidate_ids: list[str]    # existing candidate_ids this segment relates to
    segment_embedding: np.ndarray       # cached for downstream (Slot Classifier)


# ---------------------------------------------------------------------------
# Contradiction signal patterns
# ---------------------------------------------------------------------------

# Lexical markers that suggest a fact is being updated/contradicted.
# Language-agnostic where possible; extend for Polish, German, etc.
_CONTRADICTION_PATTERNS: list[re.Pattern] = [
    # English
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bno longer\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bnot\s+\w+\s+but\b", re.IGNORECASE),
    re.compile(r"\bchanged\s+(?:my|our|the)\b", re.IGNORECASE),
    re.compile(r"\bupdate:\s", re.IGNORECASE),
    re.compile(r"\bcorrection:\s", re.IGNORECASE),
    re.compile(r"\bwrong[.,\s]", re.IGNORECASE),
    re.compile(r"\bthat['’]s\s+not\s+(?:right|correct|true)\b", re.IGNORECASE),
    re.compile(r"\bi\s+meant\b", re.IGNORECASE),
    # Polish
    re.compile(r"\bwłaściwie\b", re.IGNORECASE),
    re.compile(r"\bjednak\b", re.IGNORECASE),
    re.compile(r"\bzmieniamy\b", re.IGNORECASE),
    re.compile(r"\bzmieni(?:łem|łam|liśmy)\b", re.IGNORECASE),
    re.compile(r"\bpoprawka:\s", re.IGNORECASE),
    re.compile(r"\bnie\s+\w+\s+tylko\b", re.IGNORECASE),
    re.compile(r"\bto\s+nie\s+(?:jest|działa)\b", re.IGNORECASE),
]


def _has_contradiction_signal(text: str) -> bool:
    """Return True if the text contains lexical contradiction markers."""
    return any(p.search(text) for p in _CONTRADICTION_PATTERNS)


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """
    Tiny numpy-backed vector index. No FAISS, no Pinecone — just dot products.
    
    For a <1000-candidate memory graph, brute-force cosine similarity over
    numpy arrays is <1ms. No need for approximate nearest neighbors.
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None  # shape (N, D)
        self._metadata: list[dict] = []

    def add(self, vector: np.ndarray, metadata: dict) -> None:
        """Add a single vector with associated metadata (candidate_id, text, etc.)."""
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)  # normalize
        if self._vectors is None:
            self._vectors = v
        else:
            self._vectors = np.concatenate([self._vectors, v], axis=0)
        self._metadata.append(metadata)

    def add_batch(self, vectors: list[np.ndarray], metadatas: list[dict]) -> None:
        """Add multiple vectors at once (more efficient)."""
        for v, m in zip(vectors, metadatas):
            self.add(v, m)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[float, dict]]:
        """
        Return top_k most similar items as (cosine_similarity, metadata) pairs.
        Returns empty list if index is empty.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-8)  # normalize
        similarities = np.dot(self._vectors, q)  # shape (N,)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(float(similarities[i]), self._metadata[i]) for i in top_indices]

    def __len__(self) -> int:
        return len(self._metadata)


# ---------------------------------------------------------------------------
# Contrastive Retriever
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveRetriever:
    """
    Filters dense conversation segments to keep only novel or contradictory content.
    
    Algorithm:
    1. Embed each segment using the shared embedding model
    2. For each segment, find top-K most similar existing memory candidates
    3. Novelty = 1.0 - max_similarity
    4. If novelty is high (>threshold): KEEP (clearly new)
    5. If novelty is low (<threshold_low): check for contradiction signals
       - If contradiction: KEEP (potential update to known fact)
       - If no contradiction: DROP (just restating known fact)
    6. If novelty is medium: check contradiction + confidence of related facts
       - If related facts have low confidence: KEEP (reinforcement opportunity)
       - Otherwise: DROP
    
    The embedding model is only loaded once and shared with the DenseChunker.
    Embeddings for existing candidates are cached in MemorySnapshot.
    """
    
    # Thresholds (tunable)
    novelty_keep: float = 0.70       # Above this: definitely new → KEEP
    novelty_drop: float = 0.30       # Below this: very similar to known → DROP (unless contradiction)
    relevance_floor: float = 0.15    # Below this: no relation to any known fact → DROP (noise)
    contradiction_boost: float = 0.20  # How much contradiction lowers the effective novelty threshold
    low_confidence_threshold: float = 0.70  # Below this confidence, reinforcement is valuable
    
    top_k_similar: int = 5           # How many top candidates to compare against
    
    embedding_model: str = "nomic-embed-text"  # Small, fast embedding model
    
    def retrieve(
        self,
        segments: list[TextSegment],
        snapshot: MemorySnapshot,
        embed_fn: Callable[[list[str]], list[np.ndarray]],
    ) -> list[RelevantSegment]:
        """
        Main entry point.
        
        Args:
            segments: Dense chunks from DenseChunker
            snapshot: Current memory state with cached embeddings
            embed_fn: Function that takes list[str] and returns list[np.ndarray]
        
        Returns:
            Filtered list of RelevantSegment (only novel or contradictory ones)
        """
        # 1. Embed all segments in one batch (more efficient than one-by-one)
        segment_texts = [s.text for s in segments]
        segment_vectors = embed_fn(segment_texts)
        
        # 2. Build index from cached candidate embeddings if not already done
        if not hasattr(self, '_candidate_index') or self._candidate_index is None:
            self._candidate_index = EmbeddingIndex()
        
        # 3. Score and filter each segment
        relevant: list[RelevantSegment] = []
        for seg, vec in zip(segments, segment_vectors):
            result = self._score_segment(seg, vec, snapshot)
            if result is not None:
                relevant.append(result)
        
        return relevant
    
    def _score_segment(
        self,
        segment: TextSegment,
        embedding: np.ndarray,
        snapshot: MemorySnapshot,
    ) -> RelevantSegment | None:
        """
        Score a single segment. Returns RelevantSegment if it passes filters,
        None if it should be dropped.
        """
        # Search existing candidates for similar content
        top_hits = self._candidate_index.search(embedding, top_k=self.top_k_similar)
        
        # Empty index: no existing memories → all segments are novel → KEEP
        if not top_hits:
            return RelevantSegment(
                text=segment.text,
                novelty_score=1.0,
                contradiction_flag=False,
                related_candidate_ids=[],
                segment_embedding=embedding,
            )
        
        # No similar candidates at all → check if it's even relevant to our domain
        if not top_hits or top_hits[0][0] < self.relevance_floor:
            # This segment has no similarity to ANY known fact.
            # It's either noise or from a completely different domain.
            # Drop it — downstream extraction would have no context to anchor it.
            return None
        
        best_similarity, best_meta = top_hits[0]
        novelty = 1.0 - best_similarity
        
        # Collect related candidate IDs for traceability
        related_ids = [hit[1].get("candidate_id", "") for hit in top_hits[:3]]
        related_ids = [cid for cid in related_ids if cid]
        
        # Decision logic
        if novelty >= self.novelty_keep:
            # Clearly new information → KEEP
            return RelevantSegment(
                text=segment.text,
                novelty_score=novelty,
                contradiction_flag=False,
                related_candidate_ids=related_ids,
                segment_embedding=embedding,
            )
        
        # Check for contradiction signals
        has_contradiction = _has_contradiction_signal(segment.text)
        
        if novelty < self.novelty_drop:
            # Very similar to known fact.
            if has_contradiction:
                # Explicitly contradictory → KEEP (supersession candidate)
                return RelevantSegment(
                    text=segment.text,
                    novelty_score=novelty,
                    contradiction_flag=True,
                    related_candidate_ids=related_ids,
                    segment_embedding=embedding,
                )
            # Check reinforcement: restating a weak fact adds confidence.
            related_confidence = best_meta.get("confidence", 0.0)
            if related_confidence < self.low_confidence_threshold:
                return RelevantSegment(
                    text=segment.text,
                    novelty_score=novelty,
                    contradiction_flag=False,
                    related_candidate_ids=related_ids,
                    segment_embedding=embedding,
                )
            # Just restating a high-confidence known fact → DROP
            return None
        
        # Medium novelty (0.30–0.70): ambiguous.
        if has_contradiction:
            # Contradiction signal + medium similarity → likely update → KEEP
            return RelevantSegment(
                text=segment.text,
                novelty_score=novelty,
                contradiction_flag=True,
                related_candidate_ids=related_ids,
                segment_embedding=embedding,
            )
        
        # Check if related facts have low confidence (reinforcement opportunity)
        related_confidence = best_meta.get("confidence", 0.0)
        if related_confidence < self.low_confidence_threshold:
            return RelevantSegment(
                text=segment.text,
                novelty_score=novelty,
                contradiction_flag=False,
                related_candidate_ids=related_ids,
                segment_embedding=embedding,
            )
        
        # Medium novelty, no contradiction, related facts are confident → DROP
        return None
    
    def build_candidate_index(self, snapshot: MemorySnapshot) -> None:
        """
        Build the search index from snapshot's candidate embeddings.
        Called once per session after loading memory state.
        """
        self._candidate_index = EmbeddingIndex()
        for candidate in snapshot.candidates:
            cid = candidate.get("candidate_id", "")
            if cid in snapshot.candidate_embeddings:
                self._candidate_index.add(
                    snapshot.candidate_embeddings[cid],
                    {
                        "candidate_id": cid,
                        "type": candidate.get("type", ""),
                        "subject": candidate.get("subject", ""),
                        "predicate": candidate.get("predicate", ""),
                        "object": candidate.get("object", ""),
                        "confidence": candidate.get("confidence", 0.0),
                        "text": candidate.get("text", ""),
                    },
                )
    
    # ------------------------------------------------------------------
    # Persistence: cache candidate embeddings to avoid re-embedding
    # the entire memory graph on every extraction run.
    # ------------------------------------------------------------------
    
    def cache_candidate_embedding(
        self, candidate_id: str, embedding: np.ndarray, store
    ) -> None:
        """
        Store a candidate embedding so it survives process restarts.
        Uses the memory_store's entity_aliases table as a key-value store
        (entity_id → embedding vector serialized as JSON metadata).
        
        This is a pragmatic shortcut — a proper vector store would be better,
        but for <1000 candidates, JSON in SQLite is fast enough.
        """
        import json
        # Store as alias with a special type marker
        store.store_entity_alias({
            "entity_id": f"embedding:{candidate_id}",
            "alias": "embedding_vector",
            "type": "embedding_cache",
            "scope": "system",
            "confidence": 1.0,
            "metadata": {
                "vector": embedding.tolist(),
                "candidate_id": candidate_id,
                "model": self.embedding_model,
            },
        })
    
    def load_cached_embeddings(self, store) -> dict[str, np.ndarray]:
        """
        Load cached embeddings from the store.
        Returns {candidate_id: embedding_vector}.
        """
        import json
        aliases = store.query_entity_aliases(
            type="embedding_cache",
            scope="system",
            active_only=True,
            limit=10000,
        )
        result = {}
        for alias in aliases:
            meta = alias.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            cid = meta.get("candidate_id", "")
            vec_data = meta.get("vector", [])
            if cid and vec_data:
                result[cid] = np.array(vec_data, dtype=np.float32)
        return result


# ---------------------------------------------------------------------------
# Factory function: build the full snapshot + retriever pipeline
# ---------------------------------------------------------------------------

def build_contrastive_pipeline(
    store,
    project_id: str,
    session_id: str,
    embedding_model: str = "nomic-embed-text",
) -> tuple[MemorySnapshot, ContrastiveRetriever]:
    """
    One-shot setup: loads memory state, builds embedding index, returns
    (snapshot, retriever) ready for filtering segments.
    """
    from ppmlx.memory_store import get_memory_store
    from ppmlx.engine_embed import get_embed_engine
    
    store = store or get_memory_store()
    embed_engine = get_embed_engine()
    
    # Load active candidates
    candidates = store.query_candidates(
        status="active", project_id=project_id, session_id=session_id, limit=200
    )
    global_candidates = store.query_candidates(
        status="active", scope="global", limit=50
    )
    all_candidates = candidates + global_candidates
    
    # Build embedding function
    def embed_batch(texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = embed_engine.encode(embedding_model, texts, normalize=True)
        return [np.array(v, dtype=np.float32) for v in vectors]
    
    # Load cached embeddings or compute new ones
    retriever = ContrastiveRetriever(embedding_model=embedding_model)
    cached = retriever.load_cached_embeddings(store)
    
    # Compute missing embeddings (only for new candidates)
    candidate_embeddings: dict[str, np.ndarray] = {}
    missing_texts: list[str] = []
    missing_ids: list[str] = []
    
    for c in all_candidates:
        cid = c["candidate_id"]
        if cid in cached:
            candidate_embeddings[cid] = cached[cid]
        else:
            text = c.get("text", "")
            if text:
                missing_texts.append(text)
                missing_ids.append(cid)
    
    # Batch-embed missing candidates
    if missing_texts:
        new_vectors = embed_batch(missing_texts)
        for cid, vec in zip(missing_ids, new_vectors):
            candidate_embeddings[cid] = vec
            retriever.cache_candidate_embedding(cid, vec, store)
    
    # Build summary text
    summary_lines = []
    for c in all_candidates[-40:]:  # Most recent 40
        summary_lines.append(f"[{c['scope']}] {c['type']}: {c['text']}")
    
    snapshot = MemorySnapshot(
        candidates=all_candidates,
        candidate_embeddings=candidate_embeddings,
        summary_text="\n".join(summary_lines),
    )
    
    retriever.build_candidate_index(snapshot)
    
    return snapshot, retriever
