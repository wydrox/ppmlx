# ppmlx/dense_chunker.py — pure-code dense segment extraction
"""
Dense Chunker: finds the ~20% of a conversation that contains extractable facts.

Uses sliding windows + information density scoring. No model calls — just
embeddings (shared with Contrastive Retriever), regex, and statistics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import re
import numpy as np


# ---------------------------------------------------------------------------
# Fact-indicator phrases — embedded once at startup, used for cosine similarity
# ---------------------------------------------------------------------------

FACT_INDICATOR_PHRASES: list[str] = [
    # English
    "we decided to",
    "I prefer",
    "the plan is to",
    "next step is",
    "todo:",
    "remember that",
    "important: this",
    "key decision:",
    "the architecture is",
    "we use",
    "constraint:",
    "budget is",
    "deadline:",
    "goal:",
    "the problem is",
    "we need to",
    "should be",
    "must have",
    "requires",
    "depends on",
    # Polish
    "zdecydowaliśmy",
    "wolę",
    "plan jest",
    "następny krok",
    "do zrobienia:",
    "pamiętaj że",
    "ważne:",
    "architektura jest",
    "używamy",
    "ograniczenie:",
    "budżet to",
    "termin:",
    "cel:",
    "problem polega na",
    "musimy",
    "powinno być",
    "wymaga",
    "zależy od",
]


@dataclass
class TextSegment:
    """A dense chunk from the conversation, ready for downstream processing."""
    text: str
    start_idx: int
    end_idx: int
    density_score: float


# ---------------------------------------------------------------------------
# Scoring heuristics (no model calls)
# ---------------------------------------------------------------------------

# Common technical terms — high entity density signal
_TECH_TERMS = re.compile(
    r"\b("
    r"mlx|sqlite|api|cli|llm|gpu|ram|token|embedding|model|inference|"
    r"pipeline|graph|edge|node|schema|index|query|endpoint|server|worker|"
    r"extraction|validation|storage|cache|config|env|docker|kubernetes|"
    r"ppmlx|gemma|ollama|qwen|huggingface|"
    r"baza|zapytanie|serwer|model|pipeline|graf|węzeł|krawędź"
    r")\b",
    re.IGNORECASE,
)

# Code/stack-trace patterns — penalized (low extractable fact density)
_CODE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s{4,}\S", re.MULTILINE),           # indented code
    re.compile(r"```"),                                  # markdown code fences
    re.compile(r"Traceback\s", re.IGNORECASE),          # Python traceback
    re.compile(r"Error:\s", re.IGNORECASE),             # error messages
    re.compile(r"^\s*(?:import|from|def|class|fn|const|let|var)\s", re.MULTILINE),  # code
    re.compile(r"/\S+\.(?:py|ts|tsx|js|go|rs):\d+"),   # file:line references
    re.compile(r"(?:npm|pnpm|yarn|uv|cargo|pip)\s", re.IGNORECASE),  # package commands
]

# Conversational filler — low information density
_FILLER_PATTERNS = re.compile(
    r"\b("
    r"okay|ok|thanks|thank you|got it|understood|sure|yes|no|maybe|"
    r"hello|hi|hey|bye|see you|"
    r"oka?j|dzięki|dziękuję|rozumiem|jasne|tak|nie|może|cześć|hej|pa"
    r")\b",
    re.IGNORECASE,
)


def _lexical_diversity(text: str) -> float:
    """unique tokens / total tokens. Higher = more information-dense."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < 5:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _entity_density(text: str) -> float:
    """Count of technical terms / window length. Higher = more domain content."""
    matches = len(_TECH_TERMS.findall(text))
    words = max(1, len(text.split()))
    return min(1.0, matches / max(1, words / 20))  # normalize: 1 term per 20 words = score 1.0


def _code_penalty(text: str) -> float:
    """Penalize windows that look like code blocks or stack traces."""
    penalty = 0.0
    for pattern in _CODE_PATTERNS:
        if pattern.search(text):
            penalty += 0.15
    return min(1.0, penalty)


def _filler_penalty(text: str) -> float:
    """Penalize windows heavy on conversational filler."""
    filler_count = len(_FILLER_PATTERNS.findall(text))
    words = max(1, len(text.split()))
    return min(1.0, filler_count / max(1, words / 5))


def _fact_signal_score(
    text: str,
    indicator_embeddings: list[np.ndarray],
    embed_fn: Callable[[str], np.ndarray],
) -> float:
    """
    Max cosine similarity of the window's embedding to the fact-indicator bank.
    
    Windows semantically similar to known fact-indicator phrases ("we decided",
    "next step", "I prefer") score higher — they likely contain extractable facts.
    """
    if not indicator_embeddings:
        return 0.0
    vec = embed_fn(text)
    similarities = [float(np.dot(vec, iv)) for iv in indicator_embeddings]
    return max(similarities) if similarities else 0.0


# ---------------------------------------------------------------------------
# Dense Chunker
# ---------------------------------------------------------------------------

class DenseChunker:
    """
    Sliding-window information density scorer.
    
    1. Split transcript into overlapping windows
    2. Score each window: fact_signal + entity_density + lexical_diversity - penalties
    3. Select top 20% windows
    4. Merge adjacent selected windows
    5. Expand boundaries for context
    """

    def __init__(
        self,
        window_tokens: int = 400,
        stride_tokens: int = 100,
        top_k_ratio: float = 0.20,
        min_segments: int = 1,
        max_segments: int = 12,
        # Scoring weights
        w_fact_signal: float = 0.35,
        w_entity_density: float = 0.25,
        w_lexical_diversity: float = 0.20,
        w_code_penalty: float = 0.10,
        w_filler_penalty: float = 0.10,
    ):
        self.window_tokens = window_tokens
        self.stride_tokens = stride_tokens
        self.top_k_ratio = top_k_ratio
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.w_fact_signal = w_fact_signal
        self.w_entity_density = w_entity_density
        self.w_lexical_diversity = w_lexical_diversity
        self.w_code_penalty = w_code_penalty
        self.w_filler_penalty = w_filler_penalty

    def chunk(
        self,
        messages: list[dict],
        indicator_embeddings: list[np.ndarray],
        embed_fn: Callable[[str], np.ndarray],
    ) -> list[TextSegment]:
        """
        Extract dense segments from conversation messages.
        
        Args:
            messages: list of {"role": str, "content": str}
            indicator_embeddings: pre-computed embeddings of fact-indicator phrases
            embed_fn: text → normalized embedding vector
        
        Returns:
            list of TextSegment, ordered by position in transcript
        """
        # Convert messages to flat text with position tracking
        full_text, char_map = self._flatten_messages(messages)
        if not full_text.strip():
            return []

        # Estimate tokens: ~4 chars per token
        window_chars = self.window_tokens * 4
        stride_chars = self.stride_tokens * 4

        # Generate windows
        windows: list[tuple[str, int, int]] = []  # (text, char_start, char_end)
        for start in range(0, len(full_text), stride_chars):
            end = min(start + window_chars, len(full_text))
            window_text = full_text[start:end]
            if len(window_text.strip()) < 50:  # skip near-empty windows
                continue
            windows.append((window_text, start, end))

        if not windows:
            return []

        # Pre-compute embeddings for all windows (one call per window, but avoids
        # redundant embedding in the scoring loop body).
        window_texts = [w[0] for w in windows]
        window_embeddings = [embed_fn(t) for t in window_texts]

        # Score each window (use pre-computed embeddings for fact_signal)
        scores: list[tuple[int, float]] = []  # (window_index, density_score)
        for idx, (text, start, end) in enumerate(windows):
            score = self._score_window(text, indicator_embeddings, window_embeddings[idx])
            scores.append((idx, score))

        # Select top K
        k = max(self.min_segments, min(self.max_segments, int(len(windows) * self.top_k_ratio)))
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = {idx for idx, _ in scores[:k]}

        # Merge adjacent selected windows (gap ≤ 2 strides)
        merged: list[list[int]] = []
        for idx in sorted(selected_indices):
            if merged and idx - merged[-1][-1] <= 2:
                merged[-1].append(idx)
            else:
                merged.append([idx])

        # Build segments from merged windows, expand boundaries
        segments: list[TextSegment] = []
        for group in merged:
            first_win = windows[group[0]]
            last_win = windows[group[-1]]
            
            # Expand boundaries by ±half stride for context
            expand = stride_chars // 2
            seg_start = max(0, first_win[1] - expand)
            seg_end = min(len(full_text), last_win[2] + expand)
            
            seg_text = full_text[seg_start:seg_end].strip()
            if not seg_text:
                continue
            
            # Average density score for the group
            avg_score = sum(s[1] for s in scores if s[0] in group) / len(group)
            
            segments.append(TextSegment(
                text=seg_text,
                start_idx=seg_start,
                end_idx=seg_end,
                density_score=round(avg_score, 4),
            ))

        return segments

    def _score_window(
        self,
        text: str,
        indicator_embeddings: list[np.ndarray],
        precomputed_embedding: np.ndarray,
    ) -> float:
        """Compute information density score for a text window."""
        fact = 0.0
        if indicator_embeddings and precomputed_embedding is not None:
            similarities = [float(np.dot(precomputed_embedding, iv)) for iv in indicator_embeddings]
            fact = max(similarities) if similarities else 0.0
        
        entity = _entity_density(text)
        diversity = _lexical_diversity(text)
        code_p = _code_penalty(text)
        filler_p = _filler_penalty(text)

        score = (
            self.w_fact_signal * fact
            + self.w_entity_density * entity
            + self.w_lexical_diversity * diversity
            - self.w_code_penalty * code_p
            - self.w_filler_penalty * filler_p
        )
        return max(0.0, score)

    @staticmethod
    def _flatten_messages(messages: list[dict]) -> tuple[str, dict[int, dict]]:
        """
        Convert messages to flat conversation text.
        Returns (text, char_map) where char_map[char_position] = message metadata.
        """
        lines: list[str] = []
        char_map: dict[int, dict] = {}
        pos = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if not content.strip():
                continue
            
            prefix = f"{role}: "
            line = prefix + content
            lines.append(line)
            
            for cp in range(pos, pos + len(line)):
                char_map[cp] = {"role": role, "message_idx": len(lines) - 1}
            
            pos += len(line) + 1  # +1 for newline

        return "\n".join(lines), char_map


# ---------------------------------------------------------------------------
# Factory: pre-compute fact indicator embeddings (call once at startup)
# ---------------------------------------------------------------------------

def build_indicator_embeddings(
    embed_fn: Callable[[list[str]], list[np.ndarray]],
    phrases: list[str] | None = None,
) -> list[np.ndarray]:
    """
    Embed the fact-indicator phrase bank. Call once, cache the result.
    
    Args:
        embed_fn: batch embedding function (list[str] → list[np.ndarray])
        phrases: override default phrase list
    
    Returns:
        list of normalized embedding vectors
    """
    phrases = phrases or FACT_INDICATOR_PHRASES
    vectors = embed_fn(phrases)
    # Normalize
    return [v / (np.linalg.norm(v) + 1e-8) for v in vectors]
