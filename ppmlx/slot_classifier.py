# ppmlx/slot_classifier.py — lightweight multi-label type classifier
"""
Slot Classifier: for each relevant segment, classify what memory types are present.

One small LLM call per segment (~200ms). Multi-label output: a segment can
contain multiple types (e.g., a decision AND a todo). Uses the same generation
pattern as ModelMemoryJsonExtractor.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import json

from ppmlx.memory_engine import ALLOWED_TYPES


ALLOWED_CLASSIFIER_TYPES = sorted(ALLOWED_TYPES) + ["none"]
_CLASSIFIER_MODEL = "gemma-4-e2b"


@dataclass
class ClassifiedSegment:
    """A segment with type labels and their text spans."""
    text: str
    types: list[str]           # e.g., ["decision", "todo"]
    spans: list[tuple[int, int]]  # character offsets (approximate)
    confidence: float
    raw_response: str = field(repr=False)


class SlotClassifier:
    """
    Classify a conversation segment into memory types.

    Uses a small LLM call with constrained JSON output. This is a discriminative
    task (what types are present?) rather than generative (extract S-P-O), so
    it's reliable even for 2B models.
    """

    def __init__(
        self,
        model_name: str = _CLASSIFIER_MODEL,
        generation_fn: Callable | None = None,
        max_tokens: int = 80,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.generation_fn = generation_fn or _default_generate
        self.max_tokens = max_tokens
        self.temperature = temperature

    def classify(self, segment_text: str) -> ClassifiedSegment:
        """
        Classify a single segment. Returns ClassifiedSegment with type labels.
        """
        prompt = _build_classification_prompt(segment_text)
        raw = self.generation_fn(
            self.model_name,
            [{"role": "user", "content": prompt}],
            self.max_tokens,
            self.temperature,
        )
        return _parse_classification(raw, segment_text)


def _build_classification_prompt(segment_text: str) -> str:
    types_str = ", ".join(ALLOWED_CLASSIFIER_TYPES)
    return f"""Classify this conversation segment. Return ONLY a JSON array of applicable types.

Types: {types_str}

Definitions:
  fact — a statement about what is true
  preference — someone's stated preference
  decision — a choice was made
  todo — an action item or next step
  constraint — a limitation, budget, or requirement
  instruction — a directive about how to behave
  relationship — a connection between entities
  entity_note — a note about a specific entity (file, tool, etc.)
  workflow_state — current task, blocker, or action in progress
  none — no extractable memory in this segment

Segment:
{segment_text[:1500]}

Types found:"""


def _parse_classification(raw: str, segment_text: str) -> ClassifiedSegment:
    """Parse the model's JSON array output. Defensive — handles markdown, prose, etc."""
    raw = raw.strip()

    # Extract JSON array from the response
    types: list[str] = []
    confidence = 0.0

    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            types = [t for t in parsed if isinstance(t, str)]
        elif isinstance(parsed, dict) and "types" in parsed:
            types = [t for t in parsed["types"] if isinstance(t, str)]
    except json.JSONDecodeError:
        # Try to find array in the text
        import re
        m = re.search(r"\[(.*?)\]", raw, re.DOTALL)
        if m:
            inner = m.group(1)
            types = [t.strip().strip("'\"") for t in inner.split(",")]
            types = [t for t in types if t]

    # Filter to allowed types
    allowed = set(ALLOWED_TYPES) | {"none"}
    types = [t.lower() for t in types if t.lower() in allowed]

    if not types:
        types = ["none"]

    # Confidence: if the model returned a clean JSON array, high confidence.
    # If we had to regex-extract it, lower.
    if raw.startswith("[") and raw.endswith("]"):
        confidence = 0.9
    elif types[0] == "none" and len(types) == 1:
        confidence = 0.85  # Model was confident there's nothing here
    else:
        confidence = 0.6

    # Approximate spans (simple — we don't ask the model for precise spans)
    spans = [(0, min(100, len(segment_text)))] * len(types)

    return ClassifiedSegment(
        text=segment_text,
        types=types,
        spans=spans,
        confidence=confidence,
        raw_response=raw,
    )


def _default_generate(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Default generation using local MLX engine."""
    from ppmlx.engine import get_engine
    result = get_engine().generate(
        model_name,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=False,
    )
    return result.text if hasattr(result, "text") else str(result[0])
