# ppmlx/slot_extractor.py — type-specific structured extraction
"""
Slot Extractor: for each classified segment, extract {subject, predicate, object}
using type-specific prompts with constrained fill-in-the-blanks format.

One LLM call per type in the segment. Each type has its own template that
guides the model to specific spans in the text rather than open-ended generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import re



_EXTRACTION_MODEL = "gemma-4-e2b"


@dataclass
class ExtractedCandidate:
    """A candidate produced by slot extraction."""
    type: str
    subject: str
    predicate: str
    object: str
    text: str
    scope: str
    confidence: float
    salience: float
    source_quote: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type-specific extraction templates
#
# Each template uses {text} placeholder and expects the model to fill in
# labeled fields. The format is: FIELD: value
# This is easier for small models than generating structured JSON.
# ---------------------------------------------------------------------------

_EXTRACTION_TEMPLATES: dict[str, str] = {
    "fact": """Extract a fact from this text. A fact is a true statement about an entity.

Text: {text}

SUBJECT (the entity the fact is about):
PREDICATE (the property or relation):
OBJECT (the value or target):
SUMMARY (one sentence):
SCOPE (global, project, or session):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0, how durable/important):
QUOTE (verbatim from text):
""",

    "preference": """Extract a preference from this text. A preference is what someone likes or wants.

Text: {text}

SUBJECT (who has the preference):
PREDICATE (use "prefers" unless there is a better verb):
OBJECT (what they prefer, be specific):
SUMMARY (one sentence):
SCOPE (usually "global" unless project-specific):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "decision": """Extract a decision from this text. A decision is a choice that was made.

Text: {text}

SUBJECT (who or what made the decision, e.g. a project name):
PREDICATE (use "decided" unless there is a better verb):
OBJECT (what was decided, be specific):
SUMMARY (one sentence):
SCOPE (project if project-specific, otherwise global):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "todo": """Extract a todo item from this text. A todo is an action item or next step.

Text: {text}

SUBJECT (who or what needs to do it, e.g. a project or person):
PREDICATE (use "needs" unless there is a better verb):
OBJECT (what needs to be done, be specific):
SUMMARY (one sentence):
SCOPE (project if project-specific, otherwise global):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "constraint": """Extract a constraint from this text. A constraint is a limitation or requirement.

Text: {text}

SUBJECT (what the constraint applies to):
PREDICATE (use "requires" or "budget" or "must" or "max" as appropriate):
OBJECT (the constraint value, be specific):
SUMMARY (one sentence):
SCOPE (usually project):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "instruction": """Extract an instruction from this text. An instruction tells how to behave.

Text: {text}

SUBJECT (who should follow the instruction):
PREDICATE (use "should" unless there is a better verb):
OBJECT (what the instruction says):
SUMMARY (one sentence):
SCOPE (session if temporary, global if permanent):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "relationship": """Extract a relationship from this text. A relationship connects two entities.

Text: {text}

SUBJECT (the first entity):
PREDICATE (the relationship, e.g. "depends_on", "uses", "contains"):
OBJECT (the second entity):
SUMMARY (one sentence):
SCOPE (project or global):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "entity_note": """Extract a note about an entity from this text.

Text: {text}

SUBJECT (the entity, e.g. a project or system):
PREDICATE (what happened to the entity, e.g. "file_changed", "version", "status"):
OBJECT (the value or detail):
SUMMARY (one sentence):
SCOPE (project or session):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",

    "workflow_state": """Extract workflow state from this text. Workflow state tracks current progress.

Text: {text}

SUBJECT (the project or session):
PREDICATE (use "current_task", "next_action", "blocker", or "command_run"):
OBJECT (the specific task, action, or blocker):
SUMMARY (one sentence):
SCOPE (usually project):
CONFIDENCE (0.0-1.0):
SALIENCE (0.0-1.0):
QUOTE (verbatim from text):
""",
}


# Field order in each template (must match the template text)
_FIELD_NAMES = [
    "subject", "predicate", "object", "summary",
    "scope", "confidence", "salience", "quote",
]


class SlotExtractor:
    """
    Extract structured candidates using type-specific prompts.

    For each type in a classified segment, runs one small LLM call with a
    template designed for that specific type. The fill-in-the-blanks format
    is much easier for small models than open-ended generation.
    """

    def __init__(
        self,
        model_name: str = _EXTRACTION_MODEL,
        generation_fn: Callable | None = None,
        max_tokens: int = 250,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.generation_fn = generation_fn or _default_generate
        self.max_tokens = max_tokens
        self.temperature = temperature

    def extract(
        self,
        segment_text: str,
        fact_types: list[str],
    ) -> list[ExtractedCandidate]:
        """
        Extract candidates from a segment for the given types.

        Args:
            segment_text: the text to extract from
            fact_types: list of type strings (e.g., ["decision", "todo"])

        Returns:
            list of ExtractedCandidate (one per type that produced valid output)
        """
        candidates: list[ExtractedCandidate] = []
        for ftype in fact_types:
            if ftype == "none" or ftype not in _EXTRACTION_TEMPLATES:
                continue

            template = _EXTRACTION_TEMPLATES[ftype]
            prompt = template.replace("{text}", segment_text[:2000])

            raw = self.generation_fn(
                self.model_name,
                [{"role": "user", "content": prompt}],
                self.max_tokens,
                self.temperature,
            )

            candidate = _parse_slot_output(raw, ftype, segment_text)
            if candidate is not None:
                candidates.append(candidate)

        return candidates


def _parse_slot_output(
    raw: str,
    fact_type: str,
    source_text: str,
) -> ExtractedCandidate | None:
    """
    Parse the model's fill-in-the-blanks output.

    The model responds with lines like:
        SUBJECT: ppmlx
        PREDICATE: uses
        OBJECT: MLX
        SUMMARY: ppmlx uses MLX for inference
        SCOPE: project
        CONFIDENCE: 0.9
        SALIENCE: 0.85
        QUOTE: ppmlx uses MLX

    We extract values with regex, apply validation, and return a candidate.
    """
    fields: dict[str, str] = {}

    for field_name in _FIELD_NAMES:
        # Match "FIELD: value" or "FIELD value" — model might vary format
        m = re.search(
            rf"{field_name}\s*[:=-]\s*(.+?)(?:\n\s*(?:{ '|'.join(_FIELD_NAMES) })\s*[:=-]|\n\s*\n|\Z)",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            fields[field_name] = m.group(1).strip()

    # Validate required fields
    subject = _clean(fields.get("subject", ""))
    predicate = _clean(fields.get("predicate", ""))
    object_ = _clean(fields.get("object", ""))
    summary = _clean(fields.get("summary", ""))

    if not subject or not predicate or not object_:
        return None

    # Scope validation
    scope = _clean(fields.get("scope", "global")).lower()
    if scope not in {"global", "project", "session"}:
        # Heuristic: if the segment mentions project-specific things, scope=project
        scope = "project" if any(w in source_text.lower() for w in ["ppmlx", "project", "repo", "codebase"]) else "global"

    # Numeric fields
    confidence = _parse_float(fields.get("confidence", ""), 0.7)
    salience = _parse_float(fields.get("salience", ""), 0.8)

    # Source quote validation
    quote = _clean(fields.get("quote", ""))
    if not quote or quote.lower() not in source_text.lower():
        # Fallback: use the best matching span from source text
        # Try to find the most relevant sentence
        quote = _find_best_quote(source_text, subject, predicate, object_)
        if not quote:
            return None

    # Build text from summary or construct from S-P-O
    text = summary if summary else f"{subject} {predicate} {object_}"

    return ExtractedCandidate(
        type=fact_type,
        subject=subject,
        predicate=predicate,
        object=object_,
        text=text,
        scope=scope,
        confidence=confidence,
        salience=salience,
        source_quote=quote,
        metadata={
            "extractor": "slot_extractor_v1",
            "extraction_model": _EXTRACTION_MODEL,
        },
    )


def _find_best_quote(
    source_text: str,
    subject: str,
    predicate: str,
    object_: str,
) -> str:
    """Find the sentence in source_text that best matches the S-P-O triple."""
    sentences = re.split(r"(?<=[.!?])\s+", source_text)
    best_score = 0.0
    best_sentence = ""

    subj_terms = set(subject.lower().split())
    pred_terms = set(predicate.lower().split())
    obj_terms = set(object_.lower().split())
    all_terms = subj_terms | pred_terms | obj_terms

    if not all_terms:
        return ""

    for sent in sentences:
        sent_lower = sent.lower()
        sent_terms = set(re.findall(r"\b\w+\b", sent_lower))
        overlap = len(all_terms & sent_terms)
        score = overlap / max(len(all_terms), 1)
        if score > best_score and len(sent) > 10:
            best_score = score
            best_sentence = sent.strip()

    return best_sentence


def _clean(value: str) -> str:
    """Clean a string value from model output."""
    value = value.strip()
    # Remove trailing punctuation that the model might add
    while value and value[-1] in ",.;:":
        value = value[:-1].strip()
    # Remove leading markers
    value = re.sub(r"^[-*•]\s*", "", value)
    return value


def _parse_float(value: str, default: float) -> float:
    """Parse a float with defensive fallback."""
    if not value:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (ValueError, TypeError):
        return default


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
