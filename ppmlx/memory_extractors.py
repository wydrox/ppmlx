"""Optional model-backed memory extractors.

This module is intentionally independent from memory_engine configuration and
storage. Extractors here return ``ShadowMemoryCandidate`` instances compatible
with the existing shadow-memory validator/write path, but they do not write to
SQLite or mutate any global engine state on their own.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ppmlx.memory_engine import ALLOWED_TYPES, ShadowMemoryCandidate, event_source_text


DEFAULT_MEMORY_EXTRACTION_MODEL = "gemma-4-e2b"
DEFAULT_GEMMA_MEMORY_MODEL = DEFAULT_MEMORY_EXTRACTION_MODEL  # backward-compatible alias
MODEL_MEMORY_JSON_EXTRACTOR = "model_memory_json_v1"
LLM_STRICT_JSON_EXTRACTOR = MODEL_MEMORY_JSON_EXTRACTOR  # backward-compatible alias
GEMMA_STRICT_JSON_EXTRACTOR = MODEL_MEMORY_JSON_EXTRACTOR  # backward-compatible alias
_ALLOWED_SCOPES = {"global", "project", "session"}

GenerationFn = Callable[[str, list[dict[str, str]], int, float], str]


class ModelMemoryJsonExtractor:
    """Extract small evidence-backed memory candidates via strict JSON.

    ``generation_fn`` is injectable so tests and downstream callers can replace
    local MLX generation. It receives ``(model_name, messages, max_tokens,
    temperature)`` and must return model text.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MEMORY_EXTRACTION_MODEL,
        *,
        generation_fn: GenerationFn | None = None,
        max_candidates: int = 8,
        max_tokens: int = 900,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.generation_fn = generation_fn or _generate_with_local_engine
        self.max_candidates = max_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_prompt: str | None = None

    def build_prompt(self, event: dict[str, Any]) -> str:
        """Build an explicit strict-JSON prompt for the memory extraction task."""
        project_id = event.get("project_id") or ""
        session_id = event.get("session_id") or ""
        source = event_source_text(event)
        allowed_types = ", ".join(sorted(ALLOWED_TYPES))
        return f"""You are a high-precision memory extraction function.
Return ONLY strict JSON. Do not use markdown, comments, prose, or trailing commas.

Task: extract at most {self.max_candidates} small durable memory candidates from the evidence.
Only include facts that are explicitly supported by a source_quote copied verbatim from the evidence.
Prefer precision over recall. If there are no safe candidates, return {{"candidates": []}}.

Allowed candidate types: {allowed_types}
Allowed scopes: global, project, session
Project id, if relevant: {project_id}
Session id, if relevant: {session_id}

Each candidate object must use this exact shape:
{{
  "type": "fact|preference|decision|todo|constraint|entity_note|instruction|relationship",
  "subject": "short stable subject",
  "predicate": "short relation/action",
  "object": "small atomic value",
  "text": "one concise sentence describing the memory",
  "scope": "global|project|session",
  "confidence": 0.0,
  "salience": 0.0,
  "source_quote": "verbatim quote from evidence"
}}

Drop candidates that are speculative, unsupported, sensitive secrets, merely conversational, or missing verbatim evidence.
Return schema exactly: {{"candidates": [candidate, ...]}}

Evidence:
<<<EVIDENCE
{source}
EVIDENCE
>>>"""

    def extract(self, event: dict[str, Any]) -> list[ShadowMemoryCandidate]:
        prompt = self.build_prompt(event)
        self.last_prompt = prompt
        raw = self.generation_fn(
            self.model_name,
            [{"role": "user", "content": prompt}],
            self.max_tokens,
            self.temperature,
        )
        payload = parse_strict_json_payload(raw)
        source_text = event_source_text(event)
        candidates = _candidate_items(payload)

        out: list[ShadowMemoryCandidate] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for item in candidates:
            candidate = self._coerce_candidate(item, source_text)
            if candidate is None:
                continue
            key = (
                _norm(candidate.type),
                _norm(candidate.subject),
                _norm(candidate.predicate),
                _norm(candidate.object),
                _norm(candidate.scope),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
            if len(out) >= self.max_candidates:
                break
        return out

    def _coerce_candidate(self, item: Any, source_text: str) -> ShadowMemoryCandidate | None:
        if not isinstance(item, Mapping):
            return None

        type_ = _clean_string(item.get("type"))
        if type_ not in ALLOWED_TYPES:
            return None

        source_quote = _clean_string(item.get("source_quote"))
        if not source_quote or source_quote.lower() not in source_text.lower():
            return None

        subject = _clean_string(item.get("subject"))
        predicate = _clean_string(item.get("predicate"))
        object_ = _clean_string(item.get("object"))
        text = _clean_string(item.get("text"))
        if not subject or not predicate or not object_ or not text:
            return None

        scope = _clean_string(item.get("scope")) or "global"
        if scope not in _ALLOWED_SCOPES:
            return None

        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata = dict(metadata)
        metadata["extractor"] = MODEL_MEMORY_JSON_EXTRACTOR
        metadata["extraction_model"] = self.model_name

        return ShadowMemoryCandidate(
            type=type_,
            subject=subject,
            predicate=predicate,
            object=object_,
            text=text,
            scope=scope,
            confidence=_clamp01(item.get("confidence")),
            source_quote=source_quote,
            salience=_clamp01(item.get("salience"), default=1.0),
            metadata=metadata,
        )


JsonMemoryExtractor = ModelMemoryJsonExtractor  # backward-compatible alias
GemmaJsonMemoryExtractor = ModelMemoryJsonExtractor  # backward-compatible alias


def parse_strict_json_payload(text: str) -> Any:
    """Parse defensive JSON from code fences or wrapped model text."""
    if not text or not text.strip():
        return None

    stripped = text.strip()
    for candidate in _json_text_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_text_candidates(text: str) -> list[str]:
    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    object_slice = _balanced_json_slice(text, "{", "}")
    if object_slice:
        candidates.append(object_slice)
    array_slice = _balanced_json_slice(text, "[", "]")
    if array_slice:
        candidates.append(array_slice)

    # Preserve order while avoiding duplicate parse attempts.
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _balanced_json_slice(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start == -1:
        return None
    in_string = False
    escaped = False
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _candidate_items(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        candidates = payload.get("candidates")
        return list(candidates) if isinstance(candidates, Sequence) and not isinstance(candidates, str) else []
    if isinstance(payload, list):
        return payload
    return []


def _generate_with_local_engine(model_name: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    from ppmlx.engine import get_engine

    result = get_engine().generate(
        model_name,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=False,
    )
    return result.text if hasattr(result, "text") else str(result[0])


def _clean_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _clamp01(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _norm(value: str) -> str:
    return " ".join(str(value).lower().strip().split())
