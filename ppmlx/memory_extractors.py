"""Optional model-backed memory extractors.

This module is intentionally independent from memory_engine configuration and
storage. Extractors here return ``ShadowMemoryCandidate`` instances compatible
with the existing shadow-memory validator/write path, but they do not write to
SQLite or mutate any global engine state on their own.
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ppmlx.memory_engine import ALLOWED_TYPES, ShadowMemoryCandidate, event_source_text


DEFAULT_MEMORY_EXTRACTION_MODEL = "gemma-4-e2b"
DEFAULT_GEMMA_MEMORY_MODEL = DEFAULT_MEMORY_EXTRACTION_MODEL  # backward-compatible alias
MODEL_MEMORY_JSON_EXTRACTOR = "model_memory_json_v1"
LLM_STRICT_JSON_EXTRACTOR = MODEL_MEMORY_JSON_EXTRACTOR  # backward-compatible alias
GEMMA_STRICT_JSON_EXTRACTOR = MODEL_MEMORY_JSON_EXTRACTOR  # backward-compatible alias
MODEL_MEMORY_PIPE_EXTRACTOR = "model_memory_pipe_v1"
_ALLOWED_SCOPES = {"global", "project", "session"}

# Pipe-delimited field order (must match prompt and parser).
_PIPE_FIELDS = ["type", "subject", "predicate", "object", "text", "scope", "confidence", "salience", "source_quote"]

GenerationFn = Callable[[str, list[dict[str, str]], int, float], str]
_LOCAL_GENERATION_LOCK = threading.Lock()


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
        """Build a token-efficient pipe-delimited prompt for memory extraction."""
        project_id = event.get("project_id") or ""
        session_id = event.get("session_id") or ""
        source = event_source_text(event)
        allowed_types = ", ".join(sorted(ALLOWED_TYPES))
        header = "|".join(_PIPE_FIELDS)
        return f"""You are a high-precision memory extraction function.
Return ONLY pipe-delimited rows. No markdown, no json, no code fences, no prose.
If there are no safe candidates, return nothing (empty).

Task: extract at most {self.max_candidates} small durable memory candidates from the evidence.
Only include facts explicitly supported by a source_quote copied verbatim from the evidence.

Allowed types: {allowed_types}
Allowed scopes: global, project, session
Project id, if relevant: {project_id}
Session id, if relevant: {session_id}

Format (one row per candidate):
{header}

Examples (format demonstration only — the evidence starts after EVIDENCE):
fact|alice|has_pet|golden retriever|Alice has a golden retriever.|global|0.95|0.9|"Alice has a golden retriever"
preference|bob|prefers|dark mode|Bob prefers dark mode for reading.|global|0.88|0.8|"Bob prefers dark mode"

Rules:
  type = one of: {allowed_types}
  subject = short stable subject (single concept, not a sentence)
  predicate = short relation/action (verb phrase)
  object = small atomic value
  text = one concise sentence
  scope = global | project | session
  confidence = 0.0 to 1.0 (how certain the fact is)
  salience = 0.0 to 1.0 (how important/durable this memory is)
  source_quote = verbatim quote from the EVIDENCE section below (wrap in double-quotes if it contains "|")

Drop speculative, unsupported, sensitive, or merely conversational candidates.

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
        source_text = event_source_text(event)

        # Primary: pipe-delimited format (compact, fewer tokens).
        pipe_dicts = parse_pipe_delimited_payload(raw)
        if pipe_dicts:
            return self._coerce_and_dedupe(pipe_dicts, source_text, extractor_tag=MODEL_MEMORY_PIPE_EXTRACTOR)

        # Fallback: strict JSON (backward compatible with older models).
        payload = parse_strict_json_payload(raw)
        json_items = _candidate_items(payload)
        return self._coerce_and_dedupe(json_items, source_text, extractor_tag=MODEL_MEMORY_JSON_EXTRACTOR)

    def _coerce_and_dedupe(
        self,
        items: list[Any],
        source_text: str,
        *,
        extractor_tag: str,
    ) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for item in items:
            candidate = self._coerce_candidate(item, source_text)
            if candidate is None:
                continue
            candidate.metadata["extractor"] = extractor_tag
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

        raw_metadata = item.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
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


def parse_pipe_delimited_payload(text: str) -> list[dict[str, Any]]:
    """Parse pipe-delimited memory candidate rows from model output.

    Each line is a pipe-separated row. Fields containing the pipe character
    must be double-quoted. The header row is auto-detected and skipped.
    """
    if not text or not text.strip():
        return []
    lines = text.strip().split("\n")
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split_pipe_row(line)
        # Skip header rows (field count matches and first field is "type").
        if len(fields) >= 8 and _norm(fields[0]) == "type":
            continue
        if len(fields) < 8:
            continue
        try:
            out.append({
                "type": fields[0].strip(),
                "subject": fields[1].strip(),
                "predicate": fields[2].strip(),
                "object": fields[3].strip(),
                "text": fields[4].strip(),
                "scope": fields[5].strip(),
                "confidence": float(fields[6].strip()),
                "salience": float(fields[7].strip()),
                "source_quote": fields[8].strip().strip('"') if len(fields) > 8 else "",
            })
        except (ValueError, IndexError):
            continue
    return out


def _split_pipe_row(line: str) -> list[str]:
    """Split a pipe-delimited row, respecting double-quoted fields."""
    fields: list[str] = []
    current: list[str] = []
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "|" and not in_quotes:
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    fields.append("".join(current))
    return fields


def _candidate_items(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        candidates = payload.get("candidates")
        return list(candidates) if isinstance(candidates, Sequence) and not isinstance(candidates, str) else []
    if isinstance(payload, list):
        return payload
    return []


def _generate_with_local_engine(model_name: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    from ppmlx.engine import get_engine

    with _exclusive_local_generation():
        result = get_engine().generate(
            model_name,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=False,
        )
    return result.text if hasattr(result, "text") else str(result[0])


@contextmanager
def _exclusive_local_generation():
    """Serialize local model-backed extraction across threads/processes.

    MLX local generation is not safe to fan out aggressively from multiple
    memory workers on one machine. The thread lock serializes chunk extraction
    within one worker process; the file lock serializes separate CLI workers.
    """
    with _LOCAL_GENERATION_LOCK:
        lock_handle = None
        fcntl_module = None
        try:
            import fcntl as imported_fcntl
            from ppmlx.config import get_ppmlx_dir

            fcntl_module = imported_fcntl
            lock_handle = (get_ppmlx_dir() / "memory-extractor.lock").open("w")
            fcntl_module.flock(lock_handle.fileno(), fcntl_module.LOCK_EX)
        except Exception:
            lock_handle = None
            fcntl_module = None
        try:
            yield
        finally:
            if lock_handle is not None and fcntl_module is not None:
                try:
                    fcntl_module.flock(lock_handle.fileno(), fcntl_module.LOCK_UN)
                except Exception:
                    pass
            if lock_handle is not None:
                lock_handle.close()


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
