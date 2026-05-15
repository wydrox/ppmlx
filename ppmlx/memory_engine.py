"""Shadow temporal-memory engine for ppmlx.

The engine captures request/response events, extracts high-precision memory
candidates, validates them defensively, and writes a temporal graph projection.
It deliberately does not inject memory into prompts; this is the write-path used
for shadow-mode evaluation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any

from ppmlx.memory_store import MemoryStore, get_memory_store
from ppmlx.tool_distillers import DistilledMemoryCandidate, GenericJsonToolDistiller, ToolDistiller


STATUS_ACTIVE = "active"
STATUS_QUARANTINED = "quarantined"
STATUS_REJECTED = "rejected"
STATUS_DISPUTED = "disputed"

ALLOWED_TYPES = {
    "fact",
    "preference",
    "decision",
    "todo",
    "constraint",
    "entity_note",
    "instruction",
    "relationship",
}

SENSITIVE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
]

SUPERSEDE_SIGNALS = (
    "actually",
    "from now on",
    "instead",
    "no longer",
    "not anymore",
    "supersede",
)

REJECT_SIGNALS = (
    "do not remember",
    "don't remember",
    "forget this",
    "ignore this",
)


@dataclass
class ShadowMemoryCandidate:
    type: str
    subject: str
    predicate: str
    object: str
    text: str
    scope: str
    confidence: float
    source_quote: str
    salience: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    candidate_id: str = ""

    def with_event(self, event_id: str) -> "ShadowMemoryCandidate":
        self.event_id = event_id
        self.candidate_id = _candidate_id(event_id, self.type, self.subject, self.predicate, self.object, self.scope)
        return self

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "event_id": self.event_id,
            "type": self.type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "text": self.text,
            "scope": self.scope,
            "confidence": self.confidence,
            "source_quote": self.source_quote,
            "salience": self.salience,
            "metadata": self.metadata,
        }


class RuleBasedMemoryExtractor:
    """High-precision extractor for shadow mode.

    Recall is intentionally low.  The first production slice should discover
    whether explicit durable context can be captured without polluting memory.
    """

    def __init__(self, max_candidates: int = 12, tool_distillers: list[ToolDistiller] | None = None):
        self.max_candidates = max_candidates
        self.tool_distillers = tool_distillers or [GenericJsonToolDistiller()]

    def extract(self, event: dict[str, Any]) -> list[ShadowMemoryCandidate]:
        source_text = event_source_text(event)
        messages_text = "\n".join(_message_to_text(msg) for msg in event.get("messages", []))
        project_id = event.get("project_id")
        candidates: list[ShadowMemoryCandidate] = []

        for text in _candidate_sources(messages_text):
            candidates.extend(self._extract_session_instruction(text))
            candidates.extend(self._extract_goals(text, project_id=project_id))
            candidates.extend(self._extract_preferences(text))
            candidates.extend(self._extract_decisions(text, project_id=project_id))
            candidates.extend(self._extract_constraints(text, project_id=project_id))
            candidates.extend(self._extract_shortlist(text, project_id=project_id))
            candidates.extend(self._extract_rejections(text, project_id=project_id))
            candidates.extend(self._extract_todos(text, project_id=project_id))
            candidates.extend(self._extract_remembered_facts(text, project_id=project_id))

        for message in event.get("messages", []):
            for distiller in self.tool_distillers:
                for distilled in distiller.distill(message, event):
                    candidates.append(self._from_distilled(distilled))

        # Preserve source evidence and anti-memory signals in metadata.
        reject_requested = any(signal in source_text.lower() for signal in REJECT_SIGNALS)
        unique: dict[tuple[str, str, str, str, str], ShadowMemoryCandidate] = {}
        for candidate in candidates:
            candidate.metadata.setdefault("extractor", "rule_based_v1")
            candidate.metadata["reject_requested"] = reject_requested
            key = (
                _norm(candidate.type),
                _norm(candidate.subject),
                _norm(candidate.predicate),
                _norm(candidate.object),
                _norm(candidate.scope),
            )
            if key not in unique:
                unique[key] = candidate
        return list(unique.values())[: self.max_candidates]

    @staticmethod
    def _from_distilled(candidate: DistilledMemoryCandidate) -> ShadowMemoryCandidate:
        return ShadowMemoryCandidate(
            type=candidate.type,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            text=candidate.text,
            scope=candidate.scope,
            confidence=candidate.confidence,
            source_quote=candidate.source_quote,
            salience=candidate.salience,
            metadata=dict(candidate.metadata),
        )

    @staticmethod
    def _extract_session_instruction(text: str) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        for match in re.finditer(r"for this session only,?\s+(.+?)(?:[.!?]|$)", text, re.IGNORECASE):
            obj = _clean_phrase(match.group(1))
            if obj:
                out.append(ShadowMemoryCandidate(
                    type="instruction",
                    subject="assistant",
                    predicate="should",
                    object=obj,
                    text=f"For this session only, assistant should {obj}.",
                    scope="session",
                    confidence=0.9,
                    source_quote=match.group(0).strip(),
                    salience=0.75,
                ))
        return out

    @staticmethod
    def _extract_goals(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "user"
        scope = "project" if project_id else "global"
        for match in re.finditer(r"\bgoal:\s*(.+?)(?:\n|$)", text, re.IGNORECASE):
            obj = _clean_phrase(match.group(1))
            if obj:
                out.append(ShadowMemoryCandidate(
                    type="fact",
                    subject=subject,
                    predicate="goal",
                    object=obj,
                    text=f"Goal: {obj}.",
                    scope=scope,
                    confidence=0.82,
                    source_quote=match.group(0).strip(),
                    salience=0.86,
                ))
        return out

    @staticmethod
    def _extract_preferences(text: str) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        patterns = [
            r"\bI prefer\s+(.+?)(?:[.!?]|$)",
            r"\buser prefers\s+(.+?)(?:[.!?]|$)",
            r"\bfrom now on I prefer\s+(.+?)(?:[.!?]|$)",
            r"\bkeep answers?\s+(.+?)\s+by default(?:[.!?]|$)",
            r"\bpreference:\s*(.+?)(?:\n|$)",
            r"(?:^|[.\n]\s*)prefer\s+(.+?)(?:[.!?]|$)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                obj = _clean_phrase(match.group(1))
                if obj:
                    out.append(ShadowMemoryCandidate(
                        type="preference",
                        subject="user",
                        predicate="prefers",
                        object=obj,
                        text=f"User prefers {obj}.",
                        scope="global",
                        confidence=0.88,
                        source_quote=match.group(0).strip(),
                        salience=0.85,
                    ))
        return out

    @staticmethod
    def _extract_decisions(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        scoped_patterns: list[tuple[str, str | None]] = [
            (r"\bin\s+([A-Za-z0-9_.:-]+),?\s+we decided\s+(?:to\s+)?(.+?)(?:[.!?]|$)", None),
            (r"\bwe decided\s+(?:to\s+)?(.+?)(?:[.!?]|$)", project_id),
            (r"\bdecision:\s*(.+?)(?:\n|$)", project_id),
        ]
        for pattern, default_subject in scoped_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if default_subject is None and len(match.groups()) >= 2:
                    subject = _clean_phrase(match.group(1))
                    obj = _clean_phrase(match.group(2))
                else:
                    subject = default_subject or "user"
                    obj = _clean_phrase(match.group(1))
                if obj:
                    scope = "project" if subject != "user" else "global"
                    out.append(ShadowMemoryCandidate(
                        type="decision",
                        subject=subject,
                        predicate="decided",
                        object=obj,
                        text=f"{subject} decision: {obj}.",
                        scope=scope,
                        confidence=0.9,
                        source_quote=match.group(0).strip(),
                        salience=0.9,
                    ))
        if project_id:
            for match in re.finditer(r"\bposition(?:ed|ing)?(?: it)?\s+as\s+(.+?)(?:[.!?]|$)", text, re.IGNORECASE):
                obj = _clean_phrase(match.group(1))
                if obj:
                    out.append(ShadowMemoryCandidate(
                        type="decision",
                        subject=project_id,
                        predicate="positioning",
                        object=obj,
                        text=f"{project_id} is positioned as {obj}.",
                        scope="project",
                        confidence=0.86,
                        source_quote=match.group(0).strip(),
                        salience=0.88,
                    ))
        return out

    @staticmethod
    def _extract_constraints(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "user"
        scope = "project" if project_id else "global"
        patterns = [
            (r"\b(?:budget|max budget|maximum budget)\s*(?:is|=|<=|under|up to)?\s*([0-9][0-9\s.,]*(?:PLN|zł|zl|EUR|USD)?)", "budget"),
            (r"\b(?:need|needs|must have|required|required feature)\s+(.+?)(?:\.(?=\s|$)|[!?]|$)", "requires"),
            (r"\b(?:screen size|size)\s*(?:is|=|:)?\s*([0-9]{2}\s*(?:-|–|to)\s*[0-9]{2}\s*(?:inch|inches|\"|cal)?)", "screen_size"),
            (r"\b(?:viewing distance|distance)\s*(?:is|=|:)?\s*([0-9][0-9.,]*\s*(?:m|meter|meters))", "viewing_distance"),
        ]
        for pattern, predicate in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                obj = _clean_phrase(match.group(1))
                if obj:
                    out.append(ShadowMemoryCandidate(
                        type="constraint",
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        text=f"{subject} constraint: {predicate} = {obj}.",
                        scope=scope,
                        confidence=0.86,
                        source_quote=match.group(0).strip(),
                        salience=0.88,
                    ))
        return out

    @staticmethod
    def _extract_shortlist(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "shopping_session"
        for match in re.finditer(r"\b(?:shortlist|shortlisted|current shortlist)\s*(?::|is|=)?\s*(.+?)(?:[.!?]|$)", text, re.IGNORECASE):
            obj = _clean_phrase(match.group(1))
            if obj:
                out.append(ShadowMemoryCandidate(
                    type="entity_note",
                    subject=subject,
                    predicate="shortlist",
                    object=obj,
                    text=f"Current shortlist: {obj}.",
                    scope="project" if project_id else "session",
                    confidence=0.84,
                    source_quote=match.group(0).strip(),
                    salience=0.86,
                ))
        return out

    @staticmethod
    def _extract_rejections(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "shopping_session"
        for match in re.finditer(r"\b(?:rejected|reject)\s+(.+?)\s+because\s+(.+?)(?:\.(?=\s|$)|[!?]|$)", text, re.IGNORECASE):
            item = _clean_phrase(match.group(1))
            reason = _clean_phrase(match.group(2))
            if item and reason:
                obj = f"{item} because {reason}"
                out.append(ShadowMemoryCandidate(
                    type="decision",
                    subject=subject,
                    predicate="rejected",
                    object=obj,
                    text=f"Rejected {item}: {reason}.",
                    scope="project" if project_id else "session",
                    confidence=0.86,
                    source_quote=match.group(0).strip(),
                    salience=0.82,
                ))
        return out

    @staticmethod
    def _extract_todos(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        for match in re.finditer(r"\b(?:todo|task):\s*(.+?)(?:\n|$)", text, re.IGNORECASE):
            obj = _clean_phrase(match.group(1))
            if obj:
                subject = project_id or "user"
                out.append(ShadowMemoryCandidate(
                    type="todo",
                    subject=subject,
                    predicate="needs",
                    object=obj,
                    text=f"{subject} todo: {obj}.",
                    scope="project" if project_id else "global",
                    confidence=0.84,
                    source_quote=match.group(0).strip(),
                    salience=0.82,
                ))
        return out

    @staticmethod
    def _extract_remembered_facts(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        for match in re.finditer(r"\bremember\s+(?:that\s+)?(.+?)(?:\.(?=\s|$)|[!?]|$)", text, re.IGNORECASE):
            obj = _clean_phrase(match.group(1))
            if obj:
                subject = project_id or "user"
                out.append(ShadowMemoryCandidate(
                    type="fact",
                    subject=subject,
                    predicate="remembered",
                    object=obj,
                    text=f"Remember that {obj}.",
                    scope="project" if project_id else "global",
                    confidence=0.78,
                    source_quote=match.group(0).strip(),
                    salience=0.78,
                ))
        return out


class MemoryValidator:
    def __init__(
        self,
        store: MemoryStore,
        *,
        min_active_confidence: float = 0.72,
        min_quarantine_confidence: float = 0.55,
        min_salience: float = 0.35,
    ):
        self.store = store
        self.min_active_confidence = min_active_confidence
        self.min_quarantine_confidence = min_quarantine_confidence
        self.min_salience = min_salience

    def validate(self, event: dict[str, Any], candidate: ShadowMemoryCandidate) -> dict[str, Any]:
        reasons: list[str] = []
        invalidates: list[str] = []
        source_text = event_source_text(event)

        if candidate.type not in ALLOWED_TYPES:
            return self._decision(STATUS_REJECTED, candidate, ["unsupported_type"])
        if candidate.metadata.get("reject_requested"):
            return self._decision(STATUS_REJECTED, candidate, ["reject_requested"])
        if _contains_sensitive("\n".join([candidate.text, candidate.object, candidate.source_quote])):
            return self._decision(STATUS_REJECTED, candidate, ["sensitive"])
        if candidate.source_quote and candidate.source_quote.lower() not in source_text.lower():
            return self._decision(STATUS_REJECTED, candidate, ["unsupported"])
        if not candidate.source_quote:
            return self._decision(STATUS_REJECTED, candidate, ["missing_evidence"])
        if candidate.salience < self.min_salience:
            return self._decision(STATUS_REJECTED, candidate, ["low_salience"])
        if self._is_scope_leakage(event, candidate):
            return self._decision(STATUS_REJECTED, candidate, ["wrong_scope"])
        if candidate.confidence < self.min_active_confidence:
            if candidate.confidence >= self.min_quarantine_confidence:
                return self._decision(STATUS_QUARANTINED, candidate, ["low_confidence"])
            return self._decision(STATUS_REJECTED, candidate, ["low_confidence"])

        active_slot = self.store.find_active_slot(
            type=candidate.type,
            subject=candidate.subject,
            predicate=candidate.predicate,
            scope=candidate.scope,
        )
        for active in active_slot:
            if _norm(active["object"]) == _norm(candidate.object):
                return self._decision(STATUS_REJECTED, candidate, ["duplicate"])
            if any(signal in source_text.lower() for signal in SUPERSEDE_SIGNALS):
                invalidates.append(active["candidate_id"])
                reasons.append("supersedes_prior")
            else:
                return self._decision(STATUS_DISPUTED, candidate, ["contradiction"])

        return self._decision(STATUS_ACTIVE, candidate, reasons, invalidates=invalidates)

    @staticmethod
    def _decision(
        status: str,
        candidate: ShadowMemoryCandidate,
        reasons: list[str],
        *,
        invalidates: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "status": status,
            "scope": candidate.scope,
            "reasons": reasons,
            "invalidates": invalidates or [],
            "valid_from": None,
            "valid_to": None,
        }

    @staticmethod
    def _is_scope_leakage(event: dict[str, Any], candidate: ShadowMemoryCandidate) -> bool:
        source_lower = event_source_text(event).lower()
        if ("this session only" in source_lower or "for this session" in source_lower) and candidate.scope != "session":
            return True
        if candidate.scope == "global" and event.get("project_id") and candidate.type in {"decision", "todo", "relationship"}:
            return True
        return False


class MemoryEngine:
    """Shadow write-path engine for temporal-memory capture."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        extractor: RuleBasedMemoryExtractor | None = None,
    ):
        self.store = store or get_memory_store()
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.validator = MemoryValidator(self.store)

    def capture_chat(
        self,
        *,
        request_id: str,
        endpoint: str,
        model_alias: str,
        model_repo: str,
        messages: list[dict[str, Any]],
        response_text: str | None,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": request_id,
            "endpoint": endpoint,
            "app_id": app_id,
            "project_id": project_id,
            "session_id": session_id,
            "model_alias": model_alias,
            "model_repo": model_repo,
            "messages": messages,
            "response_text": response_text or "",
            "metadata": metadata or {},
            "request": {"messages": messages},
        }
        start = time.perf_counter()
        self.store.record_event(event)
        candidates = [candidate.with_event(request_id) for candidate in self.extractor.extract(event)]
        validations: list[dict[str, Any]] = []
        for candidate in candidates:
            validation = self.validator.validate(event, candidate)
            self.store.store_candidate(candidate.to_record(), validation)
            if validation.get("invalidates"):
                self.store.mark_invalidated(validation["invalidates"], invalidated_by=candidate.candidate_id)
            if validation.get("status") == STATUS_ACTIVE:
                self.store.upsert_memory_edge(candidate.to_record())
            validations.append(validation)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "event_id": request_id,
            "candidates": len(candidates),
            "active": sum(1 for item in validations if item["status"] == STATUS_ACTIVE),
            "quarantined": sum(1 for item in validations if item["status"] == STATUS_QUARANTINED),
            "rejected": sum(1 for item in validations if item["status"] == STATUS_REJECTED),
            "disputed": sum(1 for item in validations if item["status"] == STATUS_DISPUTED),
            "duration_ms": round(elapsed_ms, 3),
        }


def get_memory_engine(path: Path | None = None) -> MemoryEngine:
    store = get_memory_store(path) if path else get_memory_store()
    return MemoryEngine(store=store)


def event_source_text(event: dict[str, Any]) -> str:
    parts = [_message_to_text(msg) for msg in event.get("messages", [])]
    if event.get("response_text"):
        parts.append(f"assistant: {event['response_text']}")
    return "\n".join(part for part in parts if part)


def _candidate_sources(messages_text: str) -> list[str]:
    # Extract only from user/developer/system text, not assistant output.
    return [line.split(": ", 1)[1] if ": " in line else line for line in messages_text.splitlines() if line.strip()]


def _message_to_text(message: dict[str, Any]) -> str:
    role = str(message.get("role", "user"))
    content = message.get("content", "")
    if isinstance(content, str):
        return f"{role}: {content}"
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return f"{role}: {' '.join(parts)}"
    return f"{role}: {content}"


def _contains_sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _candidate_id(event_id: str, type_: str, subject: str, predicate: str, object_: str, scope: str) -> str:
    digest = sha1(f"{event_id}:{type_}:{_norm(subject)}:{_norm(predicate)}:{_norm(object_)}:{scope}".encode()).hexdigest()[:16]
    return f"mem_{digest}"


def _clean_phrase(value: str) -> str:
    cleaned = " ".join(value.strip().strip("'\"").split())
    cleaned = re.sub(r"\s+(please|thanks)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .;:-")


def _norm(value: str) -> str:
    return " ".join(str(value).lower().strip().split())
