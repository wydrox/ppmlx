"""Shadow temporal-memory engine for ppmlx.

The engine captures request/response events, extracts high-precision memory
candidates, validates them defensively, and writes a temporal graph projection.
It deliberately does not inject memory into prompts; this is the write-path used
for shadow-mode evaluation.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any

from ppmlx.config import load_config

from ppmlx.memory_store import MemoryStore, get_memory_store
from ppmlx.tool_distillers import CodingToolDistiller, DistilledMemoryCandidate, GenericJsonToolDistiller, ToolDistiller


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
    "workflow_state",
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
        self.tool_distillers = tool_distillers or [GenericJsonToolDistiller(), CodingToolDistiller()]

    def extract(self, event: dict[str, Any]) -> list[ShadowMemoryCandidate]:
        source_text = event_source_text(event)
        project_id = event.get("project_id")
        candidates: list[ShadowMemoryCandidate] = []

        for text in _candidate_sources(event.get("messages", []), response_text=event.get("response_text")):
            candidates.extend(self._extract_session_instruction(text))
            candidates.extend(self._extract_goals(text, project_id=project_id))
            candidates.extend(self._extract_preferences(text))
            candidates.extend(self._extract_decisions(text, project_id=project_id))
            candidates.extend(self._extract_constraints(text, project_id=project_id))
            candidates.extend(self._extract_shortlist(text, project_id=project_id))
            candidates.extend(self._extract_rejections(text, project_id=project_id))
            candidates.extend(self._extract_todos(text, project_id=project_id))
            candidates.extend(self._extract_workflow_state(text, project_id=project_id))
            candidates.extend(self._extract_file_changes(text, project_id=project_id))
            candidates.extend(self._extract_validation_results(text, project_id=project_id))
            candidates.extend(self._extract_command_runs(text, project_id=project_id))
            candidates.extend(self._extract_commit_state(text, project_id=project_id))
            candidates.extend(self._extract_implementation_facts(text, project_id=project_id))
            candidates.extend(self._extract_remembered_facts(text, project_id=project_id))

        for message in event.get("messages", []):
            for distiller in self.tool_distillers:
                for distilled in distiller.distill(message, event):
                    candidates.append(self._from_distilled(distilled))

        unique: dict[tuple[str, str, str, str, str], ShadowMemoryCandidate] = {}
        for candidate in candidates:
            candidate.metadata.setdefault("extractor", "rule_based_v1")
            candidate.metadata["reject_requested"] = _candidate_reject_requested(candidate, source_text)
            key = (
                _norm(candidate.type),
                _norm(candidate.subject),
                _norm(candidate.predicate),
                _norm(candidate.object),
                _norm(candidate.scope),
            )
            if key not in unique:
                unique[key] = candidate
        return _prioritize_extracted_candidates(list(unique.values()), limit=self.max_candidates)

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
        patterns = [
            r"\b(?:todo|task|next):\s*(.+?)(?:\n|$)",
            r"\bnext\s+(?:step\s+)?(?:is\s+to\s+|action\s+is\s+to\s+)?(.+?)(?:[.!?]|$)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
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
    def _extract_workflow_state(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "session"
        scope = "project" if project_id else "session"
        patterns: list[tuple[str, str, str, float]] = [
            (r"\b(?:current task|current work|working on|teraz robimy|aktualnie)\s*:?\s*(.+?)(?:\n|[.!?](?=\s|$)|$)", "current_task", "Current task", 0.94),
            (r"\b(?:next action|next step|next|dalej|następnie|nastepnie)\s*:?\s*(.+?)(?:\n|[.!?](?=\s|$)|$)", "next_action", "Next action", 0.93),
            (r"\b(?:blocker|blocked by|blocking issue|remaining issue|problem)\s*:?\s*(.+?)(?:\n|[.!?](?=\s|$)|$)", "blocker", "Blocker", 0.92),
            (r"\b(?:I(?:'|’)?m|I am|I(?:'|’)?ll|I will)\s+((?:applying|fixing|running|rerunning|testing|checking|deploying|updating|patching|implementing)\b.+?)(?:\n|[.!?](?=\s|$)|$)", "next_action", "Next action", 0.9),
        ]
        for pattern, predicate, label, salience in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                obj = _clean_phrase(match.group(1))
                if not obj or _looks_like_workflow_noise(obj):
                    continue
                out.append(ShadowMemoryCandidate(
                    type="workflow_state",
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    text=f"{label}: {obj}.",
                    scope=scope,
                    confidence=0.86,
                    source_quote=match.group(0).strip(),
                    salience=salience,
                ))
        return out

    @staticmethod
    def _extract_command_runs(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "session"
        scope = "project" if project_id else "session"
        command_patterns = [
            r"\b(?:ran|run|running|reran|re-ran|odpal(?:iłem|ilem|am)?)\s+`([^`\n]{3,180})`",
            r"`([^`\n]{0,160}(?:pnpm|npm|yarn|uv|pytest|ruff|eslint|build|test)[^`\n]{0,160})`",
        ]
        for pattern in command_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                command = _clean_phrase(match.group(1))
                if not command or _looks_like_validation_noise(command):
                    continue
                out.append(ShadowMemoryCandidate(
                    type="workflow_state",
                    subject=subject,
                    predicate="command_run",
                    object=command,
                    text=f"Command run: `{command}`.",
                    scope=scope,
                    confidence=0.8,
                    source_quote=match.group(0).strip(),
                    salience=0.88,
                ))
        return out

    @staticmethod
    def _extract_file_changes(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "session"
        scope = "project" if project_id else "session"
        patterns = [
            r"\bfile\s+`?([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|md|json|toml|ya?ml|swift|go|rs|sh|sql|css|html))`?\s+(?:changed|updated|modified|patched|created)",
            r"\b(?:changed|updated|modified|patched|created)\s+(?:file\s+)?`?([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|md|json|toml|ya?ml|swift|go|rs|sh|sql|css|html))`?",
            r"`([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|md|json|toml|ya?ml|swift|go|rs|sh|sql|css|html))`\s*(?:changed|updated|modified|patched|created|zmieniony|zaktualizowany)?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                path = _clean_phrase(match.group(1))
                if path:
                    out.append(ShadowMemoryCandidate(
                        type="entity_note",
                        subject=subject,
                        predicate="file_changed",
                        object=path,
                        text=f"File changed: {path}.",
                        scope=scope,
                        confidence=0.78,
                        source_quote=match.group(0).strip(),
                        salience=0.82,
                    ))
        return out

    @staticmethod
    def _extract_validation_results(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "session"
        scope = "project" if project_id else "session"
        patterns = [
            r"`([^`\n]{0,160}(?:pnpm|npm|yarn|uv|pytest|ruff|eslint|build|test)[^`\n]{0,160})`\s*(✅|passed|pass|ok)?",
            r"\b((?:pnpm|npm|yarn|uv)\s+[^\n`]{0,180}?(?:build|test|pytest|eslint)[^\n`]{0,80})\s*(✅|passed|pass|ok)?",
            r"\b(\d+\s+passed|\d+\s+failed|all checks passed|tests? passed|tests? failed|ruff passed|pytest passed)\b",
            r"\b(uv run pytest[^\n.]{0,120}\s+passed)\b",
            r"`?(origin/[A-Za-z0-9_.-]+)`?\s+(zaktualizowany|updated|pushed)\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                result = _clean_phrase(match.group(1))
                if not result or _looks_like_validation_noise(result):
                    continue
                marker = " ✅" if "✅" in match.group(0) else ""
                rendered = f"`{result}`{marker}" if _looks_like_command_or_ref(result) else result
                out.append(ShadowMemoryCandidate(
                    type="fact",
                    subject=subject,
                    predicate="validation",
                    object=rendered,
                    text=f"Validation result: {rendered}.",
                    scope=scope,
                    confidence=0.86,
                    source_quote=match.group(0).strip(),
                    salience=0.94,
                ))
        return out

    @staticmethod
    def _extract_commit_state(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "session"
        scope = "project" if project_id else "session"
        commit_context = bool(re.search(r"\b(commit|push|pushed|origin/)\b", text, flags=re.IGNORECASE))
        patterns = [
            r"(?:commit\s*(?:\+\s*push)?|pushed|push)\s*:?\s*-?\s*`?([0-9a-f]{7,40})\s+([^`\n]{3,160})`?",
            r"`([0-9a-f]{7,40})\s+([^`\n]{3,160})`",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if pattern.startswith("`") and not commit_context:
                    continue
                sha = match.group(1).strip()
                title = _clean_phrase(match.group(2))
                if not title:
                    continue
                obj = f"{sha} {title}"
                out.append(ShadowMemoryCandidate(
                    type="fact",
                    subject=subject,
                    predicate="commit_pushed" if commit_context else "commit",
                    object=obj,
                    text=f"Commit pushed: `{obj}`.",
                    scope=scope,
                    confidence=0.88,
                    source_quote=match.group(0).strip(),
                    salience=0.96,
                ))
        return out

    @staticmethod
    def _extract_implementation_facts(text: str, *, project_id: str | None) -> list[ShadowMemoryCandidate]:
        out: list[ShadowMemoryCandidate] = []
        subject = project_id or "session"
        scope = "project" if project_id else "session"
        for match in re.finditer(r"\b(?:globalnie\s+w|globally\s+in)\s+`?([^`\s]+)`?", text, re.IGNORECASE):
            path = _clean_phrase(match.group(1))
            if path:
                out.append(ShadowMemoryCandidate(
                    type="fact",
                    subject=subject,
                    predicate="global_fix",
                    object=path,
                    text=f"Global fix implemented in `{path}`.",
                    scope=scope,
                    confidence=0.88,
                    source_quote=match.group(0).strip(),
                    salience=0.95,
                ))
        auth_patterns = [
            r"(`?ConvexProviderWithAuth`?[^\n.]{0,220}(?:token|WorkOS|Convex|isAuthenticated)[^\n.]{0,220})",
            r"(query\s+`?api\.[^`\s]+`?[^\n.]{0,220}\bskip\b[^\n.]{0,220}(?:auth|token|gotowe|ready))",
        ]
        for pattern in auth_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                fact = _clean_phrase(match.group(1))
                if fact:
                    out.append(ShadowMemoryCandidate(
                        type="fact",
                        subject=subject,
                        predicate="auth_race_fix",
                        object=fact,
                        text=f"Auth-race fix: {fact}.",
                        scope=scope,
                        confidence=0.86,
                        source_quote=match.group(0).strip(),
                        salience=0.94,
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


class HybridMemoryExtractor:
    """Run rule-based anchors and model JSON extraction, then merge candidates.

    Rule-based extraction gives deterministic high-precision candidates; model
    extraction improves recall for less formulaic real-session language. The
    MemoryEngine still performs final dedupe and validation for all candidates.
    """

    def __init__(self, *extractors: Any):
        self.extractors = [extractor for extractor in extractors if extractor is not None]
        self.max_candidates = sum(max(0, int(getattr(extractor, "max_candidates", 12))) for extractor in self.extractors)

    def extract(self, event: dict[str, Any]) -> list[ShadowMemoryCandidate]:
        candidates: list[ShadowMemoryCandidate] = []
        for extractor in self.extractors:
            try:
                candidates.extend(extractor.extract(event))
            except Exception:
                # Hybrid extraction should degrade to the surviving extractor(s):
                # a local model/JSON failure must not erase deterministic rule
                # anchors such as todos, validations, and workflow state.
                continue
        return candidates


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
            if self._is_additive_slot(candidate):
                continue
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
    def _is_additive_slot(candidate: ShadowMemoryCandidate) -> bool:
        if candidate.type == "todo":
            return True
        if candidate.type == "constraint" and _norm(candidate.predicate) in {"requires", "required_feature"}:
            return True
        if candidate.type == "decision" and _norm(candidate.predicate) == "rejected":
            return True
        if candidate.type == "fact" and _norm(candidate.predicate) in {
            "validation",
            "commit",
            "commit_pushed",
            "global_fix",
            "auth_race_fix",
            "remembered",
        }:
            return True
        if candidate.type in {"entity_note", "relationship", "workflow_state"}:
            return True
        return False

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
        extractor: Any | None = None,
        *,
        sync_extractor: Any | None = None,
        extraction_workers: int = 1,
        parallel_extraction: bool = False,
        enqueue_extraction: bool = False,
        extraction_input_tokens: int = 6000,
        extraction_overlap_tokens: int = 600,
        extraction_max_chunks_per_event: int = 32,
        extraction_timeout_seconds: float = 0.0,
    ):
        self.store = store or get_memory_store()
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.sync_extractor = sync_extractor
        self.extraction_workers = max(1, int(extraction_workers))
        self.parallel_extraction = parallel_extraction and self.extraction_workers > 1
        self.enqueue_extraction = bool(enqueue_extraction)
        self.extraction_input_tokens = max(32, int(extraction_input_tokens))
        self.extraction_overlap_tokens = max(0, min(int(extraction_overlap_tokens), self.extraction_input_tokens // 2))
        self.extraction_max_chunks_per_event = max(1, int(extraction_max_chunks_per_event))
        self.extraction_timeout_seconds = max(0.0, float(extraction_timeout_seconds))
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
        if self.enqueue_extraction:
            sync_result = self._extract_validate_store(
                event,
                suppress_extraction_errors=True,
                start=start,
                extractor=self.sync_extractor,
            ) if self.sync_extractor is not None else None
            self.store.enqueue_extraction_job(event, source_event_id=request_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "event_id": request_id,
                "candidates": int((sync_result or {}).get("candidates") or 0),
                "active": int((sync_result or {}).get("active") or 0),
                "quarantined": int((sync_result or {}).get("quarantined") or 0),
                "rejected": int((sync_result or {}).get("rejected") or 0),
                "disputed": int((sync_result or {}).get("disputed") or 0),
                "queued": 1,
                "duration_ms": round(elapsed_ms, 3),
            }

        return self._extract_validate_store(event, suppress_extraction_errors=True, start=start)

    def process_extraction_job(self, worker_id: str = "worker", once: bool = True) -> dict[str, Any] | None:
        """Claim and process one asynchronous extraction job.

        The payload is the already-recorded event from ``capture_chat``. This
        worker path deliberately does not call ``record_event`` or enqueue any
        follow-up job; it only extracts, validates, stores candidates, and then
        marks the job completed. Exceptions mark the claimed job failed, with a
        retry when attempts remain.
        """
        if not once:
            processed: list[dict[str, Any]] = []
            while True:
                item = self.process_extraction_job(worker_id=worker_id, once=True)
                if item is None:
                    break
                processed.append(item)
            return {"processed": len(processed), "jobs": processed}

        if self.extraction_timeout_seconds > 0:
            self.store.requeue_stale_claimed_extraction_jobs(stale_after_seconds=self.extraction_timeout_seconds)

        job = self.store.claim_extraction_job(worker_id)
        if job is None:
            return None

        stop_heartbeat, heartbeat_thread = self._start_extraction_job_heartbeat(job)
        try:
            event = dict(job.get("payload") or {})
            event_id = str(event.get("event_id") or job.get("source_event_id") or job["job_id"])
            event["event_id"] = event_id
            event.setdefault("request", {"messages": event.get("messages", [])})
            result = self._extract_validate_store(event, suppress_extraction_errors=False)
            result["job_id"] = job["job_id"]
            self.store.complete_extraction_job(job["job_id"], result=result)
            return result
        except Exception as exc:  # pragma: no cover - exact extractor failures vary
            self.store.fail_extraction_job(job["job_id"], str(exc), retry=True)
            failed_job = self.store.get_extraction_job(job["job_id"]) or job
            return {
                "job_id": job["job_id"],
                "event_id": job.get("source_event_id"),
                "failed": 1,
                "status": failed_job.get("status", "failed"),
                "error": str(exc),
            }
        finally:
            if stop_heartbeat is not None:
                stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)

    def _start_extraction_job_heartbeat(
        self,
        job: dict[str, Any],
    ) -> tuple[threading.Event | None, threading.Thread | None]:
        if self.extraction_timeout_seconds <= 0:
            return None, None
        job_id = str(job.get("job_id") or "")
        worker_id = str(job.get("worker_id") or "")
        if not job_id or not worker_id:
            return None, None
        interval = max(0.05, min(10.0, self.extraction_timeout_seconds / 3.0))
        stop = threading.Event()

        def _heartbeat() -> None:
            while not stop.wait(interval):
                self.store.renew_extraction_job_claim(job_id, worker_id)

        thread = threading.Thread(target=_heartbeat, name=f"memory-extraction-heartbeat-{worker_id}", daemon=True)
        thread.start()
        return stop, thread

    def _extract_validate_store(
        self,
        event: dict[str, Any],
        *,
        suppress_extraction_errors: bool,
        start: float | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter() if start is None else start
        event_id = str(event["event_id"])
        candidates = [
            candidate.with_event(event_id)
            for candidate in self._extract_candidates(event, suppress_errors=suppress_extraction_errors, extractor=extractor)
        ]
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
            "event_id": event_id,
            "candidates": len(candidates),
            "active": sum(1 for item in validations if item["status"] == STATUS_ACTIVE),
            "quarantined": sum(1 for item in validations if item["status"] == STATUS_QUARANTINED),
            "rejected": sum(1 for item in validations if item["status"] == STATUS_REJECTED),
            "disputed": sum(1 for item in validations if item["status"] == STATUS_DISPUTED),
            "duration_ms": round(elapsed_ms, 3),
        }

    def _extract_candidates(
        self,
        event: dict[str, Any],
        *,
        suppress_errors: bool = True,
        extractor: Any | None = None,
    ) -> list[ShadowMemoryCandidate]:
        active_extractor = extractor or self.extractor
        chunks = _event_extraction_chunks(
            event,
            max_input_tokens=self.extraction_input_tokens,
            overlap_tokens=self.extraction_overlap_tokens,
            max_chunks=self.extraction_max_chunks_per_event,
        )
        try:
            if len(chunks) <= 1:
                candidates = active_extractor.extract(chunks[0])
            elif self.parallel_extraction and extractor is None:
                candidates = self._extract_candidates_parallel(chunks, suppress_errors=suppress_errors, extractor=active_extractor)
            else:
                candidates = self._extract_candidates_sequential(chunks, suppress_errors=suppress_errors, extractor=active_extractor)
        except Exception:
            if suppress_errors:
                return []
            raise
        return self._dedupe_candidates(candidates)[: self._max_candidates_per_event()]

    def _extract_candidates_sequential(
        self,
        chunks: list[dict[str, Any]],
        *,
        suppress_errors: bool,
        extractor: Any,
    ) -> list[ShadowMemoryCandidate]:
        merged: list[ShadowMemoryCandidate] = []
        for chunk in chunks:
            try:
                merged.extend(extractor.extract(chunk))
            except Exception:
                if not suppress_errors:
                    raise
        return merged

    def _extract_candidates_parallel(
        self,
        chunks: list[dict[str, Any]],
        *,
        suppress_errors: bool,
        extractor: Any,
    ) -> list[ShadowMemoryCandidate]:
        chunk_results: list[list[ShadowMemoryCandidate]] = [[] for _ in chunks]
        with ThreadPoolExecutor(max_workers=self.extraction_workers) as executor:
            futures = {executor.submit(extractor.extract, chunk): idx for idx, chunk in enumerate(chunks)}
            for future, idx in futures.items():
                try:
                    chunk_results[idx] = future.result()
                except Exception:
                    if not suppress_errors:
                        raise
                    chunk_results[idx] = []

        merged: list[ShadowMemoryCandidate] = []
        for result in chunk_results:
            merged.extend(result)
        return merged

    def _max_candidates_per_event(self) -> int:
        return max(0, int(getattr(self.extractor, "max_candidates", 12)))

    @staticmethod
    def _dedupe_candidates(candidates: list[ShadowMemoryCandidate]) -> list[ShadowMemoryCandidate]:
        unique: dict[tuple[str, str, str, str, str], ShadowMemoryCandidate] = {}
        for candidate in candidates:
            key = (
                _norm(candidate.type),
                _norm(candidate.subject),
                _norm(candidate.predicate),
                _norm(candidate.object),
                _norm(candidate.scope),
            )
            if key not in unique:
                unique[key] = candidate
        return list(unique.values())


def get_memory_engine(path: Path | None = None) -> MemoryEngine:
    store = get_memory_store(path) if path else get_memory_store()
    cfg = load_config()
    common_kwargs = {
        "extraction_input_tokens": cfg.memory.extraction_input_tokens,
        "extraction_overlap_tokens": cfg.memory.extraction_overlap_tokens,
        "extraction_max_chunks_per_event": cfg.memory.extraction_max_chunks_per_event,
        "extraction_timeout_seconds": cfg.memory.extraction_timeout_seconds,
    }
    from ppmlx.memory_extractors import ModelMemoryJsonExtractor

    rule_extractor = RuleBasedMemoryExtractor(max_candidates=cfg.memory.max_candidates_per_event)
    model_extractor = ModelMemoryJsonExtractor(
        model_name=cfg.memory.extraction_model,
        max_candidates=cfg.memory.max_candidates_per_event,
        max_tokens=cfg.memory.extraction_max_tokens,
    )
    return MemoryEngine(
        store=store,
        extractor=model_extractor,
        sync_extractor=rule_extractor,
        extraction_workers=cfg.memory.extraction_workers,
        parallel_extraction=cfg.memory.extraction_workers > 1,
        enqueue_extraction=True,
        **common_kwargs,
    )


def event_source_text(event: dict[str, Any]) -> str:
    parts = [_message_to_text(msg) for msg in event.get("messages", [])]
    if event.get("response_text"):
        parts.append(f"assistant: {event['response_text']}")
    return "\n".join(part for part in parts if part)


def _event_extraction_chunks(
    event: dict[str, Any],
    *,
    max_input_tokens: int,
    overlap_tokens: int,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """Split an event into token-budgeted extraction windows with overlap.

    The estimate is intentionally heuristic (roughly 4 chars/token) so this
    remains tokenizer-independent. It keeps full events intact when they fit;
    long events become message windows, and oversized single messages are split
    into overlapping text fragments.
    """
    max_input_tokens = max(32, int(max_input_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), max_input_tokens // 2))
    max_chunks = max(1, int(max_chunks))
    if _estimate_event_tokens(event) <= max_input_tokens:
        return [event]

    segments = _event_extraction_segments(event, max_input_tokens=max_input_tokens, overlap_tokens=overlap_tokens)
    if not segments:
        return [event]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for segment in segments:
        segment_tokens = _estimate_message_tokens(segment)
        if current and current_tokens + segment_tokens > max_input_tokens:
            chunks.append(current)
            if len(chunks) >= max_chunks:
                break
            current = _tail_overlap_messages(current, overlap_tokens)
            current_tokens = sum(_estimate_message_tokens(item) for item in current)
            if current and current_tokens + segment_tokens > max_input_tokens:
                current = []
                current_tokens = 0
        current.append(segment)
        current_tokens += segment_tokens
    if current and len(chunks) < max_chunks:
        chunks.append(current)

    out: list[dict[str, Any]] = []
    total = len(chunks)
    for idx, messages in enumerate(chunks):
        chunk = dict(event)
        chunk["messages"] = messages
        chunk["request"] = {"messages": messages}
        chunk["response_text"] = ""
        metadata = dict(event.get("metadata") or {})
        metadata["extraction_chunk"] = {
            "index": idx,
            "total": total,
            "estimated_tokens": sum(_estimate_message_tokens(item) for item in messages),
            "max_input_tokens": max_input_tokens,
            "overlap_tokens": overlap_tokens,
        }
        chunk["metadata"] = metadata
        out.append(chunk)
    return out or [event]


def _event_extraction_segments(
    event: dict[str, Any],
    *,
    max_input_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    messages = [dict(message) for message in event.get("messages", [])]
    if event.get("response_text"):
        messages.append({"role": "assistant", "content": str(event.get("response_text") or "")})

    segments: list[dict[str, Any]] = []
    for message in messages:
        if _estimate_message_tokens(message) <= max_input_tokens:
            segments.append(message)
            continue
        segments.extend(_split_oversized_message(message, max_input_tokens=max_input_tokens, overlap_tokens=overlap_tokens))
    return segments


def _split_oversized_message(
    message: dict[str, Any],
    *,
    max_input_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    content = message.get("content", "")
    if isinstance(content, str):
        text = content
    else:
        try:
            text = json.dumps(content, ensure_ascii=False, default=str)
        except TypeError:
            text = str(content)
    if not text:
        return [message]

    max_chars = max(64, (max_input_tokens - 16) * 4)
    overlap_chars = min(max_chars // 2, overlap_tokens * 4)
    step = max(1, max_chars - overlap_chars)
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars]
        if not piece:
            break
        chunk = dict(message)
        chunk["content"] = piece
        chunk["metadata"] = {
            **(message.get("metadata") if isinstance(message.get("metadata"), dict) else {}),
            "extraction_fragment": {"start": start, "end": start + len(piece)},
        }
        chunks.append(chunk)
        if start + max_chars >= len(text):
            break
    return chunks or [message]


def _tail_overlap_messages(messages: list[dict[str, Any]], overlap_tokens: int) -> list[dict[str, Any]]:
    if overlap_tokens <= 0:
        return []
    selected: list[dict[str, Any]] = []
    total = 0
    for message in reversed(messages):
        tokens = _estimate_message_tokens(message)
        if selected and total + tokens > overlap_tokens:
            break
        selected.append(message)
        total += tokens
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def _estimate_event_tokens(event: dict[str, Any]) -> int:
    return sum(_estimate_message_tokens(message) for message in event.get("messages", [])) + _estimate_text_tokens(event.get("response_text"))


def _estimate_message_tokens(message: dict[str, Any]) -> int:
    try:
        raw = json.dumps(message, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(message)
    return _estimate_text_tokens(raw) + 4


def _estimate_text_tokens(text: Any) -> int:
    if not text:
        return 0
    return max(1, (len(str(text)) + 3) // 4)


def _candidate_sources(messages: list[dict[str, Any]], *, response_text: str | None = None) -> list[str]:
    # Extract durable facts from conversational text only. Raw tool/function
    # output is handled by explicit distillers; running generic regex extractors
    # over logs/search output turns unrelated fixture data into project memory.
    out: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        if role in {"tool", "function"}:
            continue
        text = _message_to_text(message)
        _, sep, content = text.partition(": ")
        candidate = content if sep else text
        if candidate.strip():
            out.append(candidate)
    if response_text and str(response_text).strip():
        out.append(str(response_text).strip())
    return out


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


def _candidate_reject_requested(candidate: ShadowMemoryCandidate, source_text: str) -> bool:
    """Return True only when an anti-memory request appears to target this candidate.

    A broad event-level flag is too destructive: users often say "do not
    remember token X" while other non-sensitive facts in the same turn are safe
    and useful. Keep fail-closed behavior for the referenced candidate text.
    """
    lower_source = source_text.lower()
    if not any(signal in lower_source for signal in REJECT_SIGNALS):
        return False
    candidate_bits = [candidate.object, candidate.subject, candidate.source_quote]
    for bit in candidate_bits:
        bit_norm = _norm(str(bit))
        if bit_norm and len(bit_norm) >= 4 and bit_norm in _norm(source_text):
            for signal in REJECT_SIGNALS:
                signal_idx = lower_source.find(signal)
                bit_idx = lower_source.find(str(bit).lower())
                if signal_idx >= 0 and bit_idx >= 0 and signal_idx <= bit_idx < signal_idx + 240:
                    return True
    return False


def _candidate_id(event_id: str, type_: str, subject: str, predicate: str, object_: str, scope: str) -> str:
    digest = sha1(f"{event_id}:{type_}:{_norm(subject)}:{_norm(predicate)}:{_norm(object_)}:{scope}".encode()).hexdigest()[:16]
    return f"mem_{digest}"


def _prioritize_extracted_candidates(candidates: list[ShadowMemoryCandidate], *, limit: int) -> list[ShadowMemoryCandidate]:
    """Keep workflow/action anchors from being crowded out by generic facts."""
    protected_predicates = {
        "current_task",
        "next_action",
        "blocker",
        "command_run",
        "validation",
        "file_changed",
        "commit",
        "commit_pushed",
        "global_fix",
        "auth_race_fix",
    }

    def rank(candidate: ShadowMemoryCandidate) -> tuple[int, float, float, str]:
        protected = int(candidate.type == "workflow_state" or candidate.predicate in protected_predicates or candidate.type == "todo")
        return (protected, float(candidate.salience), float(candidate.confidence), candidate.candidate_id)

    return sorted(candidates, key=rank, reverse=True)[: max(0, limit)]


def _clean_phrase(value: str) -> str:
    cleaned = " ".join(value.strip().strip("'\"").split())
    cleaned = re.sub(r"\s+(please|thanks)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .;:-")


def _looks_like_command_or_ref(value: str) -> bool:
    lowered = value.lower()
    return (
        any(token in lowered for token in ("pnpm", "npm", "yarn", "uv ", "pytest", "ruff", "eslint", "build", "test"))
        or lowered.startswith("origin/")
    )


def _looks_like_validation_noise(value: str) -> bool:
    lowered = value.lower().strip()
    if len(lowered) < 3:
        return True
    if lowered in {"build", "test", "tests", "passed", "failed"}:
        return True
    return False


def _looks_like_workflow_noise(value: str) -> bool:
    lowered = value.lower().strip()
    if len(lowered) < 4:
        return True
    if lowered in {"it", "this", "that", "to", "dalej", "kontynuuj", "działaj", "dzialaj"}:
        return True
    return False


def _norm(value: str) -> str:
    return " ".join(str(value).lower().strip().split())
