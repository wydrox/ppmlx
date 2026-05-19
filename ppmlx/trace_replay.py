"""Trace export and compact replay utilities.

Trace export is local-only and may include prompts/responses/tool output. Compact
replay runs a saved trace through the reducer/distillers/graph without invoking a
model, producing compression and continuity metrics.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppmlx.context_reducer import ContextBudget, ContextReducer, estimate_messages_tokens, group_messages_into_episodes
from ppmlx.memory_engine import MemoryEngine
from ppmlx.memory_store import MemoryStore, get_memory_store

TRACE_SCHEMA = "ppmlx.trace.v1"


@dataclass
class TraceExport:
    schema: str
    exported_at: str
    filters: dict[str, Any]
    events: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "exported_at": self.exported_at,
            "filters": self.filters,
            "events": self.events,
        }


@dataclass
class ReplayResult:
    trace_schema: str
    selected_event_id: str | None
    events: int
    passed: bool
    original_tokens: int
    reduced_tokens: int
    compression_ratio: float
    context_items: int
    cold_messages: int
    session_context_tokens: int
    latency_ms: float
    retrieval_latency_ms: float = 0.0
    expected_terms: list[str] = field(default_factory=list)
    found_terms: list[str] = field(default_factory=list)
    missed_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    wrong_terms: list[str] = field(default_factory=list)
    session_context: str = ""
    reduced_context: str = ""
    reduced_messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema": self.trace_schema,
            "selected_event_id": self.selected_event_id,
            "events": self.events,
            "passed": self.passed,
            "original_tokens": self.original_tokens,
            "reduced_tokens": self.reduced_tokens,
            "compression_ratio": self.compression_ratio,
            "context_items": self.context_items,
            "cold_messages": self.cold_messages,
            "session_context_tokens": self.session_context_tokens,
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "expected_terms": self.expected_terms,
            "found_terms": self.found_terms,
            "missed_terms": self.missed_terms,
            "forbidden_terms": self.forbidden_terms,
            "wrong_terms": self.wrong_terms,
            "session_context": self.session_context,
            "reduced_context": self.reduced_context,
            "reduced_messages": self.reduced_messages,
        }


def export_trace(
    *,
    app_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    since_hours: float | None = None,
    limit: int = 100,
    include_internal: bool = False,
    store: MemoryStore | None = None,
) -> TraceExport:
    memory_store = store or get_memory_store()
    events = memory_store.query_events(
        app_id=app_id,
        project_id=project_id,
        session_id=session_id,
        since_hours=since_hours,
        limit=limit,
        include_internal=include_internal,
    )
    return TraceExport(
        schema=TRACE_SCHEMA,
        exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        filters={
            "app_id": app_id,
            "project_id": project_id,
            "session_id": session_id,
            "since_hours": since_hours,
            "limit": limit,
            "include_internal": include_internal,
        },
        events=events,
    )


def save_trace(trace: TraceExport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)
    return out


def load_trace(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Trace must be a JSON object")
    if data.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"Unsupported trace schema: {data.get('schema')!r}")
    if not isinstance(data.get("events"), list):
        raise ValueError("Trace is missing events[]")
    return data


def compact_replay(
    trace: dict[str, Any],
    *,
    expected_terms: list[str] | None = None,
    forbidden_terms: list[str] | None = None,
    budget: ContextBudget | None = None,
    extractor: Any | None = None,
) -> ReplayResult:
    events = [event for event in trace.get("events", []) if isinstance(event, dict)]
    selected = _select_replay_event(events)
    expected_terms = expected_terms or []
    forbidden_terms = forbidden_terms or []
    if selected is None:
        return ReplayResult(
            trace_schema=str(trace.get("schema", "")),
            selected_event_id=None,
            events=len(events),
            passed=False,
            original_tokens=0,
            reduced_tokens=0,
            compression_ratio=0.0,
            context_items=0,
            cold_messages=0,
            session_context_tokens=0,
            latency_ms=0.0,
            expected_terms=expected_terms,
            missed_terms=expected_terms,
            forbidden_terms=forbidden_terms,
        )

    messages = list(selected.get("messages") or selected.get("request", {}).get("messages") or [])
    replay_budget = budget or ContextBudget(
        mode="compact",
        compact_threshold_tokens=1_500,
        hot_tail_tokens=900,
        session_context_tokens=2_000,
        max_context_items=40,
    )
    replay_budget.extract_cold_messages = False
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.db")
        store.init()
        engine = MemoryEngine(store=store, extractor=extractor)
        reducer = ContextReducer(replay_budget, store=store, engine=engine)
        _preingest_replay_memory(
            messages=messages,
            reducer=reducer,
            engine=engine,
            selected=selected,
        )
        start = time.perf_counter()
        reduction = reducer.reduce(
            request_id=f"replay-{selected.get('event_id') or 'trace'}",
            model_alias=str(selected.get("model_alias") or "trace-model"),
            model_repo=str(selected.get("model_repo") or "trace/model"),
            messages=messages,
            memory_context={
                "app_id": selected.get("app_id"),
                "project_id": selected.get("project_id"),
                "session_id": selected.get("session_id"),
                "metadata": {"trace_replay": True},
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000

    session_context = _extract_session_context(reduction.messages)
    reduced_context = _render_reduced_context(reduction.messages)
    haystack = _normalize("\n".join(_message_text(message) for message in reduction.messages))
    found = [term for term in expected_terms if _normalize(term) in haystack]
    missed = [term for term in expected_terms if term not in found]
    wrong = [term for term in forbidden_terms if _normalize(term) in haystack]
    compression_ratio = round(reduction.original_tokens / max(reduction.reduced_tokens, 1), 4)
    passed = not missed and not wrong and reduction.reduced_tokens > 0
    return ReplayResult(
        trace_schema=str(trace.get("schema", "")),
        selected_event_id=selected.get("event_id"),
        events=len(events),
        passed=passed,
        original_tokens=reduction.original_tokens,
        reduced_tokens=reduction.reduced_tokens,
        compression_ratio=compression_ratio,
        context_items=reduction.context_items,
        cold_messages=reduction.cold_messages,
        session_context_tokens=reduction.session_context_tokens,
        latency_ms=round(latency_ms, 3),
        retrieval_latency_ms=float(reduction.metadata.get("retrieval_latency_ms") or 0.0),
        expected_terms=expected_terms,
        found_terms=found,
        missed_terms=missed,
        forbidden_terms=forbidden_terms,
        wrong_terms=wrong,
        session_context=session_context,
        reduced_context=reduced_context,
        reduced_messages=reduction.messages,
    )


def _preingest_replay_memory(
    *,
    messages: list[dict[str, Any]],
    reducer: ContextReducer,
    engine: MemoryEngine,
    selected: dict[str, Any],
) -> None:
    """Populate graph before timed retrieval; extraction is ingest, not query path."""
    _, non_system = _split_system(messages)
    hot, cold = reducer._select_hot_tail(non_system)  # noqa: SLF001 - replay helper mirrors reducer split
    for episode in group_messages_into_episodes(cold):
        if not episode.messages:
            continue
        engine.capture_chat(
            request_id=f"preingest-{selected.get('event_id') or 'trace'}-e{episode.index}",
            endpoint="/v1/chat/completions#preingest",
            model_alias=str(selected.get("model_alias") or "trace-model"),
            model_repo=str(selected.get("model_repo") or "trace/model"),
            messages=episode.messages,
            response_text=None,
            app_id=selected.get("app_id"),
            project_id=selected.get("project_id"),
            session_id=selected.get("session_id"),
            metadata={"trace_replay_preingest": True},
        )
    for idx, message in enumerate(hot):
        if str(message.get("role") or "").lower() not in {"tool", "function"}:
            continue
        engine.capture_chat(
            request_id=f"preingest-{selected.get('event_id') or 'trace'}-hot-tool-{idx}",
            endpoint="/v1/chat/completions#preingest-hot-tool",
            model_alias=str(selected.get("model_alias") or "trace-model"),
            model_repo=str(selected.get("model_repo") or "trace/model"),
            messages=[message],
            response_text=None,
            app_id=selected.get("app_id"),
            project_id=selected.get("project_id"),
            session_id=selected.get("session_id"),
            metadata={"trace_replay_preingest": True, "hot_tool_distill": True},
        )


def _split_system(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_messages: list[dict[str, Any]] = []
    non_system: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            system_messages.append(message)
        else:
            non_system.append(message)
    return system_messages, non_system


def _select_replay_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for event in events:
        endpoint = str(event.get("endpoint") or "")
        if "#compact" in endpoint:
            continue
        messages = event.get("messages") or event.get("request", {}).get("messages") or []
        if isinstance(messages, list) and messages:
            candidates.append((estimate_messages_tokens(messages), str(event.get("timestamp") or ""), event))
    if not candidates:
        return None
    # OpenAI-compatible clients usually send full history; choose the richest/latest request.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _extract_session_context(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system" and "Compacted local session context" in str(message.get("content", "")):
            return str(message.get("content", ""))
    return ""


def _render_reduced_context(messages: list[dict[str, Any]], *, max_chars: int = 20000) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message")
        text = _message_text(message)
        if text.strip():
            parts.append(f"[{role}]\n{text}")
    rendered = "\n\n".join(parts)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars // 2] + "\n\n...[reduced context truncated]...\n\n" + rendered[-max_chars // 2:]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())
