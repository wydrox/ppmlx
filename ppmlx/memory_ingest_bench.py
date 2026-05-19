"""Memory ingest/extraction benchmark utilities.

This benchmark intentionally measures ingest/extraction cost separately from
retrieval/query latency.  It can run rule-only, synchronous hybrid, or the
production-like async hybrid path (rule sync anchors + queued model worker).
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from ppmlx.answer_quality_replay import load_session_messages
from ppmlx.context_reducer import estimate_messages_tokens, group_messages_into_episodes
from ppmlx.memory_engine import HybridMemoryExtractor, MemoryEngine, RuleBasedMemoryExtractor, _event_extraction_chunks
from ppmlx.memory_extractors import GenerationFn, ModelMemoryJsonExtractor
from ppmlx.memory_store import MemoryStore


@dataclass
class MemoryIngestBenchEvent:
    index: int
    messages: int
    tokens: int
    chunks: int
    duration_ms: float
    candidates: int
    active: int
    rejected: int
    disputed: int
    queued: int = 0
    worker_duration_ms: float = 0.0
    worker_candidates: int = 0
    worker_active: int = 0
    worker_rejected: int = 0
    worker_disputed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "messages": self.messages,
            "tokens": self.tokens,
            "chunks": self.chunks,
            "duration_ms": self.duration_ms,
            "candidates": self.candidates,
            "active": self.active,
            "rejected": self.rejected,
            "disputed": self.disputed,
            "queued": self.queued,
            "worker_duration_ms": self.worker_duration_ms,
            "worker_candidates": self.worker_candidates,
            "worker_active": self.worker_active,
            "worker_rejected": self.worker_rejected,
            "worker_disputed": self.worker_disputed,
        }


@dataclass
class MemoryIngestBenchReport:
    path: str
    source: str
    mode: str
    extraction_model: str
    events: list[MemoryIngestBenchEvent] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        durations = [event.duration_ms for event in self.events]
        worker_durations = [event.worker_duration_ms for event in self.events if event.worker_duration_ms > 0]
        chunks = [float(event.chunks) for event in self.events]
        return {
            "events": len(self.events),
            "mode": self.mode,
            "extraction_model": self.extraction_model,
            "duration_ms_total": round(sum(durations), 3),
            "duration_ms_avg": round(mean(durations), 3) if durations else 0.0,
            "duration_ms_p95": round(_percentile(durations, 95), 3),
            "duration_ms_max": round(max(durations, default=0.0), 3),
            "worker_duration_ms_total": round(sum(worker_durations), 3),
            "worker_duration_ms_avg": round(mean(worker_durations), 3) if worker_durations else 0.0,
            "worker_duration_ms_p95": round(_percentile(worker_durations, 95), 3),
            "chunks_avg": round(mean(chunks), 3) if chunks else 0.0,
            "chunks_max": int(max(chunks, default=0.0)),
            "tokens_total": sum(event.tokens for event in self.events),
            "candidates_total": sum(event.candidates + event.worker_candidates for event in self.events),
            "active_total": sum(event.active + event.worker_active for event in self.events),
            "rejected_total": sum(event.rejected + event.worker_rejected for event in self.events),
            "disputed_total": sum(event.disputed + event.worker_disputed for event in self.events),
            "queued_total": sum(event.queued for event in self.events),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "mode": self.mode,
            "extraction_model": self.extraction_model,
            "summary": self.summary(),
            "events": [event.to_dict() for event in self.events],
        }


def run_memory_ingest_bench(
    *,
    path: Path | str,
    source: str = "auto",
    mode: str = "rule",
    extraction_model: str = "gemma-4-e2b",
    max_events: int = 10,
    max_candidates: int = 12,
    extraction_max_tokens: int = 900,
    extraction_input_tokens: int = 6000,
    extraction_overlap_tokens: int = 600,
    extraction_max_chunks_per_event: int = 32,
    generation_fn: GenerationFn | None = None,
) -> MemoryIngestBenchReport:
    resolved_source, messages = load_session_messages(path, source=source)
    mode = _normalize_mode(mode)
    episodes = group_messages_into_episodes([message for message in messages if message.get("role") != "system"])
    selected = episodes[: max(0, int(max_events))]

    with tempfile.TemporaryDirectory(prefix="ppmlx-memory-ingest-bench-") as tmp:
        store = MemoryStore(Path(tmp) / "memory.db")
        store.init()
        engine = _build_engine(
            mode=mode,
            store=store,
            extraction_model=extraction_model,
            max_candidates=max_candidates,
            extraction_max_tokens=extraction_max_tokens,
            extraction_input_tokens=extraction_input_tokens,
            extraction_overlap_tokens=extraction_overlap_tokens,
            extraction_max_chunks_per_event=extraction_max_chunks_per_event,
            generation_fn=generation_fn,
        )
        events: list[MemoryIngestBenchEvent] = []
        for episode in selected:
            event_id = f"ingest-bench-{episode.index}"
            event_payload = {
                "event_id": event_id,
                "messages": episode.messages,
                "response_text": "",
                "request": {"messages": episode.messages},
            }
            chunks = _event_extraction_chunks(
                event_payload,
                max_input_tokens=extraction_input_tokens,
                overlap_tokens=extraction_overlap_tokens,
                max_chunks=extraction_max_chunks_per_event,
            )
            started = time.perf_counter()
            result = engine.capture_chat(
                request_id=event_id,
                endpoint="/v1/chat/completions#ingest-bench",
                model_alias=extraction_model,
                model_repo=extraction_model,
                messages=episode.messages,
                response_text=None,
                project_id="memory-ingest-bench",
                session_id="bench",
                metadata={"episode_index": episode.index},
            )
            duration_ms = (time.perf_counter() - started) * 1000
            worker_result: dict[str, Any] = {}
            if mode == "async-hybrid":
                worker_result = engine.process_extraction_job(worker_id="ingest-bench-worker") or {}
            events.append(MemoryIngestBenchEvent(
                index=episode.index,
                messages=len(episode.messages),
                tokens=estimate_messages_tokens(episode.messages),
                chunks=len(chunks),
                duration_ms=round(duration_ms, 3),
                candidates=int(result.get("candidates") or 0),
                active=int(result.get("active") or 0),
                rejected=int(result.get("rejected") or 0),
                disputed=int(result.get("disputed") or 0),
                queued=int(result.get("queued") or 0),
                worker_duration_ms=float(worker_result.get("duration_ms") or 0.0),
                worker_candidates=int(worker_result.get("candidates") or 0),
                worker_active=int(worker_result.get("active") or 0),
                worker_rejected=int(worker_result.get("rejected") or 0),
                worker_disputed=int(worker_result.get("disputed") or 0),
            ))
    return MemoryIngestBenchReport(
        path=str(path),
        source=resolved_source,
        mode=mode,
        extraction_model=extraction_model,
        events=events,
    )


def _build_engine(
    *,
    mode: str,
    store: MemoryStore,
    extraction_model: str,
    max_candidates: int,
    extraction_max_tokens: int,
    extraction_input_tokens: int,
    extraction_overlap_tokens: int,
    extraction_max_chunks_per_event: int,
    generation_fn: GenerationFn | None,
) -> MemoryEngine:
    rule_extractor = RuleBasedMemoryExtractor(max_candidates=max_candidates)
    if mode == "rule":
        extractor: Any = rule_extractor
        return MemoryEngine(
            store=store,
            extractor=extractor,
            extraction_input_tokens=extraction_input_tokens,
            extraction_overlap_tokens=extraction_overlap_tokens,
            extraction_max_chunks_per_event=extraction_max_chunks_per_event,
        )

    model_extractor = ModelMemoryJsonExtractor(
        model_name=extraction_model,
        generation_fn=generation_fn,
        max_candidates=max_candidates,
        max_tokens=extraction_max_tokens,
    )
    if mode == "hybrid":
        return MemoryEngine(
            store=store,
            extractor=HybridMemoryExtractor(rule_extractor, model_extractor),
            extraction_input_tokens=extraction_input_tokens,
            extraction_overlap_tokens=extraction_overlap_tokens,
            extraction_max_chunks_per_event=extraction_max_chunks_per_event,
        )
    return MemoryEngine(
        store=store,
        extractor=model_extractor,
        sync_extractor=rule_extractor,
        enqueue_extraction=True,
        extraction_input_tokens=extraction_input_tokens,
        extraction_overlap_tokens=extraction_overlap_tokens,
        extraction_max_chunks_per_event=extraction_max_chunks_per_event,
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized in {"rule", "rule-only", "rule-based"}:
        return "rule"
    if normalized in {"hybrid", "sync-hybrid", "synchronous-hybrid"}:
        return "hybrid"
    if normalized in {"async", "async-hybrid", "worker", "production"}:
        return "async-hybrid"
    raise ValueError("mode must be one of: rule, hybrid, async-hybrid")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (percentile / 100)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
