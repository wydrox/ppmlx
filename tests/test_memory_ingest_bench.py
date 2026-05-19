"""Tests for memory ingest benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from ppmlx.memory_ingest_bench import run_memory_ingest_bench


def _write_session(path: Path) -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "I prefer concise answers."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Todo: run validation."},
        {"role": "assistant", "content": "Validation result: `pnpm build` ✅."},
    ]
    path.write_text("\n".join(
        json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}})
        for message in messages
    ) + "\n")


def test_memory_ingest_bench_rule_mode_reports_ingest_metrics(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    _write_session(path)

    report = run_memory_ingest_bench(path=path, source="pi", mode="rule", max_events=2)
    data = report.to_dict()

    assert data["summary"]["mode"] == "rule"
    assert data["summary"]["events"] == 2
    assert data["summary"]["duration_ms_avg"] >= 0
    assert data["summary"]["chunks_max"] >= 1
    assert data["summary"]["candidates_total"] >= 1
    assert data["events"][0]["worker_duration_ms"] == 0.0


def test_memory_ingest_bench_async_hybrid_reports_worker_metrics(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    _write_session(path)

    def generation_fn(model_name, messages, max_tokens, temperature):
        return json.dumps({"candidates": [{
            "type": "todo",
            "subject": "memory-ingest-bench",
            "predicate": "needs",
            "object": "run validation",
            "text": "memory-ingest-bench todo: run validation.",
            "scope": "project",
            "confidence": 0.9,
            "salience": 0.9,
            "source_quote": "Todo: run validation.",
        }]})

    report = run_memory_ingest_bench(
        path=path,
        source="pi",
        mode="async-hybrid",
        max_events=2,
        generation_fn=generation_fn,
    )
    data = report.to_dict()

    assert data["summary"]["mode"] == "async-hybrid"
    assert data["summary"]["queued_total"] == 2
    assert data["summary"]["worker_duration_ms_total"] >= 0
    assert data["summary"]["candidates_total"] >= data["summary"]["active_total"]
