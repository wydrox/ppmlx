"""Tests for trace export and compact replay."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.trace_replay import compact_replay, export_trace, load_trace, save_trace
from ppmlx.memory_store import MemoryStore, reset_memory_store


def _long_trace_messages() -> list[dict]:
    payload = {
        "products": [
            {
                "name": "LG OLED C4",
                "price": {"amount": 4599, "currency": "PLN"},
                "availability": "in_stock",
                "specs": {"panel": "OLED", "hdmi_2_1": 4},
            },
            {
                "name": "Samsung CU8000",
                "rejected": True,
                "reason": "60Hz and no HDMI 2.1",
            },
        ]
    }
    filler = "irrelevant tool chatter " * 240
    return [
        {"role": "system", "content": "You help with product research."},
        {"role": "user", "content": "Budget <= 5000 PLN. Need HDMI 2.1 for PS5."},
        {"role": "assistant", "content": filler},
        {"role": "tool", "name": "product_search_json", "content": json.dumps(payload)},
        {"role": "assistant", "content": "Results received."},
        {"role": "user", "content": "Which candidate survived and what was rejected?"},
    ]


def test_export_trace_from_memory_events(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.record_event({
        "event_id": "evt-1",
        "endpoint": "/v1/chat/completions",
        "project_id": "tv-shopping",
        "session_id": "s1",
        "model_alias": "test-model",
        "model_repo": "repo/private",
        "request": {"messages": _long_trace_messages()},
        "response_text": "ok",
        "metadata": {"source": "test"},
    })
    store.record_event({
        "event_id": "evt-internal",
        "endpoint": "/v1/chat/completions#compact",
        "project_id": "tv-shopping",
        "session_id": "s1",
        "request": {"messages": [{"role": "user", "content": "internal"}]},
    })

    trace = export_trace(project_id="tv-shopping", session_id="s1", store=store)

    assert trace.schema == "ppmlx.trace.v1"
    assert len(trace.events) == 1
    assert trace.events[0]["event_id"] == "evt-1"
    assert trace.events[0]["messages"][1]["content"].startswith("Budget")


def test_compact_replay_preserves_expected_terms(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.record_event({
        "event_id": "evt-replay",
        "endpoint": "/v1/chat/completions",
        "project_id": "tv-shopping",
        "session_id": "s1",
        "model_alias": "test-model",
        "model_repo": "repo/private",
        "request": {"messages": _long_trace_messages()},
    })
    trace = export_trace(project_id="tv-shopping", session_id="s1", store=store).to_dict()

    report = compact_replay(
        trace,
        expected_terms=[
            "budget = 5000 PLN",
            "requires = HDMI 2.1 for PS5",
            "Candidate: LG OLED C4",
            "Rejected Samsung CU8000: 60Hz and no HDMI 2.1",
        ],
        forbidden_terms=["budget = 8000 PLN"],
    )

    assert report.passed is True
    assert report.compression_ratio > 1
    assert report.missed_terms == []
    assert report.wrong_terms == []
    assert "LG OLED C4" in report.session_context
    assert "Rejected Samsung CU8000" in report.session_context


def test_trace_export_and_compact_replay_cli(tmp_home, tmp_path):
    reset_memory_store()
    store = MemoryStore(tmp_home / ".ppmlx" / "memory.db")
    store.record_event({
        "event_id": "evt-cli",
        "endpoint": "/v1/chat/completions",
        "project_id": "tv-shopping",
        "session_id": "s1",
        "model_alias": "test-model",
        "model_repo": "repo/private",
        "request": {"messages": _long_trace_messages()},
    })
    reset_memory_store()
    trace_path = tmp_path / "trace.json"

    runner = CliRunner()
    export_result = runner.invoke(app, [
        "trace", "export",
        "--project", "tv-shopping",
        "--session", "s1",
        "--output", str(trace_path),
    ])
    assert export_result.exit_code == 0
    assert trace_path.exists()
    assert load_trace(trace_path)["events"][0]["event_id"] == "evt-cli"

    replay_result = runner.invoke(app, [
        "compact-replay", str(trace_path),
        "--expect", "budget = 5000 PLN",
        "--expect", "Candidate: LG OLED C4",
        "--expect", "Rejected Samsung CU8000: 60Hz and no HDMI 2.1",
        "--json",
    ])
    assert replay_result.exit_code == 0
    data = json.loads(replay_result.output)
    assert data["passed"] is True
    assert data["missed_terms"] == []


def test_save_and_load_trace_roundtrip(tmp_path):
    trace = export_trace(store=MemoryStore(tmp_path / "empty.db"))
    path = save_trace(trace, tmp_path / "nested" / "trace.json")

    loaded = load_trace(path)

    assert loaded["schema"] == "ppmlx.trace.v1"
    assert loaded["events"] == []
