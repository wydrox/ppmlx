"""Tests for ppmlx.memory_engine and ppmlx.memory_store shadow write path."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.memory_engine import MemoryEngine
from ppmlx.memory_store import MemoryStore, reset_memory_store


def _make_engine(tmp_path: Path) -> tuple[MemoryEngine, MemoryStore]:
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    return MemoryEngine(store=store), store


def test_capture_preference_creates_active_memory_and_edge(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-pref",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer concise answers."}],
        response_text="ok",
    )

    assert result["active"] == 1
    stats = store.stats()
    assert stats["events"] == 1
    assert stats["candidates"] == 1
    assert stats["edges"] == 1
    rows = store.search("concise answers")
    assert len(rows) == 1
    assert rows[0]["status"] == "active"
    assert rows[0]["type"] == "preference"


def test_secret_candidate_is_rejected(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-secret",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "Remember that api_key=sk-test-abc123SECRET."}],
        response_text="ok",
    )

    assert result["active"] == 0
    assert result["rejected"] == 1
    rows = store.query_candidates(status="rejected")
    assert rows[0]["reasons"] == ["sensitive"]
    assert store.search("api key") == []


def test_supersession_invalidates_prior_active_memory(tmp_path):
    engine, store = _make_engine(tmp_path)

    engine.capture_chat(
        request_id="req-old",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer verbose explanations."}],
        response_text="ok",
    )
    result = engine.capture_chat(
        request_id="req-new",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "Actually, from now on I prefer concise answers."}],
        response_text="ok",
    )

    assert result["active"] == 1
    active = store.query_candidates(status="active")
    superseded = store.query_candidates(status="superseded")
    assert len(active) == 1
    assert active[0]["object"] == "concise answers"
    assert len(superseded) == 1
    assert superseded[0]["object"] == "verbose explanations"


def test_tv_shopping_atoms_are_extracted(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-tv",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="tv-shopping",
        session_id="s1",
        messages=[{"role": "user", "content": "Budget <= 5000 PLN. Need HDMI 2.1 for PS5. Shortlist: LG OLED C4, Samsung S90D. Rejected Samsung CU8000 because 60Hz and no HDMI 2.1. Todo: ask room brightness."}],
        response_text="ok",
    )

    assert result["active"] >= 5
    rows = store.query_candidates(status="active", project_id="tv-shopping", session_id="s1")
    texts = "\n".join(row["text"] for row in rows)
    assert "budget = 5000 PLN" in texts
    assert "requires = HDMI 2.1 for PS5" in texts
    assert "Current shortlist: LG OLED C4, Samsung S90D" in texts
    assert "Rejected Samsung CU8000: 60Hz and no HDMI 2.1" in texts
    assert "todo: ask room brightness" in texts


def test_json_tool_payload_is_distilled_through_validator(tmp_path):
    engine, store = _make_engine(tmp_path)
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

    result = engine.capture_chat(
        request_id="req-json-tool",
        endpoint="/v1/chat/completions#compact",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="tv-shopping",
        session_id="s1",
        messages=[{"role": "tool", "name": "product_search", "content": json.dumps(payload)}],
        response_text="ok",
    )

    assert result["active"] >= 6
    rows = store.query_candidates(status="active", project_id="tv-shopping", session_id="s1", limit=50)
    texts = "\n".join(row["text"] for row in rows)
    assert "Candidate: LG OLED C4." in texts
    assert "LG OLED C4 price: 4599 PLN." in texts
    assert "LG OLED C4 spec panel = OLED." in texts
    assert "Rejected Samsung CU8000: 60Hz and no HDMI 2.1." in texts
    assert all(row["metadata"].get("distiller") == "generic_json_v1" for row in rows)


def test_project_decision_is_project_scoped(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-decision",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        messages=[{"role": "user", "content": "We decided to build a temporal memory graph first."}],
        response_text="ok",
    )

    assert result["active"] == 1
    rows = store.query_candidates(status="active")
    assert rows[0]["scope"] == "project"
    assert rows[0]["subject"] == "ppmlx"


def test_memory_store_graph_snapshot_returns_nodes_edges_and_events(tmp_path):
    engine, store = _make_engine(tmp_path)
    engine.capture_chat(
        request_id="graph-seed",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        session_id="s1",
        messages=[{"role": "user", "content": "We decided to build a temporal memory graph first."}],
        response_text="ok",
    )

    snapshot = store.graph_snapshot(project_id="ppmlx", session_id="s1", query="temporal graph")

    assert snapshot["filters"]["project_id"] == "ppmlx"
    assert snapshot["nodes"]
    assert snapshot["edges"]
    assert snapshot["candidates"]
    assert snapshot["events"][0]["event_id"] == "graph-seed"
    decided_edge = next(edge for edge in snapshot["edges"] if edge["relation"] == "decided")
    assert decided_edge["source"] == decided_edge["from_entity_id"]
    assert decided_edge["target"] == decided_edge["to_entity_id"]
    assert decided_edge["label"] == "decided"
    source_node = next(node for node in snapshot["nodes"] if node["id"] == decided_edge["source"])
    assert source_node["label"] == source_node["name"]
    assert source_node["degree"] >= 1
    assert source_node["size"] > 20


def test_graph_cli_json_outputs_snapshot(tmp_home):
    reset_memory_store()
    store = MemoryStore(tmp_home / ".ppmlx" / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="graph-cli-seed",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        messages=[{"role": "user", "content": "We decided to expose ppmlx graph view."}],
        response_text="ok",
    )
    reset_memory_store()

    result = CliRunner().invoke(app, ["graph", "--project", "ppmlx", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["nodes"]
    assert data["edges"]
    assert data["candidates"][0]["project_id"] == "ppmlx"
    reset_memory_store()


def test_memory_store_compact_stats(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.record_compaction({
        "request_id": "r1",
        "mode": "compact",
        "original_tokens": 1000,
        "reduced_tokens": 100,
        "hot_tail_tokens": 60,
        "session_context_tokens": 40,
        "cold_messages": 10,
        "context_items": 4,
        "compacted": True,
        "injected": True,
        "latency_ms": 12.5,
        "project_id": "p1",
    })

    stats = store.compact_stats(project_id="p1")

    assert stats["total"] == 1
    assert stats["compacted"] == 1
    assert stats["injected"] == 1
    assert stats["avg_compression_ratio"] == 10.0
    assert stats["avg_latency_ms"] == 12.5
    assert stats["p95_latency_ms"] == 12.5
    assert stats["recent"][0]["request_id"] == "r1"


def test_memory_cli_compact_stats(tmp_home):
    reset_memory_store()
    store = MemoryStore(tmp_home / ".ppmlx" / "memory.db")
    store.record_compaction({
        "request_id": "r-cli",
        "mode": "compact",
        "original_tokens": 2000,
        "reduced_tokens": 200,
        "compacted": True,
        "injected": True,
        "latency_ms": 8.0,
    })
    reset_memory_store()

    result = CliRunner().invoke(app, ["memory", "compact-stats", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["avg_compression_ratio"] == 10.0


def test_memory_cli_handoff(tmp_home):
    reset_memory_store()
    engine = MemoryEngine(store=MemoryStore(tmp_home / ".ppmlx" / "memory.db"))
    engine.capture_chat(
        request_id="req-handoff-cli",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="tv-shopping",
        session_id="s1",
        messages=[{"role": "user", "content": "Budget <= 5000 PLN. Need HDMI 2.1 for PS5."}],
        response_text="ok",
    )
    reset_memory_store()

    result = CliRunner().invoke(app, [
        "memory", "handoff",
        "--project", "tv-shopping",
        "--session", "s1",
        "--query", "HDMI TV",
        "--json",
    ])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["project_id"] == "tv-shopping"
    assert data["session_id"] == "s1"
    assert data["items_count"] >= 2
    assert "Compacted local session context" in data["context"]
    assert "budget = 5000 PLN" in data["context"]
    assert "HDMI 2.1 for PS5" in data["context"]


def test_memory_cli_status_and_search(tmp_home):
    reset_memory_store()
    engine = MemoryEngine(store=MemoryStore(tmp_home / ".ppmlx" / "memory.db"))
    engine.capture_chat(
        request_id="req-cli",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer short answers."}],
        response_text="ok",
    )
    reset_memory_store()

    runner = CliRunner()
    status_result = runner.invoke(app, ["memory", "status", "--json"])
    assert status_result.exit_code == 0
    status = json.loads(status_result.output)
    assert status["events"] == 1
    assert status["by_status"]["active"] == 1

    search_result = runner.invoke(app, ["memory", "search", "short", "--json"])
    assert search_result.exit_code == 0
    rows = json.loads(search_result.output)
    assert len(rows) == 1
    assert rows[0]["object"] == "short answers"
