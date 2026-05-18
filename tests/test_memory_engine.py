"""Tests for ppmlx.memory_engine and ppmlx.memory_store shadow write path."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.memory_engine import MemoryEngine, ShadowMemoryCandidate, get_memory_engine
from ppmlx.memory_store import MemoryStore, reset_memory_store


def _make_engine(tmp_path: Path) -> tuple[MemoryEngine, MemoryStore]:
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    return MemoryEngine(store=store), store


class FakeSingleCandidateExtractor:
    max_candidates = 1

    def extract(self, event):
        return [ShadowMemoryCandidate(
            type="preference",
            subject="user",
            predicate="prefers",
            object="async extraction",
            text="User prefers async extraction.",
            scope="global",
            confidence=0.9,
            source_quote="async extraction",
            salience=0.8,
        )]


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


def test_long_sentence_fact_remains_searchable_without_graph_object_node_or_edge(tmp_path):
    engine, store = _make_engine(tmp_path)
    content = (
        "Remember that ppmlx should keep canonical graph nodes short because arbitrary extracted "
        "sentences make unusable node labels and noisy JSON-like graph entities."
    )

    result = engine.capture_chat(
        request_id="graph-long-fact",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        messages=[{"role": "user", "content": content}],
        response_text="ok",
    )

    assert result["active"] == 1
    assert store.stats()["edges"] == 0
    rows = store.search("unusable node labels", project_id="ppmlx")
    assert len(rows) == 1
    snapshot = store.graph_snapshot(project_id="ppmlx", query="unusable node labels")
    assert snapshot["candidates"]
    assert snapshot["edges"] == []
    node_labels = {node["label"] for node in snapshot["nodes"]}
    assert "ppmlx" in node_labels
    assert all("unusable node labels" not in label for label in node_labels)


def test_canonicalized_entity_alias_is_stored_for_raw_label(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="graph-alias",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="Project PPMLX",
        messages=[{"role": "user", "content": "We decided to support canonical aliases."}],
        response_text="ok",
    )

    assert result["active"] == 1
    snapshot = store.graph_snapshot(project_id="Project PPMLX", query="canonical aliases")
    assert any(node["label"] == "ppmlx" for node in snapshot["nodes"])
    aliases = store.query_entity_aliases(alias="Project PPMLX", scope="project")
    assert len(aliases) == 1
    assert aliases[0]["entity_id"] == next(node["id"] for node in snapshot["nodes"] if node["label"] == "ppmlx")


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


def test_capture_chat_with_enqueue_extraction_creates_job_without_candidates(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    engine = MemoryEngine(store=store, extractor=FakeSingleCandidateExtractor(), enqueue_extraction=True)

    result = engine.capture_chat(
        request_id="req-async",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer async extraction."}],
        response_text="ok",
    )

    assert result["queued"] == 1
    assert result["candidates"] == 0
    assert result["active"] == 0
    stats = store.stats()
    assert stats["events"] == 1
    assert stats["candidates"] == 0
    jobs = store.list_extraction_jobs(status="queued")
    assert len(jobs) == 1
    assert jobs[0]["source_event_id"] == "req-async"
    assert jobs[0]["payload"]["event_id"] == "req-async"


def test_process_extraction_job_consumes_job_and_stores_candidate(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    engine = MemoryEngine(store=store, extractor=FakeSingleCandidateExtractor(), enqueue_extraction=True)
    engine.capture_chat(
        request_id="req-process-async",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer async extraction."}],
        response_text="ok",
    )

    result = engine.process_extraction_job(worker_id="test-worker")

    assert result is not None
    assert result["candidates"] == 1
    assert result["active"] == 1
    assert store.list_extraction_jobs(status="queued") == []
    completed = store.list_extraction_jobs(status="completed")
    assert len(completed) == 1
    assert completed[0]["result"]["active"] == 1
    rows = store.query_candidates(status="active")
    assert len(rows) == 1
    assert rows[0]["object"] == "async extraction"
    assert store.stats()["events"] == 1


def test_memory_cli_jobs_json_lists_jobs(tmp_home):
    reset_memory_store()
    store = MemoryStore(tmp_home / ".ppmlx" / "memory.db")
    store.init()
    store.enqueue_extraction_job(
        {"event_id": "req-cli-job", "messages": []},
        source_event_id="req-cli-job",
        job_id="job-cli",
    )
    reset_memory_store()

    result = CliRunner().invoke(app, ["memory", "jobs", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["job_id"] == "job-cli"
    assert data[0]["status"] == "queued"
    assert data[0]["source_event_id"] == "req-cli-job"
    reset_memory_store()


def test_capture_chat_records_event_when_extraction_fails(tmp_path):
    class FailingExtractor:
        max_candidates = 2

        def extract(self, event):
            raise RuntimeError("boom")

    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    engine = MemoryEngine(store=store, extractor=FailingExtractor())

    result = engine.capture_chat(
        request_id="req-extract-fails",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer concise answers."}],
        response_text="ok",
    )

    assert result["candidates"] == 0
    stats = store.stats()
    assert stats["events"] == 1
    assert stats["candidates"] == 0


def test_get_memory_engine_uses_configured_gemma_extractor(tmp_home, monkeypatch):
    import ppmlx.memory_extractors as memory_extractors

    class FakeGemmaExtractor:
        def __init__(self, *, model_name, max_candidates, max_tokens):
            self.model_name = model_name
            self.max_candidates = max_candidates
            self.max_tokens = max_tokens

        def extract(self, event):
            return []

    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTOR", "gemma_json")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MODEL", "fake-gemma")
    monkeypatch.setenv("PPMLX_MEMORY_MAX_CANDIDATES", "3")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MAX_TOKENS", "321")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_WORKERS", "2")
    monkeypatch.setattr(memory_extractors, "GemmaJsonMemoryExtractor", FakeGemmaExtractor)

    engine = get_memory_engine(tmp_home / "memory.db")

    assert isinstance(engine.extractor, FakeGemmaExtractor)
    assert engine.extractor.model_name == "fake-gemma"
    assert engine.extractor.max_candidates == 3
    assert engine.extractor.max_tokens == 321
    assert engine.extraction_workers == 2
    assert engine.parallel_extraction is True
    assert engine.enqueue_extraction is True


def test_parallel_gemma_extraction_merges_dedupes_and_writes_on_main_thread(tmp_path):
    main_thread_id = threading.get_ident()
    calls: list[list[dict]] = []
    lock = threading.Lock()

    class FakeGemmaExtractor:
        max_candidates = 2

        def extract(self, event):
            with lock:
                calls.append(event["messages"])
            content = event["messages"][0]["content"]
            if "tea" in content or "coffee" in content:
                drink = "coffee" if "coffee" in content else "tea"
                return [ShadowMemoryCandidate(
                    type="fact",
                    subject="user",
                    predicate="likes",
                    object=drink,
                    text=f"User likes {drink}.",
                    scope="global",
                    confidence=0.9,
                    source_quote=drink,
                    salience=0.8,
                )]
            return [ShadowMemoryCandidate(
                type="preference",
                subject="user",
                predicate="prefers",
                object="concise answers",
                text="User prefers concise answers.",
                scope="global",
                confidence=0.9,
                source_quote="concise answers",
                salience=0.8,
            )]

    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    original_store_candidate = store.store_candidate
    original_upsert_memory_edge = store.upsert_memory_edge

    def store_candidate_on_main_thread(*args, **kwargs):
        assert threading.get_ident() == main_thread_id
        return original_store_candidate(*args, **kwargs)

    def upsert_memory_edge_on_main_thread(*args, **kwargs):
        assert threading.get_ident() == main_thread_id
        return original_upsert_memory_edge(*args, **kwargs)

    store.store_candidate = store_candidate_on_main_thread
    store.upsert_memory_edge = upsert_memory_edge_on_main_thread
    engine = MemoryEngine(
        store=store,
        extractor=FakeGemmaExtractor(),
        extraction_workers=3,
        parallel_extraction=True,
    )

    result = engine.capture_chat(
        request_id="req-parallel",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[
            {"role": "user", "content": "I prefer concise answers."},
            {"role": "user", "content": "Please remember concise answers."},
            {"role": "user", "content": "Remember tea."},
            {"role": "user", "content": "Remember coffee."},
        ],
        response_text="ok",
    )

    assert len(calls) == 4
    assert all(len(messages) == 1 for messages in calls)
    assert result["candidates"] == 2
    assert result["active"] == 2
    rows = store.query_candidates(status="active")
    assert {row["object"] for row in rows} == {"concise answers", "tea"}
