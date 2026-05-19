"""Tests for ppmlx.memory_engine and ppmlx.memory_store shadow write path."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.memory_engine import HybridMemoryExtractor, MemoryEngine, RuleBasedMemoryExtractor, ShadowMemoryCandidate, _event_extraction_chunks, get_memory_engine
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


def _seed_project_memory(store: MemoryStore, *, event_id: str, candidate_id: str, project_id: str) -> dict:
    store.record_event({
        "event_id": event_id,
        "endpoint": "/v1/chat/completions",
        "project_id": project_id,
        "request": {"messages": [{"role": "user", "content": "We decided to rebuild graph projection."}]},
        "response_text": "ok",
        "metadata": {},
    })
    candidate = {
        "candidate_id": candidate_id,
        "event_id": event_id,
        "type": "decision",
        "subject": project_id,
        "predicate": "decided",
        "object": "graph projection rebuild",
        "text": f"{project_id} decided graph projection rebuild.",
        "scope": "project",
        "confidence": 0.9,
        "source_quote": "decided graph projection rebuild",
        "salience": 0.8,
        "metadata": {},
    }
    store.store_candidate(candidate, {"status": "active", "reasons": [], "invalidates": []})
    store.upsert_memory_edge(candidate)
    return candidate


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


def test_plain_tool_noise_is_not_regex_extracted_as_project_memory(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-tool-noise",
        endpoint="/v1/chat/completions#compact",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        session_id="s1",
        messages=[
            {"role": "user", "content": "Goal: improve ppmlx evals."},
            {
                "role": "tool",
                "name": "bash",
                "content": "noise: unrelated travel budget 800 PLN\nnoise: app_1.py fixture output",
            },
            {"role": "assistant", "content": "Todo: add context coverage metrics to compact-eval."},
        ],
        response_text="ok",
    )

    assert result["active"] >= 2
    rows = store.query_candidates(status="active", project_id="ppmlx", session_id="s1", limit=50)
    texts = "\n".join(row["text"] for row in rows)
    assert "Goal: improve ppmlx evals" in texts
    assert "ppmlx todo: add context coverage metrics to compact-eval" in texts
    assert "budget = 800 PLN" not in texts


def test_multiple_todos_are_additive_not_contradictory(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-two-todos",
        endpoint="/v1/chat/completions#compact",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        session_id="s1",
        messages=[
            {"role": "assistant", "content": "Todo: rerun answerable real-session batch with include-content."},
            {"role": "assistant", "content": "Todo: add context coverage metrics to compact-eval."},
        ],
        response_text="ok",
    )

    assert result["active"] == 2
    rows = store.query_candidates(status="active", project_id="ppmlx", session_id="s1", limit=50)
    texts = "\n".join(row["text"] for row in rows)
    assert "ppmlx todo: rerun answerable real-session batch with include-content" in texts
    assert "ppmlx todo: add context coverage metrics to compact-eval" in texts


def test_response_summary_extracts_validation_commit_and_global_fix(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-validation-summary",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="devryn",
        session_id="s1",
        messages=[{"role": "user", "content": "czy możemy naprawić auth-race globalnie?"}],
        response_text=(
            "Tak — naprawiłem to globalnie w `src/app/providers.tsx`.\n\n"
            "Zmiana: `ConvexProviderWithAuth` nie zgłasza już `isAuthenticated: true`, dopóki token Convex/WorkOS nie jest gotowy.\n\n"
            "Walidacja:\n"
            "- `pnpm build` ✅\n"
            "- `pnpm exec eslint . --quiet` ✅\n\n"
            "Commit + push:\n"
            "- `bedc49e Gate Convex auth until token is ready`\n"
            "- `origin/dev` zaktualizowany."
        ),
    )

    assert result["active"] >= 5
    rows = store.query_candidates(status="active", project_id="devryn", session_id="s1", limit=50)
    texts = "\n".join(row["text"] for row in rows)
    assert "Global fix implemented in `src/app/providers.tsx`" in texts
    assert "Auth-race fix: `ConvexProviderWithAuth`" in texts
    assert "Validation result: `pnpm build` ✅" in texts
    assert "Validation result: `pnpm exec eslint . --quiet` ✅" in texts
    assert "Commit pushed: `bedc49e Gate Convex auth until token is ready`" in texts
    assert "Validation result: `origin/dev`" in texts


def test_response_summary_extracts_workflow_action_ledger(tmp_path):
    engine, store = _make_engine(tmp_path)

    result = engine.capture_chat(
        request_id="req-workflow-ledger",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        project_id="ppmlx",
        session_id="s1",
        messages=[{"role": "user", "content": "napraw workflow continuity"}],
        response_text=(
            "Current task: fix ppmlx workflow continuity.\n"
            "Next action: rerun quality-bench preflight with include-content.\n"
            "Blocker: workflow context missing for dzialaj turns.\n"
            "Ran `uv run pytest tests/test_quality_bench.py`."
        ),
    )

    assert result["active"] >= 4
    rows = store.query_candidates(status="active", project_id="ppmlx", session_id="s1", limit=50)
    texts = "\n".join(row["text"] for row in rows)
    assert "Current task: fix ppmlx workflow continuity" in texts
    assert "Next action: rerun quality-bench preflight with include-content" in texts
    assert "Blocker: workflow context missing for dzialaj turns" in texts
    assert "Command run: `uv run pytest tests/test_quality_bench.py`" in texts


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


def test_graph_projection_resolves_similar_entity_against_project_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    def add_candidate(candidate_id: str, event_id: str, project_id: str, subject: str, predicate: str, object_: str):
        store.record_event({
            "event_id": event_id,
            "endpoint": "/v1/chat/completions",
            "project_id": project_id,
            "request": {"messages": [{"role": "assistant", "content": f"{subject} {predicate} {object_}"}]},
            "metadata": {},
        })
        candidate = {
            "candidate_id": candidate_id,
            "event_id": event_id,
            "type": "fact",
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "text": f"{subject} {predicate}: {object_}.",
            "scope": "project",
            "confidence": 0.9,
            "source_quote": subject,
            "salience": 0.9,
            "metadata": {},
        }
        store.store_candidate(candidate, {"status": "active", "reasons": [], "invalidates": []})
        store.upsert_memory_edge(candidate)

    add_candidate("upsell-source", "upsell-event-1", "odealo", "UPSELL.md", "contains", "pricing backlog")
    add_candidate("upsell-followup", "upsell-event-2", "odealo", "upsell md", "needs", "margin estimates")

    snapshot = store.graph_snapshot(project_id="odealo", query="upsell margin", limit=20)
    upsell_nodes = [node for node in snapshot["nodes"] if node["label"] in {"upsell.md", "upsell md"}]
    assert [node["label"] for node in upsell_nodes] == ["upsell.md"]
    assert all(edge["from_name"] == "upsell.md" for edge in snapshot["edges"] if edge["relation"] in {"contains", "needs"})
    aliases = store.query_entity_aliases(alias="upsell md", scope="project")
    assert aliases and aliases[0]["entity_id"] == upsell_nodes[0]["id"]


def test_graph_projection_does_not_fuzzy_link_across_projects(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    for project_id, subject in (("project-a", "UPSELL.md"), ("project-b", "upsell md")):
        store.record_event({
            "event_id": f"{project_id}-event",
            "endpoint": "/v1/chat/completions",
            "project_id": project_id,
            "request": {"messages": [{"role": "assistant", "content": subject}]},
            "metadata": {},
        })
        candidate = {
            "candidate_id": f"{project_id}-candidate",
            "event_id": f"{project_id}-event",
            "type": "fact",
            "subject": subject,
            "predicate": "contains",
            "object": "pricing backlog",
            "text": f"{subject} contains pricing backlog.",
            "scope": "project",
            "confidence": 0.9,
            "source_quote": subject,
            "salience": 0.9,
            "metadata": {},
        }
        store.store_candidate(candidate, {"status": "active", "reasons": [], "invalidates": []})
        store.upsert_memory_edge(candidate)

    snapshot_b = store.graph_snapshot(project_id="project-b", query="upsell", limit=20)
    labels_b = {node["label"] for node in snapshot_b["nodes"]}
    assert "upsell md" in labels_b
    assert "upsell.md" not in labels_b


def test_graph_projection_anchors_generic_project_subject_to_project_node(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    store.record_event({
        "event_id": "workflow-anchor-event",
        "endpoint": "/v1/chat/completions",
        "project_id": "ppmlx",
        "request": {"messages": [{"role": "assistant", "content": "Next action: rerun quality-bench."}]},
        "metadata": {},
    })
    candidate = {
        "candidate_id": "workflow-anchor-candidate",
        "event_id": "workflow-anchor-event",
        "type": "workflow_state",
        "subject": "session",
        "predicate": "next_action",
        "object": "rerun quality-bench",
        "text": "Next action: rerun quality-bench.",
        "scope": "project",
        "confidence": 0.9,
        "source_quote": "Next action: rerun quality-bench.",
        "salience": 0.9,
        "metadata": {},
    }
    store.store_candidate(candidate, {"status": "active", "reasons": [], "invalidates": []})
    store.upsert_memory_edge(candidate)

    snapshot = store.graph_snapshot(project_id="ppmlx", query="quality-bench", limit=20)
    labels = {node["label"] for node in snapshot["nodes"]}
    assert "ppmlx" in labels
    assert "session" not in labels
    edge = next(edge for edge in snapshot["edges"] if edge["relation"] == "next_action")
    assert edge["from_name"] == "ppmlx"


def test_memory_store_rebuild_graph_projection_dry_run_is_non_destructive(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    _seed_project_memory(store, event_id="rebuild-dry-event", candidate_id="rebuild-dry-candidate", project_id="ppmlx")

    result = store.rebuild_graph_projection(project_id="ppmlx", dry_run=True)

    assert result["dry_run"] is True
    assert result["candidates"] == 1
    assert result["existing_edges"] == 1
    assert result["projectable_candidates"] == 1
    assert result["deleted_edges"] == 0
    assert store.stats()["edges"] == 1


def test_memory_store_confirmed_rebuild_recreates_only_matching_projection(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    _seed_project_memory(store, event_id="rebuild-p-event", candidate_id="rebuild-p-candidate", project_id="ppmlx")
    _seed_project_memory(store, event_id="rebuild-q-event", candidate_id="rebuild-q-candidate", project_id="other-project")
    with store._connect() as conn:
        conn.execute("DELETE FROM memory_edges WHERE source_candidate_id = ?", ("rebuild-p-candidate",))
        conn.commit()

    result = store.rebuild_graph_projection(project_id="ppmlx", dry_run=False)

    assert result["dry_run"] is False
    assert result["candidates"] == 1
    assert result["rebuilt_edges"] == 1
    assert store.stats()["edges"] == 2
    assert store.inspect_candidate("rebuild-p-candidate")["edges"][0]["relation"] == "decided"
    assert store.inspect_candidate("rebuild-q-candidate")["edges"]


def test_memory_store_enqueue_extraction_jobs_from_events(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    store.record_event({
        "event_id": "event-queue-a",
        "endpoint": "/v1/chat/completions",
        "project_id": "ppmlx",
        "request": {"messages": [{"role": "user", "content": "Remember queueing."}]},
        "response_text": "ok",
        "metadata": {},
    })
    store.record_event({
        "event_id": "event-queue-b",
        "endpoint": "/v1/chat/completions",
        "project_id": "other-project",
        "request": {"messages": [{"role": "user", "content": "Ignore."}]},
        "response_text": "ok",
        "metadata": {},
    })

    result = store.enqueue_extraction_jobs_from_events(project_id="ppmlx", limit=10, dry_run=False)

    assert result["events"] == 1
    assert result["queued"] == 1
    jobs = store.list_extraction_jobs(status="queued")
    assert len(jobs) == 1
    assert jobs[0]["source_event_id"] == "event-queue-a"
    assert jobs[0]["payload"]["event_id"] == "event-queue-a"
    assert jobs[0]["payload"]["messages"][0]["content"] == "Remember queueing."
    assert store.stats()["candidates"] == 0


def test_memory_store_prune_noisy_namespaces_dry_run(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    _seed_project_memory(store, event_id="prune-normal-event", candidate_id="prune-normal-candidate", project_id="ppmlx")
    _seed_project_memory(store, event_id="prune-noisy-event", candidate_id="prune-noisy-candidate", project_id="answer-quality-real-dogfood")

    result = store.prune_noisy_namespaces(dry_run=True)

    assert result["dry_run"] is True
    assert result["candidates"] == 1
    assert result["edges"] == 1
    assert result["candidate_ids"] == ["prune-noisy-candidate"]
    assert store.inspect_candidate("prune-noisy-candidate")["status"] == "active"
    assert store.stats()["edges"] == 2


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
    assert status["candidates"] == 1
    assert status["atoms"] == 0
    assert status["extraction_jobs"] == 0
    assert status["by_status"]["active"] == 1
    assert status["jobs_by_status"] == {}

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


def test_process_extraction_job_recovers_stale_claimed_job(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    queue_engine = MemoryEngine(store=store, extractor=FakeSingleCandidateExtractor(), enqueue_extraction=True)
    queue_engine.capture_chat(
        request_id="req-stale-claimed",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer async extraction."}],
        response_text="ok",
    )

    claimed = store.claim_extraction_job("crashed-worker")
    assert claimed is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE memory_extraction_jobs SET claimed_at = '2000-01-01T00:00:00.000' WHERE job_id = ?",
            (claimed["job_id"],),
        )
        conn.commit()

    worker_engine = MemoryEngine(
        store=store,
        extractor=FakeSingleCandidateExtractor(),
        extraction_timeout_seconds=1,
    )
    result = worker_engine.process_extraction_job(worker_id="recovery-worker")

    assert result is not None
    assert result["active"] == 1
    completed = store.get_extraction_job(claimed["job_id"])
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["worker_id"] == "recovery-worker"
    assert completed["attempts"] == 2
    assert store.list_extraction_jobs(status="claimed") == []


def test_process_extraction_job_renews_claim_while_running(tmp_path):
    class SlowExtractor:
        max_candidates = 1

        def extract(self, event):
            time.sleep(0.3)
            return [ShadowMemoryCandidate(
                type="preference",
                subject="user",
                predicate="prefers",
                object="slow extraction",
                text="User prefers slow extraction.",
                scope="global",
                confidence=0.9,
                source_quote="slow extraction",
                salience=0.8,
            )]

    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    queue_engine = MemoryEngine(store=store, extractor=SlowExtractor(), enqueue_extraction=True)
    queue_engine.capture_chat(
        request_id="req-heartbeat",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer slow extraction."}],
        response_text="ok",
    )
    worker_a = MemoryEngine(store=store, extractor=SlowExtractor(), extraction_timeout_seconds=0.15)
    worker_b = MemoryEngine(store=store, extractor=SlowExtractor(), extraction_timeout_seconds=0.15)

    results: list[dict | None] = []
    thread = threading.Thread(target=lambda: results.append(worker_a.process_extraction_job(worker_id="worker-a")))
    thread.start()
    time.sleep(0.22)

    assert worker_b.process_extraction_job(worker_id="worker-b") is None
    thread.join(timeout=2)

    assert results and results[0] is not None
    completed = store.list_extraction_jobs(status="completed")
    assert len(completed) == 1
    assert completed[0]["worker_id"] == "worker-a"
    assert completed[0]["attempts"] == 1
    assert store.list_extraction_jobs(status="claimed") == []


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


def test_event_extraction_chunks_use_token_budget_and_overlap():
    messages = [
        {"role": "user", "content": f"chunk-marker-{idx} " + ("x" * 90)}
        for idx in range(5)
    ]
    event = {"event_id": "chunked", "messages": messages, "response_text": ""}

    chunks = _event_extraction_chunks(event, max_input_tokens=80, overlap_tokens=40, max_chunks=10)

    assert len(chunks) > 1
    assert all(chunk["response_text"] == "" for chunk in chunks)
    assert all(chunk["metadata"]["extraction_chunk"]["max_input_tokens"] == 80 for chunk in chunks)
    assert chunks[0]["messages"][-1]["content"] == chunks[1]["messages"][0]["content"]


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


def test_hybrid_async_capture_runs_rule_based_sync_and_queues_model_job(tmp_path):
    class FakeModelExtractor:
        max_candidates = 4

        def extract(self, event):
            return [ShadowMemoryCandidate(
                type="fact",
                subject="user",
                predicate="remembered",
                object="model-only signal is gamma",
                text="Model extracted fact: model-only signal is gamma.",
                scope="global",
                confidence=0.9,
                source_quote="model-only signal is gamma",
                salience=0.9,
            )]

    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    engine = MemoryEngine(
        store=store,
        extractor=FakeModelExtractor(),
        sync_extractor=RuleBasedMemoryExtractor(),
        enqueue_extraction=True,
    )

    result = engine.capture_chat(
        request_id="req-hybrid",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "I prefer concise answers. The model-only signal is gamma."}],
        response_text="ok",
    )

    assert result["queued"] == 1
    assert result["active"] >= 1
    active_before_worker = store.query_candidates(status="active")
    assert any(row["object"] == "concise answers" for row in active_before_worker)

    worker = MemoryEngine(store=store, extractor=FakeModelExtractor())
    worker_result = worker.process_extraction_job(worker_id="hybrid-test")

    assert worker_result and worker_result["active"] >= 1
    active_after_worker = store.query_candidates(status="active")
    assert any(row["object"] == "model-only signal is gamma" for row in active_after_worker)


def test_hybrid_extractor_keeps_rule_candidates_when_model_fails():
    class FailingModelExtractor:
        max_candidates = 4

        def extract(self, event):
            raise RuntimeError("model unavailable")

    extractor = HybridMemoryExtractor(RuleBasedMemoryExtractor(), FailingModelExtractor())

    candidates = extractor.extract({
        "event_id": "hybrid-degrade",
        "project_id": "ppmlx",
        "messages": [{"role": "assistant", "content": "Todo: rerun quality-bench preflight."}],
        "response_text": "",
    })

    assert any(candidate.object == "rerun quality-bench preflight" for candidate in candidates)


def test_get_memory_engine_uses_hybrid_rule_plus_configured_model_extractor(tmp_home, monkeypatch):
    import ppmlx.memory_extractors as memory_extractors

    class FakeModelMemoryJsonExtractor:
        def __init__(self, *, model_name, max_candidates, max_tokens):
            self.model_name = model_name
            self.max_candidates = max_candidates
            self.max_tokens = max_tokens

        def extract(self, event):
            return []

    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTOR", "model_memory_json")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MODEL", "fake-extractor-model")
    monkeypatch.setenv("PPMLX_MEMORY_MAX_CANDIDATES", "3")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MAX_TOKENS", "321")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_WORKERS", "2")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_INPUT_TOKENS", "2048")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_OVERLAP_TOKENS", "256")
    monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MAX_CHUNKS", "7")
    monkeypatch.setattr(memory_extractors, "ModelMemoryJsonExtractor", FakeModelMemoryJsonExtractor)

    engine = get_memory_engine(tmp_home / "memory.db")

    assert isinstance(engine.extractor, FakeModelMemoryJsonExtractor)
    assert isinstance(engine.sync_extractor, RuleBasedMemoryExtractor)
    assert engine.extractor.model_name == "fake-extractor-model"
    assert engine.extractor.max_candidates == 3
    assert engine.extractor.max_tokens == 321
    assert engine.extraction_workers == 2
    assert engine.parallel_extraction is True
    assert engine.enqueue_extraction is True
    assert engine.extraction_input_tokens == 2048
    assert engine.extraction_overlap_tokens == 256
    assert engine.extraction_max_chunks_per_event == 7


def test_parallel_model_memory_json_extraction_merges_dedupes_and_writes_on_main_thread(tmp_path):
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
        extraction_input_tokens=24,
        extraction_overlap_tokens=0,
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
        response_text="",
    )

    assert len(calls) >= 3
    assert calls[0] == [{"role": "user", "content": "I prefer concise answers."}]
    assert result["candidates"] == 2
    assert result["active"] == 2
    rows = store.query_candidates(status="active")
    assert {row["object"] for row in rows} == {"concise answers", "tea"}
