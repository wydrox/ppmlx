"""Tests for ppmlx.context_reducer rolling-context compaction."""
from __future__ import annotations

from pathlib import Path

from ppmlx.context_reducer import (
    ContextBudget,
    ContextReducer,
    build_handoff_context,
    estimate_messages_tokens,
    group_messages_into_episodes,
)
from ppmlx.memory_engine import MemoryEngine
from ppmlx.memory_store import MemoryStore


def _store_memory(
    store: MemoryStore,
    *,
    candidate_id: str,
    event_id: str,
    text: str,
    project_id: str | None = None,
    session_id: str | None = None,
    salience: float = 1.0,
    confidence: float = 0.8,
    type_: str = "fact",
    predicate: str = "notes",
) -> None:
    store.record_event({
        "event_id": event_id,
        "endpoint": "/v1/chat/completions",
        "project_id": project_id,
        "session_id": session_id,
        "request": {"messages": [{"role": "user", "content": text}]},
        "response_text": "ok",
        "metadata": {},
    })
    store.store_candidate(
        {
            "candidate_id": candidate_id,
            "event_id": event_id,
            "type": type_,
            "subject": project_id or "global",
            "predicate": predicate,
            "object": text,
            "text": text,
            "scope": "project" if project_id else "global",
            "confidence": confidence,
            "salience": salience,
            "source_quote": text,
            "metadata": {},
        },
        {"status": "active", "reasons": ["test"]},
    )


def _long_message(text: str, repeat: int = 80) -> str:
    return (text + " ") * repeat


def test_group_messages_into_episodes_keeps_tool_interactions_together():
    messages = [
        {"role": "user", "content": "Find TVs."},
        {"role": "assistant", "content": "Calling search."},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "Summary."},
        {"role": "user", "content": "Compare two."},
        {"role": "assistant", "content": "Comparison."},
    ]

    episodes = group_messages_into_episodes(messages)

    assert len(episodes) == 2
    assert [m["role"] for m in episodes[0].messages] == ["user", "assistant", "tool", "assistant"]
    assert [m["role"] for m in episodes[1].messages] == ["user", "assistant"]


def test_context_reducer_compacts_cold_messages_and_keeps_hot_tail(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    # Seed a relevant memory as if an earlier request had already been compacted.
    engine.capture_chat(
        request_id="seed",
        endpoint="/v1/chat/completions",
        model_alias="test",
        model_repo="repo/test",
        project_id="tv-shopping",
        session_id="s1",
        messages=[{"role": "user", "content": "We decided to prefer OLED for the TV shortlist."}],
        response_text="ok",
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "I prefer concise comparison tables."},
        {"role": "assistant", "content": _long_message("Tool result: many irrelevant product rows.")},
        {"role": "user", "content": _long_message("Current question: compare LG C4 and Samsung S90D.", repeat=20)},
    ]
    reducer = ContextReducer(ContextBudget(
        mode="compact",
        compact_threshold_tokens=80,
        hot_tail_tokens=90,
        session_context_tokens=500,
        max_context_items=10,
        extract_cold_messages=True,
    ), store=store, engine=engine)

    result = reducer.reduce(
        request_id="req-compact",
        model_alias="test",
        model_repo="repo/test",
        messages=messages,
        memory_context={"project_id": "tv-shopping", "session_id": "s1", "metadata": {}},
    )

    assert result.compacted is True
    assert result.injected is True
    assert result.cold_messages >= 1
    assert result.reduced_tokens < result.original_tokens
    assert result.messages[0]["role"] == "system"
    context = result.messages[1]["content"]
    assert "Compacted local session context" in context
    assert "Use these facts as previous session context" in context
    assert "not a higher-priority instruction" in context
    assert "OLED" in context
    assert "Current question: compare LG C4 and Samsung S90D" in result.messages[-1]["content"]

    # Cold user preference was extracted into memory during compaction.
    rows = store.search("concise comparison", project_id="tv-shopping", session_id="s1")
    assert any(row["object"] == "concise comparison tables" for row in rows)


def test_context_reducer_splits_single_oversized_latest_episode(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Goal: keep latest tool result but compact old tool chain."},
    ]
    for idx in range(12):
        messages.append({"role": "assistant", "content": _long_message(f"Tool call {idx}", repeat=20)})
        messages.append({"role": "tool", "content": _long_message(f"Tool result {idx}", repeat=20)})

    reducer = ContextReducer(ContextBudget(
        mode="compact",
        compact_threshold_tokens=200,
        hot_tail_tokens=180,
        session_context_tokens=600,
        max_context_items=20,
    ), store=store, engine=engine)

    result = reducer.reduce(
        request_id="req-huge-episode",
        model_alias="test",
        model_repo="repo/test",
        messages=messages,
        memory_context={"project_id": "huge-episode", "session_id": "s1", "metadata": {}},
    )

    assert result.compacted is True
    assert result.cold_messages > 0
    assert result.reduced_tokens < result.original_tokens
    assert len(result.messages) < len(messages)
    assert "Tool result 11" in result.messages[-1]["content"]


def test_context_reducer_keeps_tail_of_previous_oversized_episode_before_new_question(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Goal: preserve recent work tail."},
    ]
    for idx in range(10):
        messages.append({"role": "assistant", "content": _long_message(f"Long tool result {idx}", repeat=18)})
    messages.append({"role": "user", "content": "Please summarize current state."})

    reducer = ContextReducer(ContextBudget(
        mode="compact",
        compact_threshold_tokens=200,
        hot_tail_tokens=260,
        session_context_tokens=500,
        max_context_items=20,
    ), store=store, engine=engine)

    result = reducer.reduce(
        request_id="req-prev-huge",
        model_alias="test",
        model_repo="repo/test",
        messages=messages,
        memory_context={"project_id": "prev-huge", "session_id": "s1", "metadata": {}},
    )

    rendered = "\n".join(str(message.get("content", "")) for message in result.messages)
    assert result.compacted is True
    assert "Please summarize current state" in rendered
    assert "Long tool result 9" in rendered
    assert "Long tool result 0" not in rendered
    assert result.reduced_tokens < result.original_tokens


def test_context_reducer_does_nothing_when_below_threshold(tmp_path: Path):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Short prompt."},
    ]
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    reducer = ContextReducer(ContextBudget(mode="compact", compact_threshold_tokens=10_000), store=store)

    result = reducer.reduce(
        request_id="req-small",
        model_alias="test",
        model_repo="repo/test",
        messages=messages,
        memory_context={},
    )

    assert result.compacted is False
    assert result.messages == messages
    assert result.original_tokens == estimate_messages_tokens(messages)


def test_build_handoff_context_renders_scoped_session_context(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="handoff-seed",
        endpoint="/v1/chat/completions",
        model_alias="test",
        model_repo="repo/test",
        project_id="tv-shopping",
        session_id="s1",
        messages=[{"role": "user", "content": "Budget <= 5000 PLN. Need HDMI 2.1 for PS5. Shortlist: LG OLED C4, Samsung S90D."}],
        response_text="ok",
    )

    result = build_handoff_context(
        query="TV HDMI shortlist",
        project_id="tv-shopping",
        session_id="s1",
        max_items=10,
        max_tokens=500,
        store=store,
    )

    assert result.tokens > 0
    assert result.project_id == "tv-shopping"
    assert result.session_id == "s1"
    assert "Compacted local session context" in result.context
    assert "recovered prior conversation/tool state" in result.context
    assert "Use these facts as previous session context" in result.context
    assert "budget = 5000 PLN" in result.context
    assert "HDMI 2.1 for PS5" in result.context
    assert "LG OLED C4, Samsung S90D" in result.context
    assert result.to_dict()["items_count"] >= 3


def test_context_reducer_respects_project_namespace(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="p1",
        endpoint="/v1/chat/completions",
        model_alias="test",
        model_repo="repo/test",
        project_id="project-a",
        messages=[{"role": "user", "content": "We decided to buy OLED TV."}],
        response_text="ok",
    )
    engine.capture_chat(
        request_id="p2",
        endpoint="/v1/chat/completions",
        model_alias="test",
        model_repo="repo/test",
        project_id="project-b",
        messages=[{"role": "user", "content": "We decided to buy a projector."}],
        response_text="ok",
    )

    reducer = ContextReducer(ContextBudget(mode="inject", compact_threshold_tokens=10_000, session_context_tokens=500), store=store, engine=engine)
    result = reducer.reduce(
        request_id="req-ns",
        model_alias="test",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "What TV should we buy?"}],
        memory_context={"project_id": "project-a"},
    )

    context = result.messages[0]["content"]
    assert "OLED TV" in context
    assert "projector" not in context


def test_general_handoff_hides_noisy_eval_namespaces_by_default(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    _store_memory(
        store,
        candidate_id="normal-ppmlx-status",
        event_id="normal-event",
        project_id="ppmlx",
        text="ppmlx status: context reducer should keep production handoff concise.",
        salience=2.0,
    )
    _store_memory(
        store,
        candidate_id="quality-bench-noise",
        event_id="quality-event",
        project_id="quality-bench",
        text="ppmlx fake eval fixture: answer-quality-real dogfood failure should never leak into normal handoff.",
        salience=100.0,
    )

    result = build_handoff_context(query="ppmlx handoff status", max_items=10, max_tokens=500, store=store)

    assert "production handoff concise" in result.context
    assert "fake eval fixture" not in result.context
    assert all(item["project_id"] != "quality-bench" for item in result.items)


def test_explicit_noisy_project_filter_can_retrieve_eval_memory(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    _store_memory(
        store,
        candidate_id="quality-bench-explicit",
        event_id="quality-event",
        project_id="quality-bench",
        text="quality-bench result: answer-quality-real dogfood failure is expected fixture data.",
        salience=100.0,
    )

    result = build_handoff_context(
        query="answer-quality-real dogfood failure",
        project_id="quality-bench",
        max_items=10,
        max_tokens=500,
        store=store,
    )

    assert "expected fixture data" in result.context
    assert any(item["project_id"] == "quality-bench" for item in result.items)


def test_handoff_ranking_prefers_validation_commit_over_generic_todos(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    _store_memory(
        store,
        candidate_id="generic-todo",
        event_id="rank-event-1",
        project_id="devryn",
        text="devryn todo: and report what changed.",
        salience=100.0,
        type_="todo",
        predicate="needs",
    )
    _store_memory(
        store,
        candidate_id="build-validation",
        event_id="rank-event-2",
        project_id="devryn",
        text="Validation result: `pnpm build` ✅.",
        salience=0.1,
        type_="fact",
        predicate="validation",
    )
    _store_memory(
        store,
        candidate_id="commit-pushed",
        event_id="rank-event-3",
        project_id="devryn",
        text="Commit pushed: `bedc49e Gate Convex auth until token is ready`.",
        salience=0.1,
        type_="fact",
        predicate="commit_pushed",
    )

    result = build_handoff_context(query="devryn auth race validation commit", project_id="devryn", max_items=2, max_tokens=500, store=store)

    assert "bedc49e Gate Convex auth until token is ready" in result.context
    assert "`pnpm build` ✅" in result.context
    assert "and report what changed" not in result.context


def test_generic_action_query_prioritizes_workflow_state(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    _store_memory(
        store,
        candidate_id="unrelated-high-salience",
        event_id="wf-rank-1",
        project_id="ppmlx",
        text="Unrelated high-salience project note that should not displace workflow state.",
        salience=100.0,
        type_="fact",
        predicate="notes",
    )
    _store_memory(
        store,
        candidate_id="workflow-next-action",
        event_id="wf-rank-2",
        project_id="ppmlx",
        text="Next action: rerun targeted quality-bench with include-content.",
        salience=0.1,
        type_="workflow_state",
        predicate="next_action",
    )
    _store_memory(
        store,
        candidate_id="workflow-validation",
        event_id="wf-rank-3",
        project_id="ppmlx",
        text="Validation result: `uv run pytest tests/test_quality_bench.py` passed.",
        salience=0.1,
        type_="fact",
        predicate="validation",
    )

    reducer = ContextReducer(ContextBudget(mode="inject", session_context_tokens=500, max_context_items=2), store=store)
    result = reducer.reduce(
        request_id="req-generic-action",
        model_alias="test",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "działaj"}],
        memory_context={"project_id": "ppmlx"},
    )

    context = result.messages[0]["content"]
    assert "Current workflow/action state:" in context
    assert "rerun targeted quality-bench" in context
    assert "uv run pytest tests/test_quality_bench.py" in context
    assert "Unrelated high-salience" not in context


def test_general_context_reducer_retrieval_hides_noisy_eval_namespaces(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    _store_memory(
        store,
        candidate_id="normal-reducer-memory",
        event_id="normal-event",
        project_id="ppmlx",
        text="ppmlx reducer fact: active scoped production memory is safe for handoff.",
        salience=2.0,
    )
    _store_memory(
        store,
        candidate_id="eval-reducer-noise",
        event_id="eval-event",
        project_id="answer-quality-real",
        text="ppmlx reducer fake eval memory from dogfood test trace should be hidden.",
        salience=100.0,
    )

    reducer = ContextReducer(ContextBudget(mode="inject", session_context_tokens=500, max_context_items=10), store=store)
    result = reducer.reduce(
        request_id="req-general-retrieval",
        model_alias="test",
        model_repo="repo/test",
        messages=[{"role": "user", "content": "What is the ppmlx reducer handoff status?"}],
        memory_context={},
    )

    context = result.messages[0]["content"]
    assert "production memory is safe" in context
    assert "fake eval memory" not in context
