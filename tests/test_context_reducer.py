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
    assert result.messages[-1]["content"].startswith("Current question")

    # Cold user preference was extracted into memory during compaction.
    rows = store.search("concise comparison", project_id="tv-shopping", session_id="s1")
    assert any(row["object"] == "concise comparison tables" for row in rows)


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
