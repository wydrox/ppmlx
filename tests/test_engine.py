"""Tests for ppmlx.engine — all mlx_lm calls are mocked."""
from __future__ import annotations
import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_fake_mlx_lm(generate_return="Hello!", stream_chunks=None):
    """Build a fake mlx_lm module with controllable return values."""
    fake = types.ModuleType("mlx_lm")
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "mocked prompt"
    mock_tokenizer.encode.return_value = [1, 2, 3]

    fake.load = MagicMock(return_value=(mock_model, mock_tokenizer))
    fake.generate = MagicMock(return_value=generate_return)

    if stream_chunks is None:
        stream_chunks = ["Hello", ", ", "world!"]

    def _stream_gen(model, tokenizer, **kwargs):
        for chunk in stream_chunks:
            obj = MagicMock()
            obj.text = chunk
            yield obj

    fake.stream_generate = MagicMock(side_effect=_stream_gen)
    return fake, mock_model, mock_tokenizer


def _install_fake(fake):
    """Install fake mlx_lm into sys.modules and return it."""
    sys.modules["mlx_lm"] = fake
    return fake


# ---------------------------------------------------------------------------
# 1. LRU eviction
# ---------------------------------------------------------------------------
def test_lru_eviction():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.load("model/a")
    engine.load("model/b")
    engine.load("model/c")  # should evict model/a

    loaded = engine.list_loaded()
    assert "model/a" not in loaded
    assert "model/b" in loaded
    assert "model/c" in loaded
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# 2. Load already-loaded model moves to end
# ---------------------------------------------------------------------------
def test_load_already_loaded_moves_to_end():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.load("model/a")
    engine.load("model/b")
    engine.load("model/a")  # touch A → B becomes LRU

    # Now loading model/c should evict B (LRU), not A
    engine.load("model/c")
    loaded = engine.list_loaded()
    assert "model/b" not in loaded
    assert "model/a" in loaded
    assert "model/c" in loaded


# ---------------------------------------------------------------------------
# 3. generate calls mlx
# ---------------------------------------------------------------------------
def test_generate_calls_mlx():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="Hello!")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    text, reasoning, pt, ct, rt = engine.generate(
        "some/model", [{"role": "user", "content": "hi"}]
    )

    fake.generate.assert_called_once()
    call_kwargs = fake.generate.call_args
    assert call_kwargs is not None
    # model and tokenizer are positional args
    args, kwargs = call_kwargs
    assert args[0] is mock_model
    assert args[1] is mock_tokenizer


# ---------------------------------------------------------------------------
# 4. generate returns correct tuple structure
# ---------------------------------------------------------------------------
def test_generate_returns_text():
    fake, _, _ = _make_fake_mlx_lm(generate_return="Hello there!")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = engine.generate("some/model", [{"role": "user", "content": "hi"}])
    assert isinstance(result, tuple)
    assert len(result) == 5
    text, reasoning, pt, ct, rt = result
    assert isinstance(text, str)
    assert reasoning is None
    assert isinstance(pt, int)
    assert isinstance(ct, int)
    assert isinstance(rt, int)


# ---------------------------------------------------------------------------
# 5. generate strips <think> blocks
# ---------------------------------------------------------------------------
def test_generate_strips_thinking():
    fake, _, _ = _make_fake_mlx_lm(generate_return="<think>reasoning here</think>answer text")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    text, reasoning, pt, ct, rt = engine.generate(
        "some/model", [{"role": "user", "content": "hi"}], strip_thinking=True
    )
    assert text == "answer text"
    assert reasoning == "reasoning here"


# ---------------------------------------------------------------------------
# 6. stream_generate yields chunks
# ---------------------------------------------------------------------------
def test_stream_generate_yields_chunks():
    chunks = ["tok1", "tok2", "tok3"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = list(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert result == chunks


# ---------------------------------------------------------------------------
# 7. _strip_thinking helper
# ---------------------------------------------------------------------------
def test_strip_thinking_helper():
    from ppmlx.engine import _strip_thinking

    # Basic case
    text, reasoning = _strip_thinking("<think>inner reasoning</think>final answer")
    assert text == "final answer"
    assert reasoning == "inner reasoning"

    # Multiline thinking
    text, reasoning = _strip_thinking("<think>\nline1\nline2\n</think>response")
    assert text == "response"
    assert reasoning == "line1\nline2"

    # Multiple think blocks — only first matched by search
    text, reasoning = _strip_thinking("<think>r1</think>mid<think>r2</think>end")
    # sub removes all, search finds first
    assert "mid" in text or "end" in text
    assert reasoning == "r1"


# ---------------------------------------------------------------------------
# 8. No thinking tag — text unchanged
# ---------------------------------------------------------------------------
def test_no_thinking_unchanged():
    from ppmlx.engine import _strip_thinking

    original = "This is a plain response."
    text, reasoning = _strip_thinking(original)
    assert text == original
    assert reasoning is None


# ---------------------------------------------------------------------------
# 8b. Template-injected think block (no opening <think>)
# ---------------------------------------------------------------------------
def test_strip_thinking_no_opening_tag():
    """Qwen3 template injects <think> into the prompt, so the model output
    starts inside the think block — only </think> appears in the text."""
    from ppmlx.engine import _strip_thinking

    text, reasoning = _strip_thinking("reasoning content\n</think>\nactual answer")
    assert text == "actual answer"
    assert reasoning == "reasoning content"

    # Only closing tag, no reasoning
    text, reasoning = _strip_thinking("</think>answer only")
    assert text == "answer only"
    assert reasoning is None


# ---------------------------------------------------------------------------
# 9. unload
# ---------------------------------------------------------------------------
def test_unload():
    fake, _, _ = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.load("model/a")
    assert "model/a" in engine.list_loaded()

    result = engine.unload("model/a")
    assert result is True
    assert engine.list_loaded() == []

    # Unloading again returns False
    assert engine.unload("model/a") is False


# ---------------------------------------------------------------------------
# 10. unload_all
# ---------------------------------------------------------------------------
def test_unload_all():
    fake, _, _ = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=3)

    engine.load("model/a")
    engine.load("model/b")
    assert len(engine.list_loaded()) == 2

    engine.unload_all()
    assert engine.list_loaded() == []


# ---------------------------------------------------------------------------
# 11. list_loaded empty on fresh engine
# ---------------------------------------------------------------------------
def test_list_loaded_empty():
    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)
    assert engine.list_loaded() == []


# ---------------------------------------------------------------------------
# 12. Singleton get_engine / reset_engine
# ---------------------------------------------------------------------------
def test_singleton():
    from ppmlx.engine import get_engine, reset_engine

    reset_engine()
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2

    reset_engine()
    e3 = get_engine()
    assert e3 is not e1


# ---------------------------------------------------------------------------
# 13. _apply_chat_template fallback (no apply_chat_template method)
# ---------------------------------------------------------------------------
def test_apply_chat_template_fallback():
    from ppmlx.engine import TextEngine, LoadedModel, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    # Tokenizer without apply_chat_template
    tokenizer = object()  # plain object has no apply_chat_template

    lm = LoadedModel(repo_id="test/model", model=MagicMock(), tokenizer=tokenizer)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = engine._apply_chat_template(lm, messages)

    assert "system: You are helpful." in prompt
    assert "user: Hello!" in prompt
    assert prompt.endswith("assistant:")


# ---------------------------------------------------------------------------
# 14. apply_chat_template called on tokenizer when available
# ---------------------------------------------------------------------------
def test_chat_template_applied():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="response")
    mock_tokenizer.apply_chat_template.return_value = "TEMPLATED PROMPT"
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    messages = [{"role": "user", "content": "test"}]
    engine.generate("some/model", messages)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def test_strict_tool_template_does_not_silently_drop_tools():
    from ppmlx.engine import LoadedModel, TextEngine

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = TypeError("tools are unsupported")
    engine = TextEngine(max_loaded=1)
    loaded = LoadedModel(repo_id="test/model", model=MagicMock(), tokenizer=tokenizer)

    with pytest.raises(RuntimeError, match="cannot accept tool definitions"):
        engine._apply_chat_template(
            loaded,
            [{"role": "user", "content": "Use the tool."}],
            tools=[{"type": "function", "function": {"name": "bash"}}],
            strict_tools=True,
        )


# ---------------------------------------------------------------------------
# 15. stream_generate strips <think> blocks by default
# ---------------------------------------------------------------------------
def test_stream_generate_strips_think_blocks():
    chunks = ["<think>", "reasoning", " here", "</think>", "answer", " text"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert "<think>" not in result
    assert "</think>" not in result
    assert "reasoning" not in result
    assert "answer text" == result


def test_stream_generate_strips_think_split_across_chunks():
    # Tags split across token boundaries
    chunks = ["<thi", "nk>", "internal", "</thi", "nk>", "visible"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert result == "visible"


def test_stream_generate_no_think_tags_passes_through():
    chunks = ["Hello", " ", "world"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert result == "Hello world"


def test_stream_generate_strip_thinking_false():
    chunks = ["<think>", "reason", "</think>", "answer"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}],
        strip_thinking=False,
    ))
    assert "<think>" in result
    assert "reason" in result


# ---------------------------------------------------------------------------
# 16. _is_thinking_model detects <think> in chat_template
# ---------------------------------------------------------------------------
def test_is_thinking_model_true():
    from ppmlx.engine import _is_thinking_model

    tokenizer = MagicMock()
    tokenizer.chat_template = "...some template with <think> tag..."
    assert _is_thinking_model(tokenizer) is True


def test_is_thinking_model_false():
    from ppmlx.engine import _is_thinking_model

    tokenizer = MagicMock()
    tokenizer.chat_template = "plain template without thinking"
    assert _is_thinking_model(tokenizer) is False

    # No chat_template attribute at all
    tokenizer2 = MagicMock(spec=[])
    assert _is_thinking_model(tokenizer2) is False


# ---------------------------------------------------------------------------
# 17. generate retries on empty answer after thinking
# ---------------------------------------------------------------------------
def test_generate_empty_answer_retries():
    """When strip_thinking produces empty text, generate() retries with
    enable_thinking=False."""
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(
        generate_return="<think>long reasoning</think>"
    )
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    # First call returns thinking-only, second call (retry) returns real answer
    fake.generate.side_effect = [
        "<think>long reasoning</think>",
        "actual answer",
    ]

    result = engine.generate("some/model", [{"role": "user", "content": "hi"}])
    assert result.text == "actual answer"
    assert fake.generate.call_count == 2


# ---------------------------------------------------------------------------
# 18. reasoning_budget caps thinking in stream
# ---------------------------------------------------------------------------
def test_reasoning_budget_stream():
    """When reasoning_budget is set, thinking is capped and text is yielded."""
    # Drip thinking content one chunk at a time so the budget check fires
    # before the </think> close tag arrives.
    slow_chunks = ["<think>"] + ["x" * 20 for _ in range(20)] + ["</think>", "answer"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=slow_chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    # Budget = 5 tokens => 20 chars. After ~20 chars of thinking the budget is hit.
    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}],
        reasoning_budget=5,
    ))
    assert "answer" in result


# ---------------------------------------------------------------------------
# 19. GenerateResult unpacks as 4-tuple (backward compatibility)
# ---------------------------------------------------------------------------
def test_generate_result_unpacks_as_4tuple():
    """GenerateResult supports named access and extended unpacking."""
    from ppmlx.engine import GenerateResult

    r = GenerateResult(
        text="hello",
        reasoning="thought",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=3,
    )

    # Extended unpacking (existing callers can use text, reasoning, pt, ct, *_ = r)
    text, reasoning, pt, ct, *rest = r
    assert text == "hello"
    assert reasoning == "thought"
    assert pt == 10
    assert ct == 5
    assert rest == [3]

    # Named access
    assert r.text == "hello"
    assert r.reasoning_tokens == 3

    # Default value for reasoning_tokens
    r2 = GenerateResult("hi", None, 1, 1)
    assert r2.reasoning_tokens == 0


def test_strip_thinking_gemma_channel_markers():
    from ppmlx.engine import _strip_thinking

    text, reasoning = _strip_thinking(
        "<|channel>thought\nprivate reasoning<channel|>visible answer"
    )
    assert text == "visible answer"
    assert reasoning == "private reasoning"

    text, reasoning = _strip_thinking("private reasoning<channel|>visible answer")
    assert text == "visible answer"
    assert reasoning == "private reasoning"


def test_stream_generate_strips_gemma_channel_markers():
    chunks = ["<|chan", "nel>thought\nprivate", "<chan", "nel|>", "visible"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = list(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}], strip_thinking=True
    ))
    assert "".join(result) == "visible"


def test_gemma4_extra_weight_fallback_is_quiet(tmp_path, monkeypatch, caplog):
    fake = types.ModuleType("mlx_lm")
    fake.__path__ = []
    fake.load = MagicMock(side_effect=ValueError("Received 140 parameters not in model: x"))

    fake_utils = types.ModuleType("mlx_lm.utils")
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    fake_utils.load_model = MagicMock(return_value=(mock_model, {"eos_token_id": 1}))
    fake_utils.load_tokenizer = MagicMock(return_value=mock_tokenizer)

    sys.modules["mlx_lm"] = fake
    sys.modules["mlx_lm.utils"] = fake_utils

    model_dir = tmp_path / "gemma4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"gemma4"}')

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    monkeypatch.setattr("ppmlx.engine._resolve_model_path", lambda _: str(model_dir))
    caplog.set_level("WARNING", logger="ppmlx.engine")

    lm = TextEngine(max_loaded=1).load("gemma-4-e2b")

    assert lm.model is mock_model
    fake_utils.load_model.assert_called_once_with(model_dir, strict=False)
    assert not caplog.records
