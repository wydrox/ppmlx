"""Tests for ppmlx.batch — BatchEngine with mocked TextEngine."""
from __future__ import annotations

import asyncio
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
    sys.modules["mlx_lm"] = fake
    return fake


# ---------------------------------------------------------------------------
# 1. BatchEngine queues and returns generate result
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_generate_returns_result():
    fake, _, _ = _make_fake_mlx_lm(generate_return="Hello there!")
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)

    result = await engine.generate(
        "some/model", [{"role": "user", "content": "hi"}]
    )
    assert isinstance(result, tuple)
    assert len(result) >= 4
    text, reasoning, pt, ct = result[:4]
    assert isinstance(text, str)
    assert text == "Hello there!"

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 2. BatchEngine streaming yields chunks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_stream_generate_yields_chunks():
    chunks = ["tok1", "tok2", "tok3"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)

    result = []
    async for chunk in engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ):
        result.append(chunk)

    assert result == chunks
    await engine.shutdown()


# ---------------------------------------------------------------------------
# 3. Multiple concurrent generate requests are all served
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_concurrent_generates():
    call_count = 0

    def _gen_side_effect(model, tokenizer, **kwargs):
        nonlocal call_count
        call_count += 1
        return f"response_{call_count}"

    fake, _, _ = _make_fake_mlx_lm()
    fake.generate = MagicMock(side_effect=_gen_side_effect)
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=10)

    # Launch 3 concurrent requests
    msgs = [{"role": "user", "content": "hi"}]
    tasks = [
        asyncio.create_task(engine.generate("some/model", msgs)),
        asyncio.create_task(engine.generate("some/model", msgs)),
        asyncio.create_task(engine.generate("some/model", msgs)),
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    # Each should have returned a valid result tuple (4 or 5 fields)
    for r in results:
        assert isinstance(r, tuple)
        assert len(r) >= 4

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 4. Result dispatched to correct future
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_results_dispatched_correctly():
    counter = [0]

    def _gen_side_effect(model, tokenizer, **kwargs):
        counter[0] += 1
        return f"answer_{counter[0]}"

    fake, _, _ = _make_fake_mlx_lm()
    fake.generate = MagicMock(side_effect=_gen_side_effect)
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)

    msgs = [{"role": "user", "content": "hi"}]
    r1 = await engine.generate("some/model", msgs)
    r2 = await engine.generate("some/model", msgs)

    # Results should be sequential since batch_window=0 forces immediate dispatch
    text1 = r1[0]
    text2 = r2[0]
    assert text1 != text2  # different responses
    assert "answer_" in text1
    assert "answer_" in text2

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 5. Streaming with multiple concurrent requests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_concurrent_streams():
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=["a", "b", "c"])
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=10)

    msgs = [{"role": "user", "content": "hi"}]

    async def collect_stream():
        parts = []
        async for chunk in engine.stream_generate("some/model", msgs):
            parts.append(chunk)
        return parts

    tasks = [
        asyncio.create_task(collect_stream()),
        asyncio.create_task(collect_stream()),
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 2
    for parts in results:
        assert len(parts) == 3
        assert parts == ["a", "b", "c"]

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 6. Queue size reflects pending requests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_queue_size():
    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)
    assert engine.queue_size == 0
    await engine.shutdown()


# ---------------------------------------------------------------------------
# 7. Proxy methods (list_loaded, load, unload, unload_all) work
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_proxy_methods():
    fake, _, _ = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)

    assert engine.list_loaded() == []

    engine.load("model/a")
    assert "model/a" in engine.list_loaded()

    engine.unload("model/a")
    assert engine.list_loaded() == []

    engine.load("model/b")
    engine.unload_all()
    assert engine.list_loaded() == []

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 8. Singleton get_batch_engine / reset_batch_engine
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_singleton():
    from ppmlx.batch import get_batch_engine, reset_batch_engine

    reset_batch_engine()
    e1 = get_batch_engine()
    e2 = get_batch_engine()
    assert e1 is e2

    reset_batch_engine()
    e3 = get_batch_engine()
    assert e3 is not e1


# ---------------------------------------------------------------------------
# 9. Error propagation from generate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_generate_error_propagation():
    fake, _, _ = _make_fake_mlx_lm()
    fake.generate = MagicMock(side_effect=RuntimeError("model failed"))
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)

    with pytest.raises(RuntimeError, match="model failed"):
        await engine.generate(
            "some/model", [{"role": "user", "content": "hi"}]
        )

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 10. Error propagation from stream_generate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_stream_error_propagation():
    def _bad_stream(model, tokenizer, **kwargs):
        yield "start"
        raise RuntimeError("stream failed")

    fake, _, _ = _make_fake_mlx_lm()
    fake.stream_generate = MagicMock(side_effect=_bad_stream)
    # Need to also make load work
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2, batch_window_ms=0)

    with pytest.raises(RuntimeError, match="stream failed"):
        async for _ in engine.stream_generate(
            "some/model", [{"role": "user", "content": "hi"}]
        ):
            pass

    await engine.shutdown()


# ---------------------------------------------------------------------------
# 11. batch_mode=False uses TextEngine directly (no BatchEngine overhead)
# ---------------------------------------------------------------------------
def test_batch_mode_flag():
    from ppmlx.server import set_batch_mode, is_batch_mode

    # Default should be False
    set_batch_mode(False)
    assert is_batch_mode() is False

    set_batch_mode(True)
    assert is_batch_mode() is True

    # Restore
    set_batch_mode(False)


# ---------------------------------------------------------------------------
# 12. Shutdown is idempotent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shutdown_idempotent():
    from ppmlx.batch import BatchEngine, reset_batch_engine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2)

    await engine.shutdown()
    await engine.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# 13. Engine property exposes TextEngine
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_engine_property():
    fake, _, _ = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.batch import BatchEngine, reset_batch_engine
    from ppmlx.engine import TextEngine
    reset_batch_engine()
    engine = BatchEngine(max_loaded=2)

    assert isinstance(engine.engine, TextEngine)
    await engine.shutdown()
