"""Tests for ppmlx.server — FastAPI OpenAI-compatible API."""
from __future__ import annotations
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock all ppmlx modules that server.py tries to import lazily
for mod in [
    "ppmlx.engine", "ppmlx.engine_vlm", "ppmlx.engine_embed",
    "ppmlx.models", "ppmlx.db", "ppmlx.config", "ppmlx.memory",
    "ppmlx.schema",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Set up mock engine
mock_engine = MagicMock()
mock_engine.generate.return_value = ("Hello!", None, 10, 5)
mock_engine.stream_generate.return_value = iter(["Hello", " ", "world"])
mock_engine.list_loaded.return_value = []
# Mock tokenizer without tool calling so fallback parsing is used
mock_tokenizer = MagicMock()
mock_tokenizer.has_tool_calling = False
mock_engine.get_tokenizer.return_value = mock_tokenizer
sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

# Set up mock embed engine
mock_embed_engine = MagicMock()
mock_embed_engine.encode.return_value = [[0.1, 0.2, 0.3]]
sys.modules["ppmlx.engine_embed"].get_embed_engine = MagicMock(return_value=mock_embed_engine)

# Set up mock models
sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=lambda x: x)
sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=[])
sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value=[])
sys.modules["ppmlx.models"].is_vision_model = MagicMock(return_value=False)
sys.modules["ppmlx.models"].is_embed_model = MagicMock(return_value=False)

# Set up mock db
mock_db = MagicMock()
mock_db.get_stats.return_value = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}
sys.modules["ppmlx.db"].get_db = MagicMock(return_value=mock_db)

# Set up mock memory
sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

# Set up mock config
mock_config = MagicMock()
mock_config.logging = SimpleNamespace(snapshot_interval_seconds=60)
mock_config.server = SimpleNamespace(max_tools_tokens=12000)
mock_config.tool_awareness = SimpleNamespace(mode="no_tools_only")
mock_config.thinking = SimpleNamespace(
    enabled=True,
    default_reasoning_budget=2048,
    effort_to_budget=lambda effort: {"low": 256, "medium": 1024, "high": 8192}.get(effort.lower()),
)
sys.modules["ppmlx.config"].load_config = MagicMock(return_value=mock_config)

import pytest
from fastapi.testclient import TestClient
from ppmlx.server import app


@pytest.fixture
def client():
    from ppmlx import config as config_module

    mock_config.tool_awareness.mode = "no_tools_only"
    config_module.load_config = MagicMock(return_value=mock_config)
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_has_required_fields(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "loaded_models" in data
    assert "uptime_seconds" in data
    assert isinstance(data["loaded_models"], list)
    assert isinstance(data["uptime_seconds"], int)


def test_metrics_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data or "loaded_models" in data


def test_list_models_returns_200(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_chat_completion_nonstreaming(client):
    # Reset engine mock to return fresh values
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert content  # non-empty


def test_chat_completion_streaming_format(client):
    # Reset stream_generate mock
    def fresh_stream(*args, **kwargs):
        return iter(["Hello", " ", "world"])
    mock_engine.stream_generate = fresh_stream
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    body = response.text
    assert "data:" in body
    assert "data: [DONE]" in body


def test_completions_endpoint(client):
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/completions", json={
        "model": "test-model",
        "prompt": "Once upon a time",
        "max_tokens": 50,
    })
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    text = data["choices"][0]["text"]
    assert text  # non-empty


def test_embeddings_endpoint(client):
    mock_embed_engine.encode.return_value = [[0.1, 0.2, 0.3]]
    sys.modules["ppmlx.engine_embed"].get_embed_engine = MagicMock(return_value=mock_embed_engine)

    response = client.post("/v1/embeddings", json={
        "model": "embed-model",
        "input": "Hello world",
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, list)


def test_unknown_model_uses_name_directly(client):
    """Model not in aliases falls back to raw name; engine still called."""
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=Exception("not found"))
    mock_engine.generate.return_value = ("Response!", None, 5, 3)
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/chat/completions", json={
        "model": "unknown-model-xyz",
        "messages": [{"role": "user", "content": "test"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "unknown-model-xyz"

    # Restore
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=lambda x: x)


def test_cors_headers_present(client):
    client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS middleware sets Access-Control-Allow-Origin on actual requests too
    response2 = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in response2.headers


def test_chat_completion_injects_tool_awareness_without_tools(client):
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    mock_engine.generate.reset_mock()
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    })

    assert response.status_code == 200
    sent_messages = mock_engine.generate.call_args.args[1]
    assert sent_messages[0]["role"] == "system"
    assert "You do not have access to any external tools" in sent_messages[0]["content"]


def test_chat_completion_skips_tool_awareness_for_tools_in_no_tools_only_mode(client):
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    mock_engine.generate.reset_mock()
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run shell commands",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        "stream": False,
    })

    assert response.status_code == 200
    sent_messages = mock_engine.generate.call_args.args[1]
    system_content = sent_messages[0]["content"] if sent_messages and sent_messages[0]["role"] == "system" else ""
    assert "You have access ONLY to these tools" not in system_content


def test_inject_tool_awareness_returns_messages_unchanged_when_disabled(monkeypatch):
    from ppmlx.server import _inject_tool_awareness
    from ppmlx import config as config_module
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: SimpleNamespace(tool_awareness=SimpleNamespace(mode="off")),
    )
    messages = [{"role": "user", "content": "Hi"}]
    assert _inject_tool_awareness(messages, None) == messages


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------
def test_parse_tool_calls_single():
    from ppmlx.server import _parse_tool_calls
    text = '<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "ls"}}\n</tool_call>'
    remaining, calls = _parse_tool_calls(text)
    assert remaining == ""
    assert len(calls) == 1
    assert calls[0]["name"] == "exec_command"
    assert '"cmd"' in calls[0]["arguments"]


def test_parse_tool_calls_with_text():
    from ppmlx.server import _parse_tool_calls
    text = 'Let me check.\n<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "ls"}}\n</tool_call>\nDone.'
    remaining, calls = _parse_tool_calls(text)
    assert "Let me check" in remaining
    assert "Done" in remaining
    assert len(calls) == 1


def test_parse_tool_calls_multiple():
    from ppmlx.server import _parse_tool_calls
    text = (
        '<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "ls"}}\n</tool_call>'
        '<tool_call>\n{"name": "apply_patch", "arguments": {"patch": "..."}}\n</tool_call>'
    )
    remaining, calls = _parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "exec_command"
    assert calls[1]["name"] == "apply_patch"


def test_parse_tool_calls_no_calls():
    from ppmlx.server import _parse_tool_calls
    text = "Just a regular response with no tool calls."
    remaining, calls = _parse_tool_calls(text)
    assert remaining == text
    assert calls == []


def test_parse_tool_calls_fallback_json():
    """Fallback parser handles JSON inside <tool_call> blocks."""
    from ppmlx.server import _parse_tool_calls
    text = '<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "ls -la"}}\n</tool_call>'
    remaining, calls = _parse_tool_calls(text)
    assert remaining == ""
    assert len(calls) == 1
    assert calls[0]["name"] == "exec_command"
    import json
    args = json.loads(calls[0]["arguments"])
    assert args["cmd"] == "ls -la"


def test_parse_tool_calls_fallback_with_text():
    """Fallback parser preserves surrounding text."""
    from ppmlx.server import _parse_tool_calls
    text = (
        'Let me check.\n\n'
        '<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "pwd"}}\n</tool_call>'
    )
    remaining, calls = _parse_tool_calls(text)
    assert "Let me check" in remaining
    assert len(calls) == 1
    assert calls[0]["name"] == "exec_command"


def test_responses_with_tool_calls(client):
    """Responses API should parse <tool_call> blocks into function_call output items."""
    tool_response = '<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "ls -la"}}\n</tool_call>'
    mock_engine.generate.return_value = (tool_response, None, 10, 20)
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

    response = client.post("/v1/responses", json={
        "model": "test-model",
        "input": "list files",
        "stream": False,
        "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    # Should contain a function_call output item
    fc_items = [o for o in data["output"] if o["type"] == "function_call"]
    assert len(fc_items) == 1
    assert fc_items[0]["name"] == "exec_command"
    assert '"cmd"' in fc_items[0]["arguments"]


# ---------------------------------------------------------------------------
# Thinking model support (non-streaming)
# ---------------------------------------------------------------------------
def test_nonstream_chat_think_true(client):
    """Explicit think=True passes enable_thinking=True to engine."""
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    mock_engine.generate.reset_mock()
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "think": True,
    })
    assert response.status_code == 200
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["enable_thinking"] is True


def test_nonstream_chat_think_false(client):
    """Explicit think=False passes enable_thinking=False to engine."""
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    mock_engine.generate.reset_mock()
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "think": False,
    })
    assert response.status_code == 200
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["enable_thinking"] is False


def test_nonstream_chat_think_default_with_tools(client):
    """With tools and a reasoning_budget from config, thinking stays enabled.
    Without budget, tools disable thinking."""
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    mock_engine.generate.reset_mock()
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "tools": [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run shell commands",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    })
    assert response.status_code == 200
    call_kwargs = mock_engine.generate.call_args.kwargs
    # With config default_reasoning_budget > 0, thinking is enabled even with tools
    assert call_kwargs["enable_thinking"] is True


def test_nonstream_chat_completion_tokens_details(client):
    """Response includes completion_tokens_details with reasoning_tokens."""
    mock_engine.generate.return_value = ("Hello!", None, 10, 5)
    mock_engine.generate.reset_mock()
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    })
    assert response.status_code == 200
    data = response.json()
    usage = data["usage"]
    assert "completion_tokens_details" in usage
    assert "reasoning_tokens" in usage["completion_tokens_details"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 0


# ---------------------------------------------------------------------------
# Streaming thinking/reasoning tests
# ---------------------------------------------------------------------------

def _parse_sse_chunks(response_text):
    """Parse SSE text into a list of decoded JSON data objects."""
    chunks = []
    for line in response_text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))
    return chunks


def test_stream_chat_emits_reasoning_delta(client):
    """When model emits <think>reasoning</think>answer, streaming should
    produce reasoning deltas followed by content deltas."""
    def thinking_stream(*args, **kwargs):
        return iter(["<think>", "deep thought", "</think>", "the answer"])
    mock_engine.stream_generate = thinking_stream
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200
    chunks = _parse_sse_chunks(response.text)

    # Collect reasoning and content from deltas
    reasoning_parts = []
    content_parts = []
    for c in chunks:
        if "choices" not in c:
            continue
        delta = c["choices"][0].get("delta", {})
        if "reasoning" in delta and delta["reasoning"]:
            reasoning_parts.append(delta["reasoning"])
        if "content" in delta and delta["content"]:
            content_parts.append(delta["content"])

    assert "".join(reasoning_parts) == "deep thought"
    assert "".join(content_parts) == "the answer"


def test_stream_chat_think_false_no_reasoning(client):
    """With think=False, no reasoning deltas should be emitted — thinking
    tags are stripped by engine."""
    def plain_stream(*args, **kwargs):
        return iter(["just", " content"])
    mock_engine.stream_generate = plain_stream
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "think": False,
    })
    assert response.status_code == 200
    chunks = _parse_sse_chunks(response.text)

    for c in chunks:
        if "choices" not in c:
            continue
        delta = c["choices"][0].get("delta", {})
        assert "reasoning" not in delta or delta.get("reasoning") is None

    # Should have content
    content_parts = []
    for c in chunks:
        if "choices" not in c:
            continue
        delta = c["choices"][0].get("delta", {})
        if "content" in delta and delta["content"]:
            content_parts.append(delta["content"])
    assert "".join(content_parts) == "just content"


def test_stream_chat_no_think_tags_just_content(client):
    """Model without thinking tags should emit only content deltas."""
    def plain_stream(*args, **kwargs):
        return iter(["Hello", " world"])
    mock_engine.stream_generate = plain_stream
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)
    mock_config.tool_awareness.mode = "no_tools_only"

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200
    chunks = _parse_sse_chunks(response.text)

    reasoning_parts = []
    content_parts = []
    for c in chunks:
        if "choices" not in c:
            continue
        delta = c["choices"][0].get("delta", {})
        if "reasoning" in delta and delta["reasoning"]:
            reasoning_parts.append(delta["reasoning"])
        if "content" in delta and delta["content"]:
            content_parts.append(delta["content"])

    # State machine starts inside_think=True (Qwen3 template assumption),
    # so without </think> all output is emitted as reasoning.
    total = "".join(reasoning_parts) + "".join(content_parts)
    assert "Hello" in total
    assert "world" in total
