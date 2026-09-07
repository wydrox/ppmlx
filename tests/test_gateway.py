"""Tests for ppmlx.gateway — Smart API Gateway routing."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ppmlx.gateway import (
    GatewayConfig,
    RouteConfig,
    build_routing_table,
    create_gateway_app,
    match_route,
    resolve_backend,
    _openai_to_anthropic_messages,
    _anthropic_to_openai_response,
    _map_anthropic_stop,
)


# ── Unit tests: routing logic ────────────────────────────────────────────


class TestMatchRoute:
    def test_exact_match(self):
        routes = [RouteConfig(pattern="gpt-4", backend="openai")]
        assert match_route("gpt-4", routes) is not None
        assert match_route("gpt-4", routes).backend == "openai"

    def test_glob_match(self):
        routes = [RouteConfig(pattern="gpt-*", backend="openai")]
        result = match_route("gpt-4o", routes)
        assert result is not None
        assert result.backend == "openai"

    def test_wildcard_catchall(self):
        routes = [
            RouteConfig(pattern="gpt-*", backend="openai"),
            RouteConfig(pattern="*", backend="local"),
        ]
        result = match_route("llama3", routes)
        assert result is not None
        assert result.backend == "local"

    def test_first_match_wins(self):
        routes = [
            RouteConfig(pattern="gpt-*", backend="openai"),
            RouteConfig(pattern="gpt-4*", backend="anthropic"),  # never reached
        ]
        result = match_route("gpt-4o", routes)
        assert result.backend == "openai"

    def test_no_match(self):
        routes = [RouteConfig(pattern="gpt-*", backend="openai")]
        assert match_route("llama3", routes) is None

    def test_claude_pattern(self):
        routes = [RouteConfig(pattern="claude-*", backend="anthropic")]
        result = match_route("claude-sonnet-4-20250514", routes)
        assert result is not None
        assert result.backend == "anthropic"

    def test_empty_routes(self):
        assert match_route("anything", []) is None


class TestResolveBackend:
    def test_local_backend(self):
        route = RouteConfig(pattern="*", backend="local")
        backend, base_url, api_key = resolve_backend(route)
        assert backend == "local"
        assert base_url == ""
        assert api_key is None

    def test_openai_backend_default_url(self):
        route = RouteConfig(pattern="gpt-*", backend="openai")
        backend, base_url, _ = resolve_backend(route)
        assert backend == "openai"
        assert base_url == "https://api.openai.com/v1"

    def test_openai_backend_custom_url(self):
        route = RouteConfig(
            pattern="gpt-*", backend="openai", base_url="https://custom.api.com/v1"
        )
        backend, base_url, _ = resolve_backend(route)
        assert base_url == "https://custom.api.com/v1"

    def test_anthropic_backend_default_url(self):
        route = RouteConfig(pattern="claude-*", backend="anthropic")
        backend, base_url, _ = resolve_backend(route)
        assert backend == "anthropic"
        assert base_url == "https://api.anthropic.com/v1"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-test-123")
        route = RouteConfig(
            pattern="gpt-*", backend="openai", api_key_env="MY_KEY"
        )
        _, _, api_key = resolve_backend(route)
        assert api_key == "sk-test-123"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        route = RouteConfig(
            pattern="gpt-*", backend="openai", api_key_env="NONEXISTENT_KEY"
        )
        _, _, api_key = resolve_backend(route)
        assert api_key is None

    def test_unknown_backend_defaults_to_local(self):
        route = RouteConfig(pattern="*", backend="something-else")
        backend, _, _ = resolve_backend(route)
        assert backend == "local"


class TestBuildRoutingTable:
    def test_basic_table(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = GatewayConfig(
            routes=[
                RouteConfig(
                    pattern="gpt-*",
                    backend="openai",
                    api_key_env="OPENAI_API_KEY",
                    fallback="local",
                ),
                RouteConfig(pattern="*", backend="local"),
            ]
        )
        table = build_routing_table(config)
        assert len(table) == 2
        assert table[0]["pattern"] == "gpt-*"
        assert table[0]["backend"] == "openai"
        assert table[0]["api_key"] == "yes"
        assert table[0]["fallback"] == "local"
        assert table[1]["pattern"] == "*"
        assert table[1]["backend"] == "local"
        assert table[1]["api_key"] == "no"
        assert table[1]["fallback"] == "none"


# ── Unit tests: format translation ───────────────────────────────────────


class TestOpenAIToAnthropicMessages:
    def test_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        system, msgs = _openai_to_anthropic_messages(messages)
        assert system is None
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Hello"}

    def test_system_message_extracted(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, msgs = _openai_to_anthropic_messages(messages)
        assert system == "You are helpful."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_multiple_system_messages_merged(self):
        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hi"},
        ]
        system, msgs = _openai_to_anthropic_messages(messages)
        assert system == "Part 1\n\nPart 2"
        assert len(msgs) == 1

    def test_assistant_preserved(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        system, msgs = _openai_to_anthropic_messages(messages)
        assert system is None
        assert len(msgs) == 3
        assert msgs[1]["role"] == "assistant"


class TestAnthropicToOpenAIResponse:
    def test_basic_response(self):
        anthropic_resp = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        result = _anthropic_to_openai_response(anthropic_resp, "claude-3")
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-3"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_multiple_content_blocks(self):
        anthropic_resp = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _anthropic_to_openai_response(anthropic_resp, "claude-3")
        assert result["choices"][0]["message"]["content"] == "Hello world!"

    def test_max_tokens_stop_reason(self):
        anthropic_resp = {
            "content": [{"type": "text", "text": "..."}],
            "usage": {"input_tokens": 10, "output_tokens": 4096},
            "stop_reason": "max_tokens",
        }
        result = _anthropic_to_openai_response(anthropic_resp, "claude-3")
        assert result["choices"][0]["finish_reason"] == "length"


class TestMapAnthropicStop:
    def test_end_turn(self):
        assert _map_anthropic_stop("end_turn") == "stop"

    def test_max_tokens(self):
        assert _map_anthropic_stop("max_tokens") == "length"

    def test_tool_use(self):
        assert _map_anthropic_stop("tool_use") == "tool_calls"

    def test_unknown(self):
        assert _map_anthropic_stop("unknown") == "stop"


# ── Integration tests: gateway app with mocked backends ──────────────────


@pytest.fixture
def local_route_config():
    """Gateway config that routes everything to local."""
    return GatewayConfig(
        port=6768,
        local_server="http://127.0.0.1:6767",
        routes=[RouteConfig(pattern="*", backend="local")],
    )


@pytest.fixture
def multi_route_config():
    """Gateway config with multiple backends."""
    return GatewayConfig(
        port=6768,
        local_server="http://127.0.0.1:6767",
        routes=[
            RouteConfig(
                pattern="gpt-*",
                backend="openai",
                api_key_env="TEST_OPENAI_KEY",
            ),
            RouteConfig(
                pattern="claude-*",
                backend="anthropic",
                api_key_env="TEST_ANTHROPIC_KEY",
            ),
            RouteConfig(pattern="*", backend="local", fallback="openai"),
        ],
    )


@pytest.fixture
def gateway_client(local_route_config):
    """Test client for the gateway app."""
    app = create_gateway_app(local_route_config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def multi_gateway_client(multi_route_config):
    """Test client for the multi-route gateway app."""
    app = create_gateway_app(multi_route_config)
    with TestClient(app) as client:
        yield client


class TestGatewayHealthEndpoint:
    def test_health(self, gateway_client):
        resp = gateway_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "ppmlx-gateway"
        assert data["routes"] == 1


class TestGatewayRoutesEndpoint:
    def test_routes(self, multi_gateway_client):
        resp = multi_gateway_client.get("/gateway/routes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["routes"]) == 3
        assert data["routes"][0]["pattern"] == "gpt-*"
        assert data["routes"][0]["backend"] == "openai"
        assert data["routes"][1]["pattern"] == "claude-*"
        assert data["routes"][1]["backend"] == "anthropic"
        assert data["routes"][2]["pattern"] == "*"
        assert data["routes"][2]["backend"] == "local"


class TestGatewayChatCompletions:
    def test_local_routing_fails_gracefully(self, gateway_client):
        """When local server is not running, gateway returns 502."""
        resp = gateway_client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.5:0.8b",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # The local server isn't running, so this should fail with 502
        assert resp.status_code == 502

    def test_openai_routing_no_key(self, multi_gateway_client, monkeypatch):
        """When API key is missing, gateway returns 502."""
        monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
        resp = multi_gateway_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # No API key set, so OpenAI call should fail
        assert resp.status_code == 502

    def test_anthropic_routing_no_key(self, multi_gateway_client, monkeypatch):
        """When API key is missing, gateway returns 502."""
        monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
        resp = multi_gateway_client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 502


class TestGatewayFallback:
    def test_fallback_on_primary_failure(self, multi_gateway_client, monkeypatch):
        """When local fails, gateway tries the fallback backend.
        Since the fallback (openai) also has no key, we get a 502 with
        a message mentioning both backends."""
        monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
        resp = multi_gateway_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3",  # matches "*" -> local with fallback=openai
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 502
        data = resp.json()
        assert "All backends failed" in data["error"]["message"]


class TestGatewayModels:
    def test_models_endpoint(self, multi_gateway_client):
        """Models endpoint returns cloud model stubs even when local server is down."""
        resp = multi_gateway_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        # At minimum, cloud model pattern stubs are listed
        patterns = [m["id"] for m in data["data"]]
        assert "gpt-*" in patterns
        assert "claude-*" in patterns


class TestGatewayConfig:
    def test_default_config(self):
        cfg = GatewayConfig()
        assert cfg.port == 6768
        assert cfg.host == "127.0.0.1"
        assert cfg.local_server == "http://127.0.0.1:6767"
        assert cfg.routes == []

    def test_config_with_routes(self):
        cfg = GatewayConfig(
            routes=[
                RouteConfig(pattern="gpt-*", backend="openai"),
                RouteConfig(pattern="*", backend="local"),
            ]
        )
        assert len(cfg.routes) == 2

    def test_route_defaults(self):
        route = RouteConfig(pattern="*", backend="local")
        assert route.timeout == 30.0
        assert route.fallback == ""
        assert route.api_key_env == ""
        assert route.base_url == ""
