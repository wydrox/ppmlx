"""Regression tests for the routing UX gaps (base_url, loud policy, causes)."""
from __future__ import annotations

import os
import tomllib

import httpx
import pytest

from ppmlx.router import (
    RequiredCapabilities,
    RouteCandidate,
    policy_from_dict,
)
from ppmlx.routing_service import RouteInput, RoutingService


POLICY_TOML = """
[routes]
version = "1"

[routes.aliases]
or-model = ["openrouter", "anthropic/claude-3.5-haiku"]

[[routes.entries]]
key = "openai-chat:or-model"
candidates = [
  { provider = "openrouter", model = "anthropic/claude-3.5-haiku", base_url = "https://openrouter.ai/api/v1" },
]
"""


def _route_input(public_model: str = "or-model") -> RouteInput:
    return RouteInput(
        public_model=public_model,
        harness="openai-chat",
        harness_version="test",
        protocol="openai-chat",
        required=RequiredCapabilities(),
        policy_version="1",
        health_snapshot_id="snap",
        request_id="req_test_1",
        session_id="sess-1",
    )


class TestCandidateSchema:
    def test_base_url_and_kind_parse_from_toml(self):
        policy = policy_from_dict(tomllib.loads(POLICY_TOML))
        entry = next(iter(policy.entries.values()))
        candidate = entry.candidates[0]
        assert candidate.provider_id == "openrouter"
        assert candidate.base_url == "https://openrouter.ai/api/v1"
        assert candidate.provider_kind == "openai"

    def test_provider_kind_anthropic(self):
        candidate = RouteCandidate(
            provider_id="gw", model="m", provider_kind="anthropic"
        )
        assert candidate.provider_kind == "anthropic"

    def test_unknown_provider_kind_rejected(self):
        with pytest.raises(ValueError):
            RouteCandidate(provider_id="gw", model="m", provider_kind="cohere")

    def test_bad_base_url_rejected(self):
        with pytest.raises(ValueError):
            RouteCandidate(provider_id="gw", model="m", base_url="ftp://x")


class TestProviderConstruction:
    def _policy(self):
        return policy_from_dict(tomllib.loads(POLICY_TOML))

    def test_openrouter_provider_gets_base_url_and_env_key(self):
        from ppmlx.server import _remote_providers_for_policy

        providers = _remote_providers_for_policy(self._policy())
        provider = providers["openrouter"]
        assert provider.provider_id == "openrouter"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._env_key == "OPENROUTER_API_KEY"

    def test_anthropic_kind_builds_anthropic_adapter(self):
        from ppmlx.server import _remote_providers_for_policy

        toml = """
[routes]
version = "1"
[[routes.entries]]
key = "openai-chat:m"
candidates = [
  { provider = "gw", model = "claude-3", provider_kind = "anthropic", base_url = "https://gw.example/v1" },
]
"""
        providers = _remote_providers_for_policy(policy_from_dict(tomllib.loads(toml)))
        from ppmlx.providers.anthropic import AnthropicProvider

        assert isinstance(providers["gw"], AnthropicProvider)
        assert providers["gw"]._base_url == "https://gw.example/v1"
        assert providers["gw"]._env_key == "GW_API_KEY"


class TestLoudPolicyFailure:
    def test_broken_policy_sets_loud_failure(self, monkeypatch, tmp_path):
        import ppmlx.server as server

        bad = tmp_path / "routes.toml"
        bad.write_text('[routes]\nversion = "1"\ndefault_model = ""\n')
        monkeypatch.setenv("PPMLX_ROUTE_POLICY", str(bad))
        monkeypatch.setattr(server, "_remote_routing_loaded", False)
        monkeypatch.setattr(server, "_remote_routing_service", None)
        monkeypatch.setattr(server, "_remote_policy_failure", None)

        assert server._get_remote_routing_service() is None
        path, reason = server._remote_policy_failure
        assert path == str(bad)
        assert "default model is invalid" in reason

    def test_alias_request_refused_when_policy_broken(self, monkeypatch, tmp_path):
        import ppmlx.server as server

        monkeypatch.setattr(
            server,
            "_remote_policy_failure",
            ("/tmp/routes.toml", "ValueError: Route policy default model is invalid"),
        )
        response = server._remote_route_chat_response(
            None, {"model": "or-model", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 503
        body = response.body.decode()
        assert "route_policy_unavailable" in body
        assert "/tmp/routes.toml" in body


class TestModelErrorSurfacing:
    def test_sanitized_detail_includes_type_and_message(self):
        from ppmlx.server import _sanitized_generation_error

        cause = FileNotFoundError(2, "No such file or directory")
        cause.filename = "model/config.json"
        detail = _sanitized_generation_error(cause)
        assert detail.startswith("FileNotFoundError:")
        assert "config.json" in detail

    def test_sanitized_detail_strips_newlines_and_truncates(self):
        from ppmlx.server import _sanitized_generation_error

        detail = _sanitized_generation_error(RuntimeError("a\nb\n" + "x" * 500))
        assert "\n" not in detail
        assert detail.endswith("...")
        assert len(detail) < 220

    def test_chat_failure_detail_includes_cause(self):
        """The non-streaming chat path embeds the sanitized cause."""
        import inspect

        import ppmlx.server as server

        source = inspect.getsource(server)
        assert 'detail=f"Model generation failed ({_sanitized_generation_error(exc)})"' in source


class TestEndToEndOpenRouter:
    def _service(self, transport):
        from ppmlx.providers.openai import OpenAIProvider

        policy = policy_from_dict(tomllib.loads(POLICY_TOML))
        provider = OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            env_key="OPENROUTER_API_KEY",
            provider_id="openrouter",
            model_catalog=("anthropic/claude-3.5-haiku",),
            transport=transport,
        )
        return RoutingService(policy, {"openrouter": provider})

    def _envelope(self):
        from ppmlx.protocols.base import DecodeContext
        from ppmlx.protocols.openai_chat import OpenAIChatAdapter

        return OpenAIChatAdapter().decode_request(
            {"model": "or-model", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
            context=DecodeContext(request_id="req_test_1", kind="initial"),
        ).request

    def test_alias_reaches_openrouter_base_url(self, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization", "")
            return httpx.Response(
                200,
                json={
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "anthropic/claude-3.5-haiku",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        service = self._service(httpx.MockTransport(handler))
        result = service.execute(_route_input(), self._envelope())
        assert seen["url"].startswith("https://openrouter.ai/api/v1")
        assert seen["auth"] == "Bearer sk-test"
        assert result.provider_id == "openrouter"
        assert any(
            getattr(event, "type", "") == "response.completed"
            for event in result.events
        )

    def test_bad_key_is_typed_provider_auth_failed(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-bad")
        service = self._service(httpx.MockTransport(handler))
        from ppmlx.routing_service import RoutingServiceError

        with pytest.raises(RoutingServiceError) as info:
            service.execute(_route_input(), self._envelope())
        assert info.value.code == "provider_auth_failed"
        assert info.value.status_code == 502
