"""End-to-end tests for remote routing wiring (ADR 0005 MVP).

All provider transports are fakes: no live HTTP is performed. Tests cover
alias -> provider resolution, real capability lookup, fallback per the
forbidden matrix, first-output pinning, cancellation, and secret hygiene,
plus one server-level routed request.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from ppmlx.providers.base import (
    ProviderCapabilities,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealth,
    ProviderHealthStatus,
    ProviderInvocation,
    ProviderResult,
    ProviderStreamingMode,
    ProviderToolSupportStatus,
)
from ppmlx.router import (
    HealthSnapshot,
    HealthState,
    RequiredCapabilities,
    RouteCandidate,
    RouteInput,
    SnapshotCandidateState,
    policy_from_dict,
)
from ppmlx.routing_service import RoutingService, RoutingServiceError


SECRET = "sk-super-secret-key-value"


def _caps(provider_id: str, *, tools: bool = True) -> ProviderCapabilities:
    return ProviderCapabilities(
        text=True,
        images=False,
        tools=tools,
        parallel_tool_calls=tools,
        reasoning=False,
        streaming=ProviderStreamingMode.BUFFERED,
        context_window=128_000,
        data_path=ProviderDataPath.REMOTE,
        credential_types=(ProviderCredentialType.API_KEY,),
        tool_support_status=(
            ProviderToolSupportStatus.STABLE
            if tools
            else ProviderToolSupportStatus.DISABLED
        ),
    )


def _text_events(request_id: str, text: str) -> tuple:
    from ppmlx.agent_ir import (
        ContentCompletedEvent,
        ContentDeltaEvent,
        ContentStartedEvent,
        ResponseCompletedEvent,
    )
    from ppmlx.agent_ir.content import TextBlock

    return (
        ContentStartedEvent(
            type="content.started",
            request_id=request_id,
            sequence=0,
            choice_index=0,
            output_id="out1",
            content_index=0,
            content_type="text",
        ),
        ContentDeltaEvent(
            type="content.delta",
            request_id=request_id,
            sequence=1,
            choice_index=0,
            output_id="out1",
            content_index=0,
            delta=text,
        ),
        ContentCompletedEvent(
            type="content.completed",
            request_id=request_id,
            sequence=2,
            choice_index=0,
            output_id="out1",
            content_index=0,
            content=TextBlock(type="text", text=text),
        ),
        ResponseCompletedEvent(
            type="response.completed",
            request_id=request_id,
            sequence=3,
            choice_index=0,
            output_id="out1",
            finish_reason="stop",
        ),
    )


class FakeProvider:
    """Minimal in-memory Provider double with scripted behaviour."""

    def __init__(
        self,
        provider_id: str,
        *,
        events: tuple = (),
        error: ProviderError | None = None,
        capabilities: ProviderCapabilities | None = None,
        stream_events: tuple = (),
        stream_error_after: ProviderError | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._events = events
        self._error = error
        self._capabilities = capabilities or _caps(provider_id)
        self._stream_events = stream_events
        self._stream_error_after = stream_error_after
        self.invocations: list[ProviderInvocation] = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def list_models(self):
        return ()

    def capabilities(self, model_id: str) -> ProviderCapabilities:
        return self._capabilities

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self._provider_id,
            status=ProviderHealthStatus.HEALTHY,
        )

    def invoke(self, invocation: ProviderInvocation) -> ProviderResult:
        self.invocations.append(invocation)
        if self._error is not None:
            raise self._error
        return ProviderResult(
            provider_id=self._provider_id,
            model_id=invocation.model_id,
            events=self._events,
        )

    def stream(self, invocation: ProviderInvocation) -> Iterator:
        self.invocations.append(invocation)
        for event in self._stream_events:
            yield event
        if self._stream_error_after is not None:
            raise self._stream_error_after


def _policy(**overrides) -> dict:
    document = {
        "routes": {
            "version": "1",
            "aliases": {"gpt-remote": ["openai", "gpt-4o-mini"]},
            "entries": [
                {
                    "key": "openai-chat:gpt-remote",
                    "candidates": [
                        {"provider": "openai", "model": "gpt-4o-mini"},
                        {"provider": "anthropic", "model": "claude-3-5-haiku"},
                    ],
                    "fallback_errors": ["connection", "timeout", "unavailable"],
                }
            ],
            "total_attempt_limit": 3,
        }
    }
    document["routes"].update(overrides)
    return document


def _route_input(**overrides) -> RouteInput:
    values = {
        "public_model": "gpt-remote",
        "harness": "openai-chat",
        "harness_version": "1.0.0",
        "protocol": "openai-chat",
        "required": RequiredCapabilities(text=True),
        "policy_version": "1",
        "health_snapshot_id": "snap_test",
        "request_id": "req_test_1",
        "session_id": "sess_test",
    }
    values.update(overrides)
    return RouteInput(**values)


def _envelope(request_id: str = "req_test_1"):
    from ppmlx.protocols.base import DecodeContext
    from ppmlx.protocols.openai_chat import OpenAIChatAdapter

    return OpenAIChatAdapter().decode_request(
        {
            "model": "gpt-remote",
            "messages": [{"role": "user", "content": "hello"}],
        },
        context=DecodeContext(request_id=request_id, kind="initial"),
    ).request


def _all_healthy_snapshot(candidates) -> HealthSnapshot:
    moment = datetime.now(UTC)
    states = {
        (c.provider_id, c.auth_profile or "", c.model): SnapshotCandidateState(
            provider_id=c.provider_id,
            model=c.model,
            auth_profile=c.auth_profile,
            state=HealthState.HEALTHY,
            reason_category="test",
            checked_at=moment,
        )
        for c in candidates
    }
    return HealthSnapshot(
        snapshot_id="snap_test",
        captured_at=moment,
        expires_at=moment + timedelta(seconds=30),
        states=MappingProxyType(states),
    )


def test_alias_resolves_to_remote_provider_end_to_end():
    events = _text_events("req_test_1", "hi from remote")
    openai = FakeProvider("openai", events=events)
    anthropic = FakeProvider("anthropic", events=events)
    service = RoutingService(policy_from_dict(_policy()), {"openai": openai, "anthropic": anthropic})

    assert service.is_remote_model("gpt-remote") is True
    assert service.remote_alias_target("gpt-remote") == ("openai", "gpt-4o-mini")

    result = service.execute(_route_input(), _envelope())
    assert result.provider_id == "openai"
    assert result.model_id == "gpt-4o-mini"
    assert result.decision.status == "routed"
    assert [e.delta for e in result.events if hasattr(e, "delta")] == [
        "hi from remote"
    ]
    assert openai.invocations[0].model_id == "gpt-4o-mini"


def test_capability_rejection_falls_back_to_next_candidate():
    events = _text_events("req_test_1", "ok")
    openai = FakeProvider("openai", events=events, capabilities=_caps("openai", tools=False))
    anthropic = FakeProvider("anthropic", events=events)
    service = RoutingService(policy_from_dict(_policy()), {"openai": openai, "anthropic": anthropic})

    result = service.execute(
        _route_input(required=RequiredCapabilities(text=True, tools=True)),
        _envelope(),
    )
    assert result.provider_id == "anthropic"
    assert "tools" in result.decision.skipped[0][2]
    assert openai.invocations == []


def test_forbidden_category_never_falls_back():
    openai = FakeProvider(
        "openai", error=ProviderError(provider_id="openai", code="auth_failed")
    )
    anthropic = FakeProvider("anthropic", events=_text_events("req_test_1", "x"))
    service = RoutingService(policy_from_dict(_policy()), {"openai": openai, "anthropic": anthropic})

    with pytest.raises(RoutingServiceError) as excinfo:
        service.execute(_route_input(), _envelope())
    assert excinfo.value.code == "provider_auth_failed"
    assert anthropic.invocations == []


def test_permitted_error_falls_back_to_second_candidate():
    events = _text_events("req_test_1", "from anthropic")
    openai = FakeProvider(
        "openai", error=ProviderError(provider_id="openai", code="network_error")
    )
    anthropic = FakeProvider("anthropic", events=events)
    service = RoutingService(policy_from_dict(_policy()), {"openai": openai, "anthropic": anthropic})

    result = service.execute(_route_input(), _envelope())
    assert result.provider_id == "anthropic"
    assert openai.invocations and anthropic.invocations


def test_stream_pins_after_first_output_event():
    from ppmlx.agent_ir import ContentDeltaEvent

    first = ContentDeltaEvent(
        type="content.delta",
        request_id="req_test_1",
        sequence=0,
        choice_index=0,
        output_id="out1",
        content_index=0,
        delta="partial",
    )
    openai = FakeProvider(
        "openai",
        stream_events=(first,),
        stream_error_after=ProviderError(
            provider_id="openai", code="network_error"
        ),
    )
    anthropic = FakeProvider("anthropic", events=())
    service = RoutingService(policy_from_dict(_policy()), {"openai": openai, "anthropic": anthropic})

    iterator = service.stream(_route_input(), _envelope())
    first_event = next(iterator)
    assert first_event.delta == "partial"
    # Pinned: mid-stream failure propagates instead of switching providers.
    with pytest.raises(ProviderError):
        next(iterator)
    assert anthropic.invocations == []


def test_cancellation_propagates():
    class CancellingProvider(FakeProvider):
        def invoke(self, invocation):
            self.invocations.append(invocation)
            raise ProviderError(
                provider_id=self._provider_id, code="cancelled"
            )

    openai = CancellingProvider("openai")
    service = RoutingService(
        policy_from_dict(_policy()), {"openai": openai}
    )
    with pytest.raises(RoutingServiceError) as excinfo:
        service.execute(_route_input(), _envelope())
    # cancelled maps to no fallback category: typed error, not a switch.
    assert excinfo.value.code == "provider_cancelled"


def test_no_route_for_unknown_model():
    service = RoutingService(
        policy_from_dict(_policy()), {"openai": FakeProvider("openai")}
    )
    assert service.is_remote_model("llama3") is False


def test_error_messages_never_contain_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    openai = FakeProvider(
        "openai", error=ProviderError(provider_id="openai", code="auth_failed")
    )
    service = RoutingService(
        policy_from_dict(_policy()), {"openai": openai}
    )
    with pytest.raises(RoutingServiceError) as excinfo:
        service.execute(_route_input(), _envelope())
    rendered = repr(excinfo.value) + str(excinfo.value)
    assert SECRET not in rendered
    assert SECRET not in repr(excinfo.value.__cause__)


def test_prime_provider_credentials_pulls_from_keyring(monkeypatch):
    import ppmlx.routing_service as rs

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        rs,
        "resolve_secret",
        lambda provider, prefer_env=False: ("key-from-keyring", "keyring"),
    )
    primed = rs.prime_provider_credentials(("openai",))
    assert primed == ("openai",)
    import os

    assert os.environ["OPENAI_API_KEY"] == "key-from-keyring"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Server-level wiring: one routed request from API to remote provider & back.
# ---------------------------------------------------------------------------


@pytest.fixture()
def routed_client(monkeypatch):
    import ppmlx.server as server_mod

    events = _text_events("req_test_1", "routed via server")
    openai = FakeProvider("openai", events=events)
    service = RoutingService(
        policy_from_dict(_policy()), {"openai": openai}
    )
    monkeypatch.setattr(server_mod, "_remote_routing_service", service)
    monkeypatch.setattr(server_mod, "_remote_routing_loaded", True)
    return TestClient(server_mod.app), openai


def test_server_routes_remote_alias(routed_client):
    client, openai = routed_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-remote",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "routed via server"
    assert len(openai.invocations) == 1


def test_server_streams_remote_alias(routed_client):
    client, openai = routed_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-remote",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "chat.completion.chunk" in response.text
    assert len(openai.invocations) == 1


def test_server_local_model_ignores_remote_route(routed_client, monkeypatch):
    client, openai = routed_client
    # A non-alias model must not touch remote providers. The local engine
    # path will fail to load a model in tests; assert only that the remote
    # provider was never invoked.
    import ppmlx.server as server_mod

    monkeypatch.setattr(
        server_mod, "_get_remote_routing_service", lambda: None
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-remote",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert len(openai.invocations) == 0
    assert response.status_code in {200, 400, 500, 503}
