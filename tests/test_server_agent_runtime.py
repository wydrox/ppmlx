from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request as StarletteRequest
from starlette.types import Message, Receive, Scope, Send

from ppmlx import server
from ppmlx import engine as engine_module
from ppmlx.local_runtime import runtime as runtime_module
from ppmlx.local_runtime.runtime import RuntimeResponse
from ppmlx.protocols.sse import parse_sse
from tests.test_local_runtime import (
    CASES,
    RuntimeCase,
    _continuation,
    _frames,
    _json,
    _mapping,
    _runtime,
    _sequence,
)


ENDPOINTS = {
    "openai-chat": "/v1/chat/completions",
    "anthropic-messages": "/v1/messages",
    "openai-responses": "/v1/responses",
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(server.app) as test_client:
        yield test_client


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    case: RuntimeCase,
):
    runtime, generator = _runtime(case)
    monkeypatch.setattr(server, "_get_agent_runtime_config", lambda: ("agent_ir", 600))
    monkeypatch.setattr(
        runtime_module,
        "get_local_agent_runtime",
        lambda *, continuation_ttl_seconds, max_tokens_cap: runtime,
    )
    return runtime, generator


def _runtime_response(case: RuntimeCase, sse: str) -> RuntimeResponse:
    frames = parse_sse(sse, protocol=case.protocol)
    first = _mapping(frames[0].data)
    if case.protocol == "openai-chat":
        native_response_id = first["id"]
    elif case.protocol == "anthropic-messages":
        native_response_id = _mapping(first["message"])["id"]
    else:
        native_response_id = _mapping(first["response"])["id"]
    assert isinstance(native_response_id, str)
    return RuntimeResponse(
        protocol=case.protocol,
        conversation_id="conv_http",
        request_id="req_http",
        native_response_id=native_response_id,
        sse=sse,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_http_two_turn_tool_stream_and_retry(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case: RuntimeCase,
) -> None:
    _, generator = _install_runtime(monkeypatch, case)
    endpoint = ENDPOINTS[case.protocol]
    headers = {"x-ppmlx-project": "project-a"}

    first = client.post(endpoint, json=_json(case, "initial-request.json"), headers=headers)

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")
    first_runtime_response = _runtime_response(case, first.text)
    assert _sequence(case, _frames(first_runtime_response)) == case.initial_lifecycle

    continuation = _continuation(case, first_runtime_response)
    second = client.post(endpoint, json=continuation, headers=headers)

    assert second.status_code == 200
    assert second.headers["content-type"].startswith("text/event-stream")
    second_runtime_response = _runtime_response(case, second.text)
    assert _sequence(case, _frames(second_runtime_response)) == case.final_lifecycle
    assert len(generator.requests) == 2

    retry = client.post(endpoint, json=continuation, headers=headers)

    assert retry.status_code == 200
    assert retry.text == second.text
    assert len(generator.requests) == 2


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_http_rejects_cross_project_tool_result_before_sse(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case: RuntimeCase,
) -> None:
    _, generator = _install_runtime(monkeypatch, case)
    endpoint = ENDPOINTS[case.protocol]
    first = client.post(
        endpoint,
        json=_json(case, "initial-request.json"),
        headers={"x-ppmlx-project": "project-a"},
    )
    continuation = _continuation(case, _runtime_response(case, first.text))

    rejected = client.post(
        endpoint,
        json=continuation,
        headers={"x-ppmlx-project": "project-b"},
    )

    assert rejected.status_code == 400
    assert rejected.headers["content-type"].startswith("application/json")
    assert rejected.json() == {"error": {"code": "tool_continuation_expired"}}
    assert len(generator.requests) == 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
@pytest.mark.parametrize("invalid_field", ("stream", "tools"))
def test_strict_tool_transcript_rejects_nonstream_or_missing_tools_before_legacy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case: RuntimeCase,
    invalid_field: str,
) -> None:
    _, generator = _install_runtime(monkeypatch, case)
    legacy_engine = MagicMock(side_effect=AssertionError("legacy engine called"))
    monkeypatch.setattr(engine_module, "get_engine", legacy_engine)
    body = _json(case, "tool-result-request.json")
    body[invalid_field] = False if invalid_field == "stream" else []

    response = client.post(ENDPOINTS[case.protocol], json=body)

    assert response.status_code == 400
    assert response.json() == {
        "error": {"code": "agent_runtime_requires_streamed_tools"}
    }
    assert not legacy_engine.called
    assert generator.requests == []


def test_strict_responses_websocket_rejects_tools_before_legacy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_get_agent_runtime_config", lambda: ("agent_ir", 600))
    legacy_engine = MagicMock(side_effect=AssertionError("legacy engine called"))
    monkeypatch.setattr(engine_module, "get_engine", legacy_engine)

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {
                "type": "response.create",
                "response": _json(CASES[2], "initial-request.json"),
            }
        )
        event = websocket.receive_json()

    assert event == {
        "type": "error",
        "error": {
            "type": "invalid_request",
            "code": "agent_ir_websocket_tools_unsupported",
        },
    }
    assert not legacy_engine.called


def test_invalid_runtime_mode_rejects_websocket_tools_before_legacy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_get_agent_runtime_config", lambda: ("invalid", 600))
    legacy_engine = MagicMock(side_effect=AssertionError("legacy engine called"))
    monkeypatch.setattr(engine_module, "get_engine", legacy_engine)

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {
                "type": "response.create",
                "response": _json(CASES[2], "initial-request.json"),
            }
        )
        event = websocket.receive_json()

    assert event == {
        "type": "error",
        "error": {
            "type": "invalid_request",
            "code": "agent_runtime_configuration_invalid",
        },
    }
    assert not legacy_engine.called


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_tool_choice_without_tools_is_strict_tool_traffic(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case: RuntimeCase,
) -> None:
    _, generator = _install_runtime(monkeypatch, case)
    legacy_engine = MagicMock(side_effect=AssertionError("legacy engine called"))
    monkeypatch.setattr(engine_module, "get_engine", legacy_engine)
    body = _json(case, "initial-request.json")
    body["tools"] = []
    body["tool_choice"] = "required"

    response = client.post(ENDPOINTS[case.protocol], json=body)

    assert response.status_code == 400
    assert response.json() == {
        "error": {"code": "agent_runtime_requires_streamed_tools"}
    }
    assert not legacy_engine.called
    assert generator.requests == []


def test_bearer_credential_is_not_used_as_a_stored_scope_identifier() -> None:
    def request(token: str) -> StarletteRequest:
        scope = cast(
            Scope,
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"authorization", f"Bearer {token}".encode())],
                "client": ("127.0.0.1", 1234),
                "server": ("127.0.0.1", 6767),
            },
        )
        return StarletteRequest(scope)

    first = server._agent_runtime_scope(request("secret-a"), protocol="openai-chat")
    second = server._agent_runtime_scope(request("secret-b"), protocol="openai-chat")

    assert first.principal_id == second.principal_id == "local-listener"
    assert "secret" not in repr(first)


def test_request_size_limit_rejects_chunked_body_without_content_length() -> None:
    sent: list[Message] = []
    downstream_called = False

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal downstream_called
        downstream_called = True
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def exercise() -> None:
        chunks: Iterator[Message] = iter(
            (
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"456", "more_body": False},
            )
        )

        async def receive() -> Message:
            return next(chunks)

        async def send(message: Message) -> None:
            sent.append(message)

        scope = cast(
            Scope,
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/responses",
                "raw_path": b"/v1/responses",
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "client": ("127.0.0.1", 1234),
                "server": ("127.0.0.1", 6767),
            },
        )
        middleware = server.RequestSizeLimitMiddleware(downstream, max_bytes=5)
        await middleware(scope, receive, send)

    asyncio.run(exercise())

    assert downstream_called is True
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


def test_http_passes_configured_max_tokens_cap_to_runtime_getter(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CASES[0]
    runtime, _ = _runtime(case)
    getter = MagicMock(return_value=runtime)
    monkeypatch.setattr(server, "_get_agent_runtime_config", lambda: ("agent_ir", 321))
    monkeypatch.setattr(server, "_get_max_tokens_cap", lambda: 777)
    monkeypatch.setattr(runtime_module, "get_local_agent_runtime", getter)

    response = client.post(
        ENDPOINTS[case.protocol],
        json=_json(case, "initial-request.json"),
    )

    assert response.status_code == 200
    getter.assert_called_once_with(
        continuation_ttl_seconds=321,
        max_tokens_cap=777,
    )


@pytest.mark.parametrize(
    "header_name",
    ("x-ppmlx-project", "x-ppmlx-harness-id", "authorization"),
)
def test_http_rejects_overlong_scope_headers_before_generation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
) -> None:
    case = CASES[0]
    _, generator = _install_runtime(monkeypatch, case)

    response = client.post(
        ENDPOINTS[case.protocol],
        json=_json(case, "initial-request.json"),
        headers={header_name: "x" * 1025},
    )

    assert response.status_code == 400
    assert response.json() == {"error": {"code": "invalid_runtime_scope"}}
    assert generator.requests == []


def test_non_loopback_strict_request_fails_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    getter = MagicMock(side_effect=AssertionError("runtime getter called"))
    monkeypatch.setattr(server, "_get_agent_runtime_config", lambda: ("agent_ir", 600))
    monkeypatch.setattr(runtime_module, "get_local_agent_runtime", getter)
    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("203.0.113.10", 1234),
            "server": ("127.0.0.1", 6767),
        },
    )
    request = StarletteRequest(scope)

    response = asyncio.run(
        server._strict_agent_runtime_response(
            request,
            _json(CASES[0], "initial-request.json"),
            protocol="openai-chat",
        )
    )

    assert response is not None
    assert response.status_code == 403
    assert response.body == b'{"error":{"code":"agent_runtime_loopback_required"}}'
    assert not getter.called
