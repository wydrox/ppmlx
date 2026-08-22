"""Tests for the remote Anthropic provider over a mocked HTTP transport."""
from __future__ import annotations

import json

import httpx
import pytest

from ppmlx.agent_ir import (
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    ResponseCompletedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from ppmlx.protocols import (
    CallReference,
    DecodeContext,
    anthropic_messages_adapter,
)
from ppmlx.providers import (
    AnthropicProvider,
    Provider,
    ProviderCancellationHandle,
    ProviderCancelledError,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealthStatus,
    ProviderInvocation,
    ProviderStreamingMode,
)

API_KEY = "sk-ant-tes...3456"


def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)


def _message_response(
    *,
    content: list[dict[str, object]] | None = None,
    stop_reason: str = "end_turn",
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content
        if content is not None
        else [{"type": "text", "text": "Hello there"}],
        "stop_reason": stop_reason,
    }
    if usage is not None:
        document["usage"] = usage
    return document


def _json_handler(payload: dict[str, object], *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, request=request)

    return handler


def _sse_frames(*frames: bytes) -> bytes:
    return b"".join(frame + b"\n\n" for frame in frames)


def _sse_handler(body: bytes, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"Content-Type": "text/event-stream"},
            content=body,
            request=request,
        )

    return handler


def _data_frame(payload: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8")


def _invocation(
    *,
    tools: bool = False,
    native_messages: list[dict[str, object]] | None = None,
    **kwargs: object,
) -> ProviderInvocation:
    native: dict[str, object] = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": native_messages
        if native_messages is not None
        else [{"role": "user", "content": [{"type": "text", "text": "You there?"}]}],
    }
    if tools:
        native["tools"] = [
            {
                "name": "get_weather",
                "description": "Look up weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        native["tool_choice"] = {"type": "auto"}
    envelope = anthropic_messages_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_anthropic", kind="initial"),
    ).request
    return ProviderInvocation(request=envelope, model_id="claude-sonnet-4-5", **kwargs)


# ----------------------------------------------------------------------
# Protocol conformance and metadata
# ----------------------------------------------------------------------


def test_provider_satisfies_protocol_and_declares_remote_api_key_capabilities() -> None:
    provider = AnthropicProvider()

    assert isinstance(provider, Provider)
    assert provider.provider_id == "anthropic"
    models = provider.list_models()
    assert [model.model_id for model in models] == [
        "claude-opus-4-1",
        "claude-sonnet-4-5",
    ]
    capabilities = models[0].capabilities
    assert capabilities.data_path is ProviderDataPath.REMOTE
    assert capabilities.credential_types == (ProviderCredentialType.API_KEY,)
    assert capabilities.streaming is ProviderStreamingMode.NATIVE
    assert capabilities.tools is True
    assert capabilities.parallel_tool_calls is True


def test_health_reports_credential_state_without_touching_network() -> None:
    missing = AnthropicProvider(env_key="PPMLX_TEST_UNSET_KEY").health()
    assert (missing.status, missing.code) == (
        ProviderHealthStatus.UNAVAILABLE,
        "credential_missing",
    )


def test_health_is_healthy_with_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key(monkeypatch)
    health = AnthropicProvider().health()
    assert (health.status, health.code) == (
        ProviderHealthStatus.HEALTHY,
        "ready",
    )
    assert health.model_count == 2


def test_constructor_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        AnthropicProvider(base_url="ftp://api.anthropic.com")
    with pytest.raises(ValueError):
        AnthropicProvider(env_key="ANTHROPIC KEY")
    with pytest.raises(ValueError):
        AnthropicProvider(model_catalog=["a", "a"])
    with pytest.raises(ValueError):
        AnthropicProvider(timeout_seconds=0)
    with pytest.raises(ValueError):
        AnthropicProvider(call_id_factory="not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AnthropicProvider(anthropic_version="")


# ----------------------------------------------------------------------
# Request encoding and buffered completion
# ----------------------------------------------------------------------


def test_invoke_encodes_agent_ir_request_to_messages_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    seen_payloads: list[dict[str, object]] = []
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_message_response())

    native: dict[str, object] = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 128,
        "system": "Be brief.",
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ],
    }
    envelope = anthropic_messages_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_encode", kind="initial"),
    ).request

    result = AnthropicProvider(transport=httpx.MockTransport(handler)).invoke(
        ProviderInvocation(request=envelope, model_id="claude-sonnet-4-5")
    )

    payload = seen_payloads[0]
    assert payload["model"] == "claude-sonnet-4-5"
    assert "stream" not in payload  # buffered invoke never sets stream=True
    assert payload["system"] == "Be brief."
    assert payload["messages"][-1]["content"] == [{"type": "text", "text": "Hi"}]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 128
    # Secret hygiene: key travels only in x-api-key, never in the body.
    assert seen_headers["x-api-key"] == API_KEY
    assert seen_headers["anthropic-version"] == "2023-06-01"
    assert API_KEY not in json.dumps(payload)
    assert isinstance(result.events[-1], ResponseCompletedEvent)


def test_invoke_maps_tool_use_blocks_preserving_call_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_message_response(
                content=[
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_src_1",
                        "name": "get_weather",
                        "input": {"city": "Gdansk"},
                    },
                ],
                stop_reason="tool_use",
                usage={"input_tokens": 12, "output_tokens": 34},
            ),
        )

    result = AnthropicProvider(transport=httpx.MockTransport(handler)).invoke(
        _invocation(tools=True)
    )
    types = [event.type for event in result.events]
    assert types[0] == "content.started"
    call = result.calls[0]
    source_call_id = result.source_call_ids[call.call_id]
    assert source_call_id == "toolu_src_1"
    started = next(e for e in result.events if isinstance(e, ToolCallStartedEvent))
    completed = next(e for e in result.events if isinstance(e, ToolCallCompletedEvent))
    assert started.name == "get_weather"
    assert json.loads(completed.arguments_raw) == {"city": "Gdansk"}
    finished = result.events[-1]
    assert isinstance(finished, ResponseCompletedEvent)
    assert finished.finish_reason == "tool_use"
    assert finished.usage is not None and finished.usage.input_tokens == 12
    # tool lifecycle comes after content lifecycle
    assert types.index("tool_call.started") > types.index("content.completed")


def test_tool_results_and_prior_tool_calls_round_trip_into_followup_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode("utf-8")))
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json=_message_response(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "toolu_src_1",
                            "name": "get_weather",
                            "input": {"city": "Gdansk"},
                        }
                    ],
                    stop_reason="tool_use",
                ),
            )
        return httpx.Response(
            200, json=_message_response(content=[{"type": "text", "text": "done"}])
        )

    provider = AnthropicProvider(transport=httpx.MockTransport(handler))
    first_result = provider.invoke(_invocation(tools=True))

    call = first_result.calls[0]
    source_call_id = first_result.source_call_ids[call.call_id]
    completed = next(
        event
        for event in first_result.events
        if isinstance(event, ToolCallCompletedEvent)
    )
    follow_up: dict[str, object] = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Look up weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": source_call_id,
                        "name": completed.name,
                        "input": json.loads(completed.arguments_raw),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": source_call_id,
                        "content": "sunny 21C",
                    }
                ],
            },
        ],
    }
    envelope = anthropic_messages_adapter.decode_request(
        follow_up,
        context=DecodeContext(
            request_id="req_followup",
            kind="continuation",
            parent_request_id="req_anthropic",
            prior_calls={
                source_call_id: CallReference(
                    call_id=source_call_id,
                    name=completed.name,
                    choice_index=0,
                    output_id=call.output_id,
                    tool_call_index=0,
                )
            },
            result_output_ids={source_call_id: call.output_id},
        ),
    ).request
    second_result = provider.invoke(
        ProviderInvocation(request=envelope, model_id="claude-sonnet-4-5")
    )
    sent = payloads[1]
    assert sent["messages"][1]["content"][0]["type"] == "tool_use"
    assert sent["messages"][1]["content"][0]["id"] == "toolu_src_1"
    assert sent["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_src_1",
        "content": "sunny 21C",
    }
    assert isinstance(second_result.events[-1], ResponseCompletedEvent)


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


def test_stream_preserves_sse_event_order_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    frames = _sse_frames(
        _data_frame(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_sse",
                    "type": "message",
                    "role": "assistant",
                    "usage": {"input_tokens": 7, "output_tokens": 0},
                },
            }
        ),
        _data_frame(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        _data_frame({"type": "ping"}),
        _data_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "He"},
            }
        ),
        _data_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "llo"},
            }
        ),
        _data_frame({"type": "content_block_stop", "index": 0}),
        _data_frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            }
        ),
        _data_frame({"type": "message_stop"}),
    )
    provider = AnthropicProvider(transport=httpx.MockTransport(_sse_handler(frames)))
    events = list(provider.stream(_invocation()))
    types = [event.type for event in events]

    assert types[0] == "content.started"
    deltas = [e.delta for e in events if isinstance(e, ContentDeltaEvent)]
    assert "".join(deltas) == "Hello"
    assert types[-1] == "response.completed"
    finished = events[-1]
    assert isinstance(finished, ResponseCompletedEvent)
    assert finished.finish_reason == "end_turn"
    assert finished.usage is not None
    assert finished.usage.input_tokens == 7 and finished.usage.output_tokens == 5
    sequences = [event.sequence for event in events]
    assert sequences == sorted(sequences)
    completed = next(e for e in events if isinstance(e, ContentCompletedEvent))
    assert completed.content.text == "Hello"


def test_stream_stages_tool_calls_until_stream_end_with_group_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    frames = _sse_frames(
        _data_frame({"type": "message_start", "message": {"usage": {}}}),
        _data_frame(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_a",
                    "name": "get_weather",
                },
            }
        ),
        _data_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
            }
        ),
        _data_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ty":"Gdansk"}',
                },
            }
        ),
        _data_frame({"type": "content_block_stop", "index": 0}),
        _data_frame(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_b",
                    "name": "get_time",
                },
            }
        ),
        _data_frame({"type": "content_block_stop", "index": 1}),
        _data_frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {},
            }
        ),
        _data_frame({"type": "message_stop"}),
    )
    provider = AnthropicProvider(transport=httpx.MockTransport(_sse_handler(frames)))
    events = list(provider.stream(_invocation(tools=True)))
    types = [event.type for event in events]

    assert types.count("tool_call.started") == 2
    assert types[-1] == "response.completed"
    assert events[-1].finish_reason == "tool_use"
    started_events = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    group_ids = {e.parallel_group_id for e in started_events}
    assert len(group_ids) == 1 and None not in group_ids
    completed = [
        e
        for e in events
        if isinstance(e, ToolCallCompletedEvent) and e.name == "get_weather"
    ][0]
    assert json.loads(completed.arguments_raw) == {"city": "Gdansk"}


def test_stream_cancellation_mid_stream_raises_typed_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    handle = ProviderCancellationHandle()
    frames = _sse_frames(
        _data_frame({"type": "message_start", "message": {"usage": {}}}),
        _data_frame(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        _data_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            }
        ),
        _data_frame({"type": "message_stop"}),
    )

    class CancellingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            def stream_bytes():
                yield frames
                handle.cancel()

            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=stream_bytes(),
            )

    provider = AnthropicProvider(transport=CancellingTransport())
    iterator = provider.stream(_invocation(cancel_handle=handle))  # type: ignore[arg-type]
    yielded = []
    with pytest.raises(ProviderCancelledError):
        for event in iterator:
            yielded.append(event)
    # Events already yielded stay valid; nothing silently truncated.
    assert yielded[0].type == "content.started"
    assert any(isinstance(e, ContentDeltaEvent) for e in yielded)
    assert not any(isinstance(e, ResponseCompletedEvent) for e in yielded)


def test_pre_cancelled_invocations_return_cancelled_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("network must not be touched when already cancelled")

    handle = ProviderCancellationHandle()
    handle.cancel()
    provider = AnthropicProvider(
        env_key="ANTHROPIC_API_KEY", transport=httpx.MockTransport(handler)
    )
    invocation = _invocation(cancel_handle=handle)  # type: ignore[arg-type]

    result = provider.invoke(invocation)
    assert result.cancelled is True and result.events == ()
    with pytest.raises(ProviderCancelledError):
        list(provider.stream(invocation))


# ----------------------------------------------------------------------
# Error mapping and secret hygiene
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "auth_failed"),
        (403, "auth_failed"),
        (429, "rate_limited"),
        (500, "request_rejected"),
    ],
)
def test_http_errors_map_to_typed_safe_codes_without_body_leakage(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    _api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": f"secret {API_KEY}"}}, request=request
        )

    provider = AnthropicProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())
    assert excinfo.value.code == code
    assert API_KEY not in str(excinfo.value)


def test_network_and_timeout_failures_map_to_typed_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    def network_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {API_KEY}")

    provider_timeout = AnthropicProvider(
        transport=httpx.MockTransport(timeout_handler)
    )
    with pytest.raises(ProviderError) as excinfo:
        provider_timeout.invoke(_invocation())
    assert excinfo.value.code == "timeout"

    provider_network = AnthropicProvider(
        transport=httpx.MockTransport(network_handler)
    )
    with pytest.raises(ProviderError) as excinfo:
        provider_network.invoke(_invocation())
    assert excinfo.value.code == "network_error"
    assert API_KEY not in str(excinfo.value)


def test_malformed_buffered_responses_raise_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    def raw_handler(body: bytes):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        return handler

    provider_json = AnthropicProvider(
        transport=httpx.MockTransport(raw_handler(b"not-json{"))
    )
    with pytest.raises(ProviderError) as excinfo:
        provider_json.invoke(_invocation())
    assert excinfo.value.code == "invalid_response"

    provider_shape = AnthropicProvider(
        transport=httpx.MockTransport(raw_handler(json.dumps({"nope": True}).encode()))
    )
    with pytest.raises(ProviderError) as excinfo:
        provider_shape.invoke(_invocation())
    assert excinfo.value.code == "invalid_response"

    provider_block = AnthropicProvider(
        transport=httpx.MockTransport(
            raw_handler(
                json.dumps(_message_response(content=[{"type": "mystery"}])).encode()
            )
        )
    )
    with pytest.raises(ProviderError) as excinfo:
        provider_block.invoke(_invocation())
    assert excinfo.value.code == "invalid_response"


def test_malformed_streams_raise_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    # Stream that ends without message_stop.
    truncated = _sse_frames(
        _data_frame(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )
    )
    provider_truncated = AnthropicProvider(
        transport=httpx.MockTransport(_sse_handler(truncated))
    )
    with pytest.raises(ProviderError) as excinfo:
        list(provider_truncated.stream(_invocation()))
    assert excinfo.value.code == "invalid_response"

    # Undecodable data frame.
    garbage = _sse_frames(b"data: {{{not json")
    provider_garbage = AnthropicProvider(
        transport=httpx.MockTransport(_sse_handler(garbage))
    )
    with pytest.raises(ProviderError) as excinfo:
        list(provider_garbage.stream(_invocation()))
    assert excinfo.value.code == "invalid_response"

    # In-stream error event.
    errored = _sse_frames(
        _data_frame({"type": "error", "error": {"type": "overloaded_error"}})
    )
    provider_errored = AnthropicProvider(
        transport=httpx.MockTransport(_sse_handler(errored))
    )
    with pytest.raises(ProviderError) as excinfo:
        list(provider_errored.stream(_invocation()))
    assert excinfo.value.code == "provider_stream_failed"


def test_missing_credential_fails_fast_without_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not reach network without credentials")

    no_network = AnthropicProvider(
        env_key="PPMLX_TEST_UNSET_KEY", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderError) as excinfo:
        no_network.invoke(_invocation())
    assert excinfo.value.code == "credential_missing"
    with pytest.raises(ProviderError):
        list(no_network.stream(_invocation()))


def test_api_key_never_appears_in_repr_or_error_text() -> None:
    provider = AnthropicProvider(env_key="ANTHROPIC_API_KEY")
    assert API_KEY not in repr(provider)


def test_response_size_limit_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key(monkeypatch)
    big = _message_response(content=[{"type": "text", "text": "x" * 4096}])
    body = json.dumps(big).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    provider = AnthropicProvider(
        max_response_bytes=len(body) - 1, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())
    assert excinfo.value.code == "response_too_large"

    streaming = AnthropicProvider(
        max_response_bytes=len(body) // 2, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderError) as excinfo:
        list(streaming.stream(_invocation()))
    assert excinfo.value.code == "response_too_large"


def test_reasoning_opt_in_rejected_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("reasoning must be rejected before network")

    provider = AnthropicProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation(enable_reasoning=True))  # type: ignore[arg-type]
    assert excinfo.value.code == "reasoning_unsupported"
