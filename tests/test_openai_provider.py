"""Tests for the remote OpenAI provider over a mocked HTTP transport."""
from __future__ import annotations

import json
import threading

import httpx
import pytest

from ppmlx.agent_ir import (
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    ResponseCompletedEvent,
    ResponseRefusedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from ppmlx.protocols import CallReference, DecodeContext, openai_chat_adapter
from ppmlx.providers import (
    OpenAIProvider,
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

API_KEY = "sk-test-secret-value-123456"


def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)


def _completion(
    *,
    content: str | None = "Hello there",
    tool_calls: list[dict[str, object]] | None = None,
    finish_reason: str | None = None,
    refusal: str | None = None,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if refusal is not None:
        message["refusal"] = refusal
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    document: dict[str, object] = {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
                or ("tool_calls" if tool_calls else "stop"),
            }
        ],
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


def _text_chunk(text: str, *, finish_reason: str | None = None) -> dict[str, object]:
    choice: dict[str, object] = {"index": 0, "delta": {"content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"id": "chatcmpl_sse", "object": "chat.completion.chunk", "choices": [choice]}


_DONE_FRAME = b"data: [DONE]"


def _invocation(
    *,
    tools: bool = False,
    native_messages: list[dict[str, object]] | None = None,
    **kwargs: object,
) -> ProviderInvocation:
    native: dict[str, object] = {
        "model": "gpt-4o",
        "messages": native_messages
        if native_messages is not None
        else [{"role": "user", "content": "You there?"}],
    }
    if tools:
        native["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        native["tool_choice"] = "auto"
    envelope = openai_chat_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_openai", kind="initial"),
    ).request
    return ProviderInvocation(request=envelope, model_id="gpt-4o", **kwargs)


# ----------------------------------------------------------------------
# Protocol conformance and metadata
# ----------------------------------------------------------------------


def test_provider_satisfies_protocol_and_declares_remote_api_key_capabilities() -> None:
    provider = OpenAIProvider()

    assert isinstance(provider, Provider)
    assert provider.provider_id == "openai"
    models = provider.list_models()
    assert [model.model_id for model in models] == ["gpt-4o", "gpt-4o-mini"]
    capabilities = models[0].capabilities
    assert capabilities.data_path is ProviderDataPath.REMOTE
    assert capabilities.credential_types == (ProviderCredentialType.API_KEY,)
    assert capabilities.streaming is ProviderStreamingMode.NATIVE
    assert capabilities.tools is True
    assert capabilities.parallel_tool_calls is True


def test_health_reports_credential_state_without_touching_network() -> None:
    missing = OpenAIProvider(env_key="PPMLX_TEST_UNSET_KEY").health()
    assert (missing.status, missing.code) == (
        ProviderHealthStatus.UNAVAILABLE,
        "credential_missing",
    )


def test_health_is_healthy_with_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key(monkeypatch)
    health = OpenAIProvider().health()
    assert (health.status, health.code) == (
        ProviderHealthStatus.HEALTHY,
        "ready",
    )
    assert health.model_count == 2


def test_constructor_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        OpenAIProvider(base_url="ftp://api.openai.com/v1")
    with pytest.raises(ValueError):
        OpenAIProvider(env_key="OPENAI KEY")
    with pytest.raises(ValueError):
        OpenAIProvider(model_catalog=["gpt-4o", "gpt-4o"])
    with pytest.raises(ValueError):
        OpenAIProvider(timeout_seconds=0)
    with pytest.raises(ValueError):
        OpenAIProvider(call_id_factory="not-callable")


# ----------------------------------------------------------------------
# Request encoding
# ----------------------------------------------------------------------


def test_invoke_encodes_agent_ir_request_to_chat_completions_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    seen_payloads: list[dict[str, object]] = []
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_completion())

    native: dict[str, object] = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    envelope = openai_chat_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_encode", kind="initial"),
    ).request

    result = OpenAIProvider(transport=httpx.MockTransport(handler)).invoke(
        ProviderInvocation(request=envelope, model_id="gpt-4o")
    )

    payload = seen_payloads[0]
    assert payload["model"] == "gpt-4o"
    assert payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
    assert payload["messages"][-1] == {"role": "user", "content": "Hi"}
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 128
    assert seen_headers["authorization"] == f"Bearer {API_KEY}"
    assert isinstance(result.events[-1], ResponseCompletedEvent)


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
                json=_completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_src_1",
                            "type": "function",
                            "index": 0,
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Gdansk"}',
                            },
                        }
                    ],
                ),
            )
        return httpx.Response(200, json=_completion(content="done"))

    provider = OpenAIProvider(transport=httpx.MockTransport(handler))
    first_result = provider.invoke(_invocation(tools=True))

    call = first_result.calls[0]
    source_call_id = first_result.source_call_ids[call.call_id]
    completed = next(
        event
        for event in first_result.events
        if isinstance(event, ToolCallCompletedEvent)
    )
    follow_up: dict[str, object] = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": source_call_id,
                        "type": "function",
                        "function": {
                            "name": completed.name,
                            "arguments": completed.arguments_raw,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": source_call_id,
                "content": "sunny 21C",
            },
        ],
    }
    continuation_envelope = openai_chat_adapter.decode_request(
        follow_up,
        context=DecodeContext(
            request_id="req_openai_2",
            kind="continuation",
            parent_request_id="req_openai",
            prior_calls={
                source_call_id: CallReference(
                    call_id=source_call_id,
                    name="get_weather",
                    choice_index=call.choice_index,
                    output_id=call.output_id,
                    tool_call_index=call.tool_call_index,
                    parallel_group_id=call.parallel_group_id,
                )
            },
            result_output_ids={
                source_call_id: call.output_id,
            },
        ),
    ).request
    second_result = provider.invoke(
        ProviderInvocation(request=continuation_envelope, model_id="gpt-4o")
    )

    assistant_message = payloads[-1]["messages"][-2]
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payloads[-1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": source_call_id,
        "content": "sunny 21C",
    }
    assert isinstance(second_result.events[-1], ResponseCompletedEvent)


# ----------------------------------------------------------------------
# Buffered completion parsing
# ----------------------------------------------------------------------


def test_successful_text_completion_emits_agent_ir_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    transport = httpx.MockTransport(
        _json_handler(
            _completion(
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            )
        )
    )

    result = OpenAIProvider(transport=transport).invoke(_invocation())

    kinds = [(type(event).__name__, event.sequence) for event in result.events]
    assert kinds == [
        ("ContentStartedEvent", 0),
        ("ContentDeltaEvent", 1),
        ("ContentCompletedEvent", 2),
        ("ResponseCompletedEvent", 3),
    ]
    delta = result.events[1]
    assert isinstance(delta, ContentDeltaEvent)
    assert delta.delta == "Hello there"
    terminal = result.events[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.finish_reason == "stop"
    assert terminal.usage is not None
    assert terminal.usage.source.value == "provider"
    assert (terminal.usage.input_tokens, terminal.usage.output_tokens, terminal.usage.total_tokens) == (
        11,
        7,
        18,
    )
    assert result.cancelled is False
    assert result.calls == ()


def test_native_tool_calls_parse_to_agent_ir_events_and_call_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    raw_arguments = '{"city":"Gdansk"}'
    transport = httpx.MockTransport(
        _json_handler(
            _completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_src_1",
                        "type": "function",
                        "index": 0,
                        "function": {"name": "get_weather", "arguments": raw_arguments},
                    }
                ],
            )
        )
    )
    provider = OpenAIProvider(
        transport=transport,
        call_id_factory=lambda: "call_local_1",
    )

    result = provider.invoke(_invocation(tools=True))

    started = next(
        event for event in result.events if isinstance(event, ToolCallStartedEvent)
    )
    delta = next(
        event
        for event in result.events
        if isinstance(event, ToolCallArgumentsDeltaEvent)
    )
    completed = next(
        event for event in result.events if isinstance(event, ToolCallCompletedEvent)
    )
    assert started.name == "get_weather"
    assert started.call_id == "call_local_1"
    assert delta.delta == raw_arguments
    assert completed.arguments_raw == raw_arguments
    assert completed.arguments_json == {"city": "Gdansk"}
    assert result.calls[0].call_id == "call_local_1"
    # Source identifiers are exposed through the sanitized mapping only.
    assert dict(result.source_call_ids) == {"call_local_1": "call_src_1"}
    terminal = result.events[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.finish_reason == "tool_calls"


def test_parallel_tool_calls_share_one_parallel_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    transport = httpx.MockTransport(
        _json_handler(
            _completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_src_b",
                        "type": "function",
                        "index": 1,
                        "function": {"name": "second_tool", "arguments": "{}"},
                    },
                    {
                        "id": "call_src_a",
                        "type": "function",
                        "index": 0,
                        "function": {"name": "first_tool", "arguments": "{}"},
                    },
                ],
            )
        )
    )
    provider = OpenAIProvider(transport=transport)

    result = provider.invoke(_invocation(tools=True))

    started = [
        event for event in result.events if isinstance(event, ToolCallStartedEvent)
    ]
    assert [event.name for event in started] == ["first_tool", "second_tool"]
    groups = {started[0].parallel_group_id, started[1].parallel_group_id}
    assert len(groups) == 1
    assert None not in groups
    assert started[0].parallel_group_id == started[1].parallel_group_id
    assert [reference.tool_call_index for reference in result.calls] == [0, 1]


def test_refusal_completion_maps_to_response_refused_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    transport = httpx.MockTransport(
        _json_handler(_completion(content=None, refusal="Cannot help with that."))
    )

    result = OpenAIProvider(transport=transport).invoke(_invocation())

    assert len(result.events) == 1
    refusal = result.events[0]
    assert isinstance(refusal, ResponseRefusedEvent)
    assert refusal.refusal.text == "Cannot help with that."


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


def test_stream_preserves_sse_delta_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key(monkeypatch)
    body = _sse_frames(
        _data_frame(_text_chunk("Hel")),
        _data_frame(_text_chunk("lo ")),
        _data_frame(_text_chunk("world")),
        _data_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _DONE_FRAME,
    )

    events = tuple(
        OpenAIProvider(transport=httpx.MockTransport(_sse_handler(body))).stream(
            _invocation()
        )
    )

    deltas = [event for event in events if isinstance(event, ContentDeltaEvent)]
    assert [event.delta for event in deltas] == ["Hel", "lo ", "world"]
    assert isinstance(events[0], ContentStartedEvent)
    completed = events[-2]
    assert isinstance(completed, ContentCompletedEvent)
    assert completed.content.text == "Hello world"
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert events[-1].finish_reason == "stop"
    sequences = [event.sequence for event in events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


def test_streamed_tool_call_fragments_assemble_to_completed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    body = _sse_frames(
        _data_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream_src",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        _data_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city"'}}
                            ]
                        },
                    }
                ]
            }
        ),
        _data_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ':"Paris"}'}}
                            ]
                        },
                    }
                ]
            }
        ),
        _data_frame(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        ),
        _DONE_FRAME,
    )

    provider = OpenAIProvider(transport=httpx.MockTransport(_sse_handler(body)))
    events = tuple(provider.stream(_invocation(tools=True)))

    deltas = [
        event
        for event in events
        if isinstance(event, ToolCallArgumentsDeltaEvent)
    ]
    assert [event.delta for event in deltas] == ['{"city"', ':"Paris"}']
    completed = next(
        event for event in events if isinstance(event, ToolCallCompletedEvent)
    )
    assert completed.arguments_raw == '{"city":"Paris"}'
    assert completed.arguments_json == {"city": "Paris"}
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert events[-1].finish_reason == "tool_calls"


def test_stream_usage_only_terminal_chunk_yields_completed_event_without_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    body = _sse_frames(
        _data_frame({"choices": []}),
        _data_frame(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 0,
                    "total_tokens": 3,
                },
            }
        ),
        _data_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _DONE_FRAME,
    )

    events = tuple(
        OpenAIProvider(transport=httpx.MockTransport(_sse_handler(body))).stream(
            _invocation()
        )
    )

    assert not any(isinstance(event, ContentStartedEvent) for event in events)
    terminal = events[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.usage is not None
    assert terminal.usage.total_tokens == 3


def test_empty_stream_body_is_malformed_not_silent_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    with pytest.raises(ProviderError) as error:
        tuple(
            OpenAIProvider(transport=httpx.MockTransport(_sse_handler(b""))).stream(
                _invocation()
            )
        )
    assert error.value.code == "invalid_response"


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------


def test_cancelled_before_start_produces_no_events_and_typed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    requested = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        requested.set()
        return httpx.Response(200, json=_completion(content="never"))

    transport = httpx.MockTransport(handler)

    buffered_handle = ProviderCancellationHandle()
    buffered_handle.cancel()
    buffered_result = OpenAIProvider(transport=transport).invoke(
        _invocation(cancel_handle=buffered_handle)
    )
    assert buffered_result.cancelled is True
    assert buffered_result.events == ()

    streaming_handle = ProviderCancellationHandle()
    streaming_handle.cancel()
    with pytest.raises(ProviderCancelledError) as cancelled:
        tuple(
            OpenAIProvider(transport=httpx.MockTransport(handler)).stream(
                _invocation(cancel_handle=streaming_handle)
            )
        )
    assert cancelled.value.code == "cancelled"
    assert not requested.is_set()


def test_cancel_mid_stream_raises_typed_error_after_yielded_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)
    body = _sse_frames(
        _data_frame(_text_chunk("first")),
        _data_frame(_text_chunk("second")),
        _DONE_FRAME,
    )
    handle = ProviderCancellationHandle()

    # Generators are lazy: the first content deltas are produced while later
    # SSE frames are still unconsumed, so cancelling here hits a real chunk
    # boundary.
    generator = OpenAIProvider(
        transport=httpx.MockTransport(_sse_handler(body))
    ).stream(_invocation(cancel_handle=handle))
    events: list[object] = []
    for event in generator:
        events.append(event)
        if isinstance(event, ContentDeltaEvent):
            break

    handle.cancel()
    with pytest.raises(ProviderCancelledError) as cancelled:
        for event in generator:
            events.append(event)

    assert cancelled.value.code == "cancelled"
    # Pre-cancellation events stay valid; nothing is silently truncated into
    # a fake successful completion.
    assert events[-1].delta == "first"
    assert not any(isinstance(event, ResponseCompletedEvent) for event in events)


# ----------------------------------------------------------------------
# Typed errors
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
def test_http_failures_map_to_typed_safe_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    _api_key(monkeypatch)
    error_body = json.dumps(
        {
            "error": {
                "message": f"Incorrect API key provided: {API_KEY}",
                "type": "invalid_request_error",
            }
        }
    ).encode("utf-8")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, content=error_body)
    )

    with pytest.raises(ProviderError) as invoke_error:
        OpenAIProvider(transport=transport).invoke(_invocation())
    with pytest.raises(ProviderError) as stream_error:
        tuple(OpenAIProvider(transport=transport).stream(_invocation()))

    assert invoke_error.value.code == code
    assert stream_error.value.code == code
    assert API_KEY not in str(invoke_error.value)
    assert "Incorrect API key" not in str(invoke_error.value)
    assert "Incorrect API key" not in str(stream_error.value)


def test_transport_failures_map_to_network_and_timeout_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _api_key(monkeypatch)

    def network_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to 10.0.0.1")

    with pytest.raises(ProviderError) as network_error:
        OpenAIProvider(transport=httpx.MockTransport(network_handler)).invoke(
            _invocation()
        )
    assert network_error.value.code == "network_error"

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ProviderError) as timeout_error:
        OpenAIProvider(transport=httpx.MockTransport(timeout_handler)).invoke(
            _invocation()
        )
    assert timeout_error.value.code == "timeout"
    assert "10.0.0.1" not in str(network_error.value)

    with pytest.raises(ProviderError) as stream_network_error:
        tuple(
            OpenAIProvider(transport=httpx.MockTransport(network_handler)).stream(
                _invocation()
            )
        )
    assert stream_network_error.value.code == "network_error"


# ----------------------------------------------------------------------
# Malformed responses
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        b'{"object": "chat.completion", "choices": []}',
        b'{"choices": [{"index": 0, "message": null}]}',
        b'{"choices": [{"index": 0, "message": {"role": "assistant", "content": 42}}]}',
        b'{"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": 7}]}',
        b'{"choices": [{"index": 0, "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]}',
    ],
)
def test_malformed_buffered_responses_are_rejected_with_typed_error(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    _api_key(monkeypatch)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))

    with pytest.raises(ProviderError) as error:
        OpenAIProvider(transport=transport).invoke(_invocation())
    assert error.value.code == "invalid_response"


@pytest.mark.parametrize(
    "frame",
    [
        b"data: not-json",
        b"data: {}",
        b'data: {"choices": [{"index": 0, "delta": "text"}]}',
        b'data: {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}]}',
        b'data: {"choices": [{"index": 5, "delta": {"content": "hi"}}]}',
    ],
)
def test_malformed_stream_chunks_are_rejected_with_typed_error(
    monkeypatch: pytest.MonkeyPatch, frame: bytes
) -> None:
    _api_key(monkeypatch)
    body = _sse_frames(frame, _DONE_FRAME)

    with pytest.raises(ProviderError) as error:
        tuple(
            OpenAIProvider(transport=httpx.MockTransport(_sse_handler(body))).stream(
                _invocation()
            )
        )
    assert error.value.code == "invalid_response"


# ----------------------------------------------------------------------
# Secret hygiene
# ----------------------------------------------------------------------


def test_missing_credentials_raise_typed_error_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent without credentials")

    with pytest.raises(ProviderError) as error:
        OpenAIProvider(transport=httpx.MockTransport(handler)).invoke(_invocation())
    assert error.value.code == "credential_missing"


def test_api_key_never_appears_in_any_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key(monkeypatch)
    leak_probe = f"Bearer {API_KEY}"

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(leak_probe)

    scenarios: list[ProviderError] = []

    def collect(call) -> None:
        try:
            call()
        except ProviderError as failure:
            scenarios.append(failure)

    collect(
        lambda: OpenAIProvider(transport=httpx.MockTransport(failing_handler)).invoke(
            _invocation()
        )
    )
    collect(
        lambda: tuple(
            OpenAIProvider(
                transport=httpx.MockTransport(failing_handler)
            ).stream(_invocation())
        )
    )
    collect(
        lambda: OpenAIProvider(
            base_url="http://127.0.0.1:1", timeout_seconds=0.001
        ).invoke(_invocation())
    )

    assert len(scenarios) >= 3
    for failure in scenarios:
        assert API_KEY not in str(failure)
        assert leak_probe not in repr(failure)
