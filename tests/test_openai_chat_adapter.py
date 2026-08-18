from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from ppmlx.agent_ir import Origin, Provenance, Sensitivity, ToolCallBlock, Trust, load_agent_ir
from ppmlx.protocols.base import (
    AdapterLimits,
    CallReference,
    DecodeContext,
    EncodeContext,
    NormalizationPolicy,
    ProtocolAdapterError,
)
from ppmlx.protocols.openai_chat import OpenAIChatAdapter


CONTRACTS = Path(__file__).parent / "fixtures" / "contracts" / "openai-chat"
FIXTURES = ("opencode-1.18.18", "pi-0.84.2")
PUBLIC_HARNESS_POLICY = NormalizationPolicy(
    sensitivity=Sensitivity.PUBLIC,
    provenance=Provenance(origin=Origin.HARNESS, trust=Trust.UNTRUSTED),
    include_native_evidence=True,
)
CALL = CallReference(
    call_id="call_capture_001",
    name="bash",
    choice_index=0,
    output_id="chatcmpl-capture-tool",
    tool_call_index=0,
    parallel_group_id="parallel_capture_001",
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _initial_context() -> DecodeContext:
    return DecodeContext(
        request_id="req_capture_initial",
        kind="initial",
        policy=PUBLIC_HARNESS_POLICY,
    )


def _continuation_context() -> DecodeContext:
    return DecodeContext(
        request_id="req_capture_continuation",
        kind="continuation",
        parent_request_id="req_capture_initial",
        sequence_start=5,
        result_output_ids=MappingProxyType({"call_capture_001": "tool_result_capture_001"}),
        prior_calls=MappingProxyType({"call_capture_001": CALL}),
        policy=PUBLIC_HARNESS_POLICY,
    )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_decode_exact_fixture_requests_and_tool_result(fixture_name: str) -> None:
    directory = CONTRACTS / fixture_name
    expected = _json(directory / "agent-ir.json")
    requests = expected["requests"]
    events = expected["events"]
    assert isinstance(requests, list)
    assert isinstance(events, list)
    adapter = OpenAIChatAdapter()

    initial = adapter.decode_request(
        (directory / "initial-request.json").read_bytes(),
        context=_initial_context(),
    )
    continuation = adapter.decode_request(
        _json(directory / "tool-result-request.json"),
        context=_continuation_context(),
    )

    assert initial.request.model_dump(mode="json", exclude_unset=True) == requests[0]
    assert initial.tool_results == ()
    assert continuation.request.model_dump(mode="json", exclude_unset=True) == requests[1]
    assert [item.model_dump(mode="json", exclude_unset=True) for item in continuation.tool_results] == [
        next(item for item in events if item["type"] == "tool_result")
    ]


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_encode_exact_fixture_streams_without_native_stream_evidence(fixture_name: str) -> None:
    directory = CONTRACTS / fixture_name
    ir = load_agent_ir(_json(directory / "agent-ir.json"))
    adapter = OpenAIChatAdapter()
    tool_events = [
        event
        for event in ir.events
        if str(event.request_id) == "req_capture_initial"
    ]
    final_events = [
        event
        for event in ir.events
        if str(event.request_id) == "req_capture_continuation" and event.type != "tool_result"
    ]

    assert adapter.encode_stream(
        tool_events,
        context=EncodeContext(model="capture-model", response_id="chatcmpl-capture-tool"),
    ) == (directory / "tool-call-stream.sse").read_text()
    assert adapter.encode_stream(
        final_events,
        context=EncodeContext(model="capture-model", response_id="chatcmpl-capture-final"),
    ) == (directory / "final-response.sse").read_text()


def test_default_policy_is_restricted_and_native_evidence_is_off() -> None:
    native = {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decoded = OpenAIChatAdapter().decode_request(
        native,
        context=DecodeContext(request_id="req_default", kind="initial"),
    )
    dumped = decoded.request.model_dump(mode="json", exclude_unset=True)

    assert dumped["sensitivity"] == "restricted"
    assert dumped["provenance"] == {"origin": "unknown", "trust": "untrusted"}
    assert "openai-chat.native_request" not in dumped["request"].get("extensions", {})


def test_named_tool_choice_is_normalized() -> None:
    native = {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": {"type": "function", "function": {"name": "bash"}},
    }
    decoded = OpenAIChatAdapter().decode_request(
        native,
        context=DecodeContext(request_id="req_choice", kind="initial"),
    )
    assert decoded.request.request.model_dump(mode="json", exclude_unset=True)["tool_choice"] == {
        "type": "tool",
        "name": "bash",
    }


@pytest.mark.parametrize(
    ("native", "code"),
    [
        ({"model": "m", "messages": [], "unknown": True}, "unsupported_request_field"),
        (
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://x"}}],
                    }
                ],
            },
            "unsupported_content",
        ),
        (
            {
                "model": "m",
                "messages": [{"role": "tool", "content": "x", "tool_call_id": "missing"}],
            },
            "broken_call_link",
        ),
    ],
)
def test_decode_rejects_unsupported_shapes_and_broken_links(
    native: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(ProtocolAdapterError) as caught:
        OpenAIChatAdapter().decode_request(native, context=_initial_context())
    assert caught.value.code == code


def test_decode_rejects_credentials_when_native_evidence_is_on() -> None:
    native = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"authorization": "Bearer secret-value"},
    }
    with pytest.raises(ProtocolAdapterError) as caught:
        OpenAIChatAdapter().decode_request(native, context=_initial_context())
    assert caught.value.code == "credential_in_evidence"
    assert "secret-value" not in str(caught.value)


def test_decode_does_not_parse_duplicate_argument_keys() -> None:
    native = {
        "model": "model",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_duplicate_json",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"x":1,"x":2}'},
                    }
                ],
            }
        ],
    }

    decoded = OpenAIChatAdapter().decode_request(
        native,
        context=DecodeContext(request_id="req_duplicate_json", kind="initial"),
    )
    tool_call = decoded.request.request.messages[0].content[0]

    assert isinstance(tool_call, ToolCallBlock)
    assert tool_call.arguments_json is None


def test_decode_rejects_two_results_for_one_call() -> None:
    native = _json(CONTRACTS / "pi-0.84.2" / "tool-result-request.json")
    messages = native["messages"]
    assert isinstance(messages, list)
    messages.append(deepcopy(messages[-1]))

    with pytest.raises(ProtocolAdapterError, match="duplicate_tool_result"):
        OpenAIChatAdapter().decode_request(native, context=_continuation_context())


def test_decode_enforces_the_cumulative_block_limit() -> None:
    native = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "three"},
                    {"type": "text", "text": "four"},
                ],
            },
        ],
    }

    with pytest.raises(ProtocolAdapterError, match="too_many_blocks"):
        OpenAIChatAdapter().decode_request(
            native,
            context=DecodeContext(
                request_id="req_block_limit",
                kind="initial",
                limits=AdapterLimits(max_blocks=2),
            ),
        )


def test_decode_rejects_arbitrary_input_and_request_size_limit() -> None:
    adapter = OpenAIChatAdapter()
    with pytest.raises(ProtocolAdapterError, match="invalid_json"):
        adapter.decode_request(object(), context=_initial_context())  # type: ignore[arg-type]
    context = DecodeContext(
        request_id="req_limit",
        kind="initial",
        limits=AdapterLimits(max_request_bytes=8),
    )
    with pytest.raises(ProtocolAdapterError, match="invalid_json"):
        adapter.decode_request('{"model":"too-large"}', context=context)


def test_encode_rejects_reasoning_leakage_and_invalid_lifecycle() -> None:
    directory = CONTRACTS / "pi-0.84.2"
    ir = load_agent_ir(_json(directory / "agent-ir.json"))
    events = [event for event in ir.events if str(event.request_id) == "req_capture_continuation"][1:]
    reasoning = list(events)
    reasoning[0] = reasoning[0].model_copy(update={"content_type": "reasoning"})
    adapter = OpenAIChatAdapter()
    context = EncodeContext(model="capture-model")

    with pytest.raises(ProtocolAdapterError, match="reasoning_leakage"):
        adapter.encode_stream(reasoning, context=context)
    with pytest.raises(ProtocolAdapterError, match="missing_terminal"):
        adapter.encode_stream(events[:-1], context=context)


def test_encode_rejects_changed_argument_fragments() -> None:
    directory = CONTRACTS / "pi-0.84.2"
    ir = load_agent_ir(_json(directory / "agent-ir.json"))
    events = [event for event in ir.events if str(event.request_id) == "req_capture_initial"]
    changed = list(events)
    changed[2] = changed[2].model_copy(update={"delta": "{}"})

    with pytest.raises(ProtocolAdapterError, match="invalid_tool_lifecycle"):
        OpenAIChatAdapter().encode_stream(changed, context=EncodeContext(model="capture-model"))


def test_encode_rejects_arbitrary_input() -> None:
    with pytest.raises(ProtocolAdapterError, match="invalid_event_stream"):
        OpenAIChatAdapter().encode_stream(object(), context=EncodeContext(model="model"))  # type: ignore[arg-type]


def test_encode_rejects_unused_non_json_metadata_and_aggregate_limits() -> None:
    directory = CONTRACTS / "pi-0.84.2"
    ir = load_agent_ir(_json(directory / "agent-ir.json"))
    events = [event for event in ir.events if str(event.request_id) == "req_capture_initial"]
    metadata: dict[str, Any] = {"value": object()}

    with pytest.raises(ProtocolAdapterError, match="unsupported_encode_metadata"):
        OpenAIChatAdapter().encode_stream(
            events,
            context=EncodeContext(model="model", metadata=metadata),
        )
    with pytest.raises(ProtocolAdapterError, match="too_many_events"):
        OpenAIChatAdapter().encode_stream(
            events,
            context=EncodeContext(
                model="model",
                limits=AdapterLimits(max_events=1),
            ),
        )
    with pytest.raises(ProtocolAdapterError, match="sse_stream_too_large"):
        OpenAIChatAdapter().encode_stream(
            events,
            context=EncodeContext(
                model="model",
                limits=AdapterLimits(max_sse_stream_bytes=1),
            ),
        )
