from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from ppmlx.agent_ir import AgentEvent, Origin, Provenance, Sensitivity, Trust
from ppmlx.protocols.anthropic_messages import AnthropicMessagesAdapter
from ppmlx.protocols.base import (
    AdapterLimits,
    CallReference,
    DecodeContext,
    EncodeContext,
    NormalizationPolicy,
    ProtocolAdapterError,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "contracts"
    / "anthropic-messages"
    / "claude-code-2.1.231"
)
EVENT_ADAPTER: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
ADAPTER = AnthropicMessagesAdapter()


def _json(name: str) -> dict[str, object]:
    value = json.loads((FIXTURE / name).read_text())
    assert isinstance(value, dict)
    return value


def _fixture_policy(*, evidence: bool = True) -> NormalizationPolicy:
    return NormalizationPolicy(
        sensitivity=Sensitivity.PUBLIC,
        provenance=Provenance(origin=Origin.HARNESS, trust=Trust.UNTRUSTED),
        include_native_evidence=evidence,
    )


def _call_reference() -> CallReference:
    return CallReference(
        call_id="toolu_capture_001",
        name="Bash",
        choice_index=0,
        output_id="msg_capture_tool",
        tool_call_index=0,
        parallel_group_id="parallel_capture_001",
    )


def _continuation_context(**changes: object) -> DecodeContext:
    values: dict[str, object] = {
        "request_id": "req_capture_continuation",
        "kind": "continuation",
        "parent_request_id": "req_capture_initial",
        "sequence_start": 4,
        "result_output_ids": {"toolu_capture_001": "tool_result_capture_001"},
        "prior_calls": {"toolu_capture_001": _call_reference()},
        "policy": _fixture_policy(),
    }
    values.update(changes)
    return DecodeContext(**values)  # type: ignore[arg-type]


def _events(start: int, stop: int | None = None) -> list[AgentEvent]:
    native = _json("agent-ir.json")["events"]
    assert isinstance(native, list)
    return [EVENT_ADAPTER.validate_python(item) for item in native[start:stop]]


def test_decode_initial_request_matches_the_claude_code_fixture() -> None:
    expected = _json("agent-ir.json")["requests"]
    assert isinstance(expected, list)
    decoded = ADAPTER.decode_request(
        (FIXTURE / "initial-request.json").read_bytes(),
        context=DecodeContext(
            request_id="req_capture_initial",
            kind="initial",
            policy=_fixture_policy(),
        ),
    )

    assert decoded.request.model_dump(mode="json", exclude_unset=True) == expected[0]
    assert decoded.tool_results == ()
    assert decoded.calls == ()


def test_decode_tool_result_request_matches_the_fixture_and_links_the_call() -> None:
    expected = _json("agent-ir.json")
    requests = expected["requests"]
    events = expected["events"]
    assert isinstance(requests, list)
    assert isinstance(events, list)

    decoded = ADAPTER.decode_request(
        (FIXTURE / "tool-result-request.json").read_text(),
        context=_continuation_context(),
    )

    assert decoded.request.model_dump(mode="json", exclude_unset=True) == requests[1]
    assert [item.model_dump(mode="json", exclude_unset=True) for item in decoded.tool_results] == [
        events[4]
    ]
    assert decoded.calls == (_call_reference(),)


@pytest.mark.parametrize(
    ("events", "response_id", "fixture_name"),
    [
        ((_events(0, 4)), "msg_capture_tool", "tool-call-stream.sse"),
        ((_events(5)), "msg_capture_final", "final-response.sse"),
    ],
)
def test_encode_stream_matches_anthropic_sse_without_native_stream_evidence(
    events: list[AgentEvent], response_id: str, fixture_name: str
) -> None:
    cleaned = [event.model_copy(update={"extensions": {}}) for event in events]

    encoded = ADAPTER.encode_stream(
        cleaned,
        context=EncodeContext(model="claude-sonnet-4-6", response_id=response_id),
    )

    assert encoded == (FIXTURE / fixture_name).read_text()


def test_default_policy_is_restricted_untrusted_and_does_not_keep_native_evidence() -> None:
    decoded = ADAPTER.decode_request(
        _json("initial-request.json"),
        context=DecodeContext(request_id="req_default_policy", kind="initial"),
    )
    request = decoded.request.model_dump(mode="json")

    assert request["sensitivity"] == "restricted"
    assert request["provenance"]["origin"] == "unknown"
    assert request["provenance"]["trust"] == "untrusted"
    assert "anthropic-messages.native_request" not in request["request"]["extensions"]
    blocks = request["request"]["messages"][0]["content"]
    assert all("anthropic-messages.native_block" not in block["extensions"] for block in blocks)
    instruction_blocks = request["request"]["instructions"][1]["content"]
    assert instruction_blocks[0]["extensions"]["anthropic-messages.cache_control"] == {
        "cache_control": {"type": "ephemeral"}
    }


def test_decode_keeps_tool_choice_generation_options_and_metadata() -> None:
    native = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "tool_choice": {"type": "tool", "name": "Read", "disable_parallel_tool_use": True},
        "max_tokens": 10,
        "temperature": 0.25,
        "top_p": 0.9,
        "stop_sequences": ["STOP"],
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": []},
        "output_config": {"effort": "high"},
        "service_tier": "auto",
        "metadata": {"user_id": "fixture-user"},
    }

    decoded = ADAPTER.decode_request(
        native,
        context=DecodeContext(request_id="req_options", kind="initial"),
    ).request.request.model_dump(mode="json", exclude_unset=True)

    assert decoded["tool_choice"] == {"type": "tool", "name": "Read"}
    assert decoded["generation"] == {
        "temperature": 0.25,
        "top_p": 0.9,
        "max_output_tokens": 10,
        "stop": ["STOP"],
        "extensions": {
            "anthropic-messages.generation": {
                "thinking": {"type": "adaptive"},
                "context_management": {"edits": []},
                "output_config": {"effort": "high"},
            }
        },
    }
    assert decoded["extensions"] == {
        "anthropic-messages.request_options": {
            "disable_parallel_tool_use": True,
            "service_tier": "auto",
        }
    }
    assert decoded["metadata"] == {"user_id": "fixture-user"}


@pytest.mark.parametrize(
    "block",
    [
        {"type": "thinking", "thinking": "private", "signature": "secret"},
        {"type": "redacted_thinking", "data": "private"},
        {"type": "text", "text": "safe", "signature": "secret"},
    ],
)
def test_decode_rejects_thinking_and_signature_blocks(block: dict[str, object]) -> None:
    native = {"model": "claude-test", "messages": [{"role": "assistant", "content": [block]}]}

    with pytest.raises(ProtocolAdapterError):
        ADAPTER.decode_request(native, context=DecodeContext(request_id="req_bad", kind="initial"))


def test_decode_rejects_a_broken_tool_result_link() -> None:
    with pytest.raises(ProtocolAdapterError, match="broken_tool_link"):
        ADAPTER.decode_request(
            _json("tool-result-request.json"),
            context=_continuation_context(prior_calls={}),
        )


def test_decode_rejects_two_results_for_one_call() -> None:
    native = _json("tool-result-request.json")
    messages = native["messages"]
    assert isinstance(messages, list)
    last_message = messages[-1]
    assert isinstance(last_message, dict)
    content = last_message["content"]
    assert isinstance(content, list)
    content.append(deepcopy(content[-1]))

    with pytest.raises(ProtocolAdapterError, match="duplicate_tool_result"):
        ADAPTER.decode_request(native, context=_continuation_context())


def test_decode_rejects_duplicate_call_ids_and_tool_names() -> None:
    duplicate_calls = {
        "model": "model",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_x", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "call_x", "name": "Read", "input": {}},
                ],
            }
        ],
    }
    duplicate_tools = {
        "model": "model",
        "messages": [],
        "tools": [
            {"name": "Read", "description": "one", "input_schema": {}},
            {"name": "Read", "description": "two", "input_schema": {}},
        ],
    }

    with pytest.raises(ProtocolAdapterError, match="duplicate_call_id"):
        ADAPTER.decode_request(
            duplicate_calls,
            context=DecodeContext(request_id="req_duplicate_call", kind="initial"),
        )
    with pytest.raises(ProtocolAdapterError, match="duplicate_tool_name"):
        ADAPTER.decode_request(
            duplicate_tools,
            context=DecodeContext(request_id="req_duplicate_tool", kind="initial"),
        )


def test_decode_rejects_credentials_before_it_keeps_native_evidence() -> None:
    native = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"authorization": "Bearer fixture-secret-value"},
    }

    with pytest.raises(ProtocolAdapterError, match="credential_in_evidence") as error:
        ADAPTER.decode_request(
            native,
            context=DecodeContext(
                request_id="req_secret",
                kind="initial",
                policy=NormalizationPolicy(include_native_evidence=True),
            ),
        )
    assert "fixture-secret-value" not in str(error.value)


def test_decode_rejects_non_json_objects_and_limit_violations() -> None:
    with pytest.raises(ProtocolAdapterError, match="invalid_json"):
        ADAPTER.decode_request(
            {"model": "claude-test", "messages": [], "metadata": {"bad": object()}},
            context=DecodeContext(request_id="req_object", kind="initial"),
        )

    with pytest.raises(ProtocolAdapterError, match="too_many_blocks"):
        ADAPTER.decode_request(
            {"model": "claude-test", "messages": [{"role": "user", "content": "Hello"}]},
            context=DecodeContext(
                request_id="req_blocks",
                kind="initial",
                limits=AdapterLimits(max_blocks=0),
            ),
        )

    with pytest.raises(ProtocolAdapterError, match="too_many_tools"):
        ADAPTER.decode_request(
            {
                "model": "claude-test",
                "messages": [],
                "tools": [{"name": "Read", "description": "Read", "input_schema": {}}],
            },
            context=DecodeContext(
                request_id="req_tools",
                kind="initial",
                limits=AdapterLimits(max_tools=0),
            ),
        )


@pytest.mark.parametrize(
    ("events", "code"),
    [
        (_events(1, 4), "invalid_lifecycle"),
        (_events(0, 3), "missing_terminal"),
        (
            _events(0, 3)
            + _events(3, 4)
            + [_events(3, 4)[0].model_copy(update={"sequence": 4})],
            "event_after_terminal",
        ),
    ],
)
def test_encode_rejects_invalid_lifecycle_or_terminal_state(
    events: list[AgentEvent], code: str
) -> None:
    with pytest.raises(ProtocolAdapterError, match=code):
        ADAPTER.encode_stream(events, context=EncodeContext(model="claude-test"))


def test_encode_rejects_unsupported_reasoning_and_argument_mismatch() -> None:
    reasoning_start = _events(5, 6)[0].model_copy(update={"content_type": "reasoning"})
    with pytest.raises(ProtocolAdapterError, match="unsupported_content"):
        ADAPTER.encode_stream(
            [reasoning_start, *_events(6, 9)], context=EncodeContext(model="claude-test")
        )

    changed = _events(0, 4)
    changed[1] = changed[1].model_copy(update={"delta": "{}"})
    with pytest.raises(ProtocolAdapterError, match="arguments_mismatch"):
        ADAPTER.encode_stream(changed, context=EncodeContext(model="claude-test"))


def test_encode_rejects_mixed_outputs_and_small_frame_limits() -> None:
    mixed = _events(5)
    mixed[1] = mixed[1].model_copy(update={"output_id": "msg_other"})
    with pytest.raises(ProtocolAdapterError, match="mixed_output"):
        ADAPTER.encode_stream(mixed, context=EncodeContext(model="claude-test"))

    with pytest.raises(ProtocolAdapterError, match="sse_frame_too_large"):
        ADAPTER.encode_stream(
            _events(5),
            context=EncodeContext(
                model="claude-test", limits=AdapterLimits(max_sse_frame_bytes=32)
            ),
        )


def test_encode_sanitizes_arbitrary_events_and_rejects_unused_metadata() -> None:
    with pytest.raises(ProtocolAdapterError) as error:
        ADAPTER.encode_stream(
            [object()],  # type: ignore[list-item]
            context=EncodeContext(model="model"),
        )
    assert error.value.code in {"unsupported_event", "invalid_adapter_input"}
    assert error.value.__cause__ is None
    assert error.value.__context__ is None

    metadata: dict[str, Any] = {"value": object()}
    with pytest.raises(ProtocolAdapterError, match="unsupported_encode_metadata"):
        ADAPTER.encode_stream(
            _events(0, 4),
            context=EncodeContext(model="model", metadata=metadata),
        )


def test_encode_enforces_aggregate_event_and_stream_limits() -> None:
    events = _events(0, 4)

    with pytest.raises(ProtocolAdapterError, match="too_many_events"):
        ADAPTER.encode_stream(
            events,
            context=EncodeContext(
                model="model",
                limits=AdapterLimits(max_events=1),
            ),
        )
    with pytest.raises(ProtocolAdapterError, match="sse_stream_too_large"):
        ADAPTER.encode_stream(
            events,
            context=EncodeContext(
                model="model",
                limits=AdapterLimits(max_sse_stream_bytes=1),
            ),
        )
