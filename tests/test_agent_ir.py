"""Contract tests for the public Agent IR serializer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from ppmlx.agent_ir import (
    AgentIRValidationError,
    dump_agent_ir,
    encode_agent_ir,
    load_agent_ir,
    new_call_id,
    new_conversation_id,
    new_output_id,
    new_parallel_group_id,
    new_request_id,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "contracts"
SCHEMA_PATH = ROOT / "docs" / "architecture" / "schema" / "agent-ir-v1.schema.json"
SCHEMA_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def _fixtures() -> list[Path]:
    return sorted(FIXTURE_ROOT.glob("**/agent-ir.json"))


def _round_trip(value: dict[str, Any]) -> dict[str, Any]:
    loaded = load_agent_ir(copy.deepcopy(value))
    dumped = dump_agent_ir(loaded)
    SCHEMA_VALIDATOR.validate(dumped)
    return dumped


@pytest.mark.parametrize("path", _fixtures(), ids=lambda path: str(path.parent.relative_to(FIXTURE_ROOT)))
def test_phase_one_fixtures_round_trip_without_loss(path: Path) -> None:
    value = json.loads(path.read_text())
    assert _round_trip(value) == value


def _base() -> dict[str, Any]:
    return {
        "ir_version": "agent-ir/v1",
        "conversation_id": "conv_synthetic",
        "source": {
            "harness": "synthetic",
            "harness_version": "1.0.0",
            "protocol": "openai-chat",
            "protocol_version": "v1",
        },
        "requests": [{
            "request_id": "req_initial",
            "kind": "initial",
            "request": {
                "model": "local-model",
                "instructions": [],
                "messages": [],
                "tools": [],
                "stream": True,
            },
        }],
        "events": [],
    }


def _event(event_type: str, sequence: int, **fields: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "request_id": "req_initial",
        "sequence": sequence,
        "choice_index": 0,
        "output_id": "out_1",
        **fields,
    }


@pytest.mark.parametrize("block", [
    {"type": "image", "media_type": "image/png", "url": "https://example.invalid/image.png"},
    {"type": "document", "media_type": "text/plain", "text": "Document text."},
    {"type": "reasoning", "data": True},
    {"type": "refusal", "text": "I cannot do that."},
    {"type": "extension", "namespace": "provider.block", "data": {"vendor": True}, "required": False},
], ids=lambda block: block["type"])
def test_content_blocks_are_serializable(block: dict[str, Any]) -> None:
    value = _base()
    value["requests"][0]["request"]["messages"] = [{
        "role": "assistant",
        "content": [block],
    }]
    value["events"] = [
        _event("content.started", 0, content_index=0, content_type=block["type"]),
        _event("content.completed", 1, content_index=0, content=block),
        _event("response.completed", 2, finish_reason="stop"),
    ]
    assert _round_trip(value) == value


@pytest.mark.parametrize("terminal", [
    _event("response.cancelled", 0, reason="client_cancelled"),
    _event("response.failed", 0, error={
        "code": "provider_unavailable",
        "category": "provider",
        "message": "Provider unavailable.",
        "retryable": True,
    }),
])
def test_control_terminal_events_are_serializable(terminal: dict[str, Any]) -> None:
    value = _base()
    value["events"] = [terminal]
    assert _round_trip(value) == value


def test_parallel_tool_calls_preserve_group_and_order() -> None:
    value = _base()
    value["requests"][0]["request"]["tools"] = [
        {"name": "read", "description": "Read.", "input_schema": {"type": "object"}},
        {"name": "write", "description": "Write.", "input_schema": {"type": "object"}},
    ]
    group = "parallel_1"
    value["events"] = [
        _event("tool_call.started", 0, tool_call_index=0, parallel_group_id=group, call_id="call_a", name="read"),
        _event("tool_call.started", 1, tool_call_index=1, parallel_group_id=group, call_id="call_b", name="write"),
        _event("tool_call.completed", 2, tool_call_index=0, parallel_group_id=group, call_id="call_a", name="read", arguments_raw="{}", arguments_json={}),
        _event("tool_call.completed", 3, tool_call_index=1, parallel_group_id=group, call_id="call_b", name="write", arguments_raw="{}", arguments_json={}),
        _event("response.completed", 4, finish_reason="tool_calls"),
    ]
    assert _round_trip(value) == value


def _assert_rejected(value: Any) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        load_agent_ir(value)


def test_unknown_top_level_fields_are_rejected() -> None:
    value = _base()
    value["unknown"] = True
    _assert_rejected(value)


def test_validation_errors_do_not_retain_private_input() -> None:
    secret = "sk-test-THIS_IS_SECRET_123456"
    value = _base()
    value["unknown"] = secret
    with pytest.raises(AgentIRValidationError) as captured:
        load_agent_ir(value)
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_explicit_null_is_not_an_absent_optional_value() -> None:
    value = _base()
    value["sensitivity"] = None
    _assert_rejected(value)


@pytest.mark.parametrize("field,value", [
    ("conversation_id", "bad"),
    ("conversation_id", "conv with spaces"),
    ("requests", [{"request_id": "bad id", "kind": "initial", "request": {}}]),
])
def test_identifiers_are_validated(field: str, value: Any) -> None:
    candidate = _base()
    candidate[field] = value
    _assert_rejected(candidate)


def test_arbitrary_python_extension_objects_are_rejected() -> None:
    value = _base()
    value["extensions"] = {"provider.data": object()}
    _assert_rejected(value)


def test_extension_names_require_a_namespace() -> None:
    value = _base()
    value["extensions"] = {"unnamespaced": True}
    _assert_rejected(value)


@pytest.mark.parametrize("path", [
    ("source", "protocol"),
    ("sensitivity",),
])
def test_enum_fields_reject_byte_strings(path: tuple[str, ...]) -> None:
    value = _base()
    target: dict[str, Any] = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = b"openai-chat" if path[-1] == "protocol" else b"restricted"
    _assert_rejected(value)


@pytest.mark.parametrize("mutator", [
    lambda value: value["requests"].append({"request_id": "req_child", "kind": "continuation", "request": {}}),
    lambda value: value["requests"][0].update({"kind": "continuation", "parent_request_id": "req_missing"}),
])
def test_request_linkage_is_validated(mutator: Any) -> None:
    value = _base()
    mutator(value)
    _assert_rejected(value)


def test_event_sequence_and_request_reference_are_validated() -> None:
    value = _base()
    value["events"] = [_event("response.completed", 1, finish_reason="stop"), _event("response.completed", 0, finish_reason="stop")]
    _assert_rejected(value)
    value["events"][0]["request_id"] = "req_missing"
    _assert_rejected(value)


@pytest.mark.parametrize("event_type", ["response.completed", "response.refused", "response.cancelled", "response.failed"])
def test_one_terminal_event_is_required(event_type: str) -> None:
    value = _base()
    value["events"] = [_event(event_type, 0)]
    if event_type == "response.completed":
        value["events"][0]["finish_reason"] = "stop"
    elif event_type == "response.refused":
        value["events"][0]["refusal"] = {"type": "refusal", "text": "No."}
    elif event_type == "response.cancelled":
        value["events"][0]["reason"] = "cancelled"
    else:
        value["events"][0]["error"] = {"code": "x", "category": "provider", "message": "x", "retryable": False}
    assert _round_trip(value) == value


def test_tool_argument_deltas_must_assemble_to_completed_raw_value() -> None:
    value = _base()
    value["events"] = [
        _event("tool_call.started", 0, tool_call_index=0, call_id="call_1", name="read"),
        _event("tool_call.arguments.delta", 1, tool_call_index=0, call_id="call_1", delta='{"path":"'),
        _event("tool_call.arguments.delta", 2, tool_call_index=0, call_id="call_1", delta='x"}'),
        _event("tool_call.completed", 3, tool_call_index=0, call_id="call_1", name="read", arguments_raw='{"path":"x"}', arguments_json={"path": "x"}),
        _event("response.completed", 4, finish_reason="tool_calls"),
    ]
    assert _round_trip(value) == value
    value["events"][3]["arguments_raw"] = '{"path":"y"}'
    _assert_rejected(value)


def test_parsed_tool_arguments_must_match_the_raw_value() -> None:
    value = _base()
    value["events"] = [
        _event("tool_call.started", 0, tool_call_index=0, call_id="call_1", name="read"),
        _event(
            "tool_call.completed",
            1,
            tool_call_index=0,
            call_id="call_1",
            name="read",
            arguments_raw='{"path":"x"}',
            arguments_json={"path": "y"},
        ),
        _event("response.completed", 2, finish_reason="tool_calls"),
    ]
    _assert_rejected(value)
    value["events"][1]["arguments_raw"] = "not-json"
    _assert_rejected(value)


def test_continuation_cannot_change_contract_fields() -> None:
    value = _base()
    continuation = copy.deepcopy(value["requests"][0])
    continuation.update({"request_id": "req_child", "kind": "continuation", "parent_request_id": "req_initial"})
    continuation["request"] = copy.deepcopy(continuation["request"])
    continuation["request"]["model"] = "different-model"
    value["requests"].append(continuation)
    _assert_rejected(value)


def test_continuation_keeps_routing_metadata_and_native_evidence_keys() -> None:
    value = _base()
    request = value["requests"][0]["request"]
    request["metadata"] = {"route": "local", "client_metadata": {"turn": 1}}
    request["extensions"] = {
        "openai-chat.native_request": {"turn": 1},
        "openai-chat.request_options": {"store": False},
    }
    continuation = copy.deepcopy(value["requests"][0])
    continuation.update({
        "request_id": "req_child",
        "kind": "continuation",
        "parent_request_id": "req_initial",
    })
    continuation["request"]["metadata"]["client_metadata"] = {"turn": 2}
    continuation["request"]["extensions"]["openai-chat.native_request"] = {"turn": 2}
    value["requests"].append(continuation)
    assert _round_trip(value) == value

    changed_route = copy.deepcopy(value)
    changed_route["requests"][1]["request"]["metadata"]["route"] = "remote"
    _assert_rejected(changed_route)

    removed_evidence = copy.deepcopy(value)
    del removed_evidence["requests"][1]["request"]["extensions"]["openai-chat.native_request"]
    _assert_rejected(removed_evidence)


def test_continuation_cannot_downgrade_prior_message_policy() -> None:
    value = _base()
    value["requests"][0]["request"]["messages"] = [{
        "role": "user",
        "content": [{"type": "text", "text": "private", "sensitivity": "restricted"}],
    }]
    continuation = copy.deepcopy(value["requests"][0])
    continuation.update({
        "request_id": "req_child",
        "kind": "continuation",
        "parent_request_id": "req_initial",
    })
    continuation["request"]["messages"][0]["content"][0]["sensitivity"] = "public"
    value["requests"].append(continuation)
    _assert_rejected(value)


def test_continuation_ignores_only_content_block_native_evidence() -> None:
    value = _base()
    value["requests"][0]["request"]["messages"] = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "hello",
            "extensions": {"provider.native_block": {"cache": True}},
        }],
    }]
    continuation = copy.deepcopy(value["requests"][0])
    continuation.update({
        "request_id": "req_child",
        "kind": "continuation",
        "parent_request_id": "req_initial",
    })
    del continuation["request"]["messages"][0]["content"][0]["extensions"]
    value["requests"].append(continuation)
    assert _round_trip(value) == value

    changed_semantic_extension = copy.deepcopy(value)
    original_block = changed_semantic_extension["requests"][0]["request"]["messages"][0]["content"][0]
    continued_block = changed_semantic_extension["requests"][1]["request"]["messages"][0]["content"][0]
    original_block["extensions"] = {"provider.semantic": "A"}
    continued_block["extensions"] = {"provider.semantic": "B"}
    _assert_rejected(changed_semantic_extension)


def test_continuation_keeps_opaque_reasoning_data_exact() -> None:
    value = _base()
    value["requests"][0]["request"]["messages"] = [{
        "role": "assistant",
        "content": [{
            "type": "reasoning",
            "data": {"extensions": {"provider.native_block": {"ciphertext": "A"}}},
        }],
    }]
    continuation = copy.deepcopy(value["requests"][0])
    continuation.update({
        "request_id": "req_child",
        "kind": "continuation",
        "parent_request_id": "req_initial",
    })
    continuation["request"]["messages"][0]["content"][0]["data"]["extensions"][
        "provider.native_block"
    ]["ciphertext"] = "B"
    value["requests"].append(continuation)
    _assert_rejected(value)


def test_nested_content_cannot_reduce_parent_sensitivity() -> None:
    value = _base()
    value["sensitivity"] = "restricted"
    value["requests"][0]["request"]["messages"] = [{
        "role": "user",
        "content": [{"type": "text", "text": "private", "sensitivity": "public"}],
    }]
    _assert_rejected(value)


def test_message_tool_call_parsed_arguments_must_match_raw_value() -> None:
    value = _base()
    value["requests"][0]["request"]["messages"] = [{
        "role": "assistant",
        "content": [{
            "type": "tool_call",
            "call_id": "call_1",
            "name": "read",
            "arguments_raw": '{"path":"x"}',
            "arguments_json": {"path": "y"},
        }],
    }]
    _assert_rejected(value)


def test_tool_result_requires_the_linked_continuation_request() -> None:
    value = _base()
    continuation = copy.deepcopy(value["requests"][0])
    continuation.update({
        "request_id": "req_child",
        "kind": "continuation",
        "parent_request_id": "req_initial",
    })
    value["requests"].append(continuation)
    value["events"] = [
        _event("tool_call.started", 0, tool_call_index=0, call_id="call_1", name="read"),
        _event(
            "tool_call.completed",
            1,
            tool_call_index=0,
            call_id="call_1",
            name="read",
            arguments_raw="{}",
            arguments_json={},
        ),
        _event("response.completed", 2, finish_reason="tool_calls"),
        {
            **_event(
                "tool_result",
                3,
                tool_call_index=0,
                call_id="call_1",
                content=[{"type": "text", "text": "ok"}],
                is_error=False,
            ),
            "request_id": "req_child",
            "output_id": "result_1",
        },
    ]
    assert _round_trip(value) == value
    value["events"][3]["request_id"] = "req_initial"
    _assert_rejected(value)


def test_completed_response_rejects_unfinished_output_lifecycles() -> None:
    value = _base()
    value["events"] = [
        _event("content.started", 0, content_index=0, content_type="text"),
        _event("response.completed", 1, finish_reason="stop"),
    ]
    _assert_rejected(value)
    value["events"] = [
        _event("tool_call.started", 0, tool_call_index=0, call_id="call_1", name="read"),
        _event("response.completed", 1, finish_reason="tool_calls"),
    ]
    _assert_rejected(value)


@pytest.mark.parametrize("field,bad_value", [
    ("stream", "false"),
    ("stream", 0),
])
def test_request_scalars_are_not_coerced(field: str, bad_value: Any) -> None:
    value = _base()
    value["requests"][0]["request"][field] = bad_value
    _assert_rejected(value)


@pytest.mark.parametrize("field,bad_value", [
    ("sequence", 0.0),
    ("sequence", "0"),
    ("choice_index", False),
])
def test_event_scalars_are_not_coerced(field: str, bad_value: Any) -> None:
    value = _base()
    event = _event("response.completed", 0, finish_reason="stop")
    event[field] = bad_value
    value["events"] = [event]
    _assert_rejected(value)


@pytest.mark.parametrize("bad_value", [float("nan"), (1, 2), object()])
def test_extensions_accept_only_json_values(bad_value: Any) -> None:
    value = _base()
    value["extensions"] = {"provider.data": bad_value}
    _assert_rejected(value)


def test_json_text_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    _assert_rejected('{"ir_version":"agent-ir/v1","ir_version":"agent-ir/v1"}')
    _assert_rejected(json.dumps(_base()).replace('"events": []', '"events": [], "x": NaN'))


def test_json_codec_accepts_text_bytes_and_bytearray() -> None:
    value = _base()
    encoded = json.dumps(value)
    for source in (encoded, encoded.encode(), bytearray(encoded.encode())):
        assert dump_agent_ir(load_agent_ir(source)) == value
    assert json.loads(encode_agent_ir(load_agent_ir(value))) == value


def test_absent_policy_defaults_stay_absent_in_serialized_data() -> None:
    value = _base()
    loaded = load_agent_ir(value)
    assert loaded.sensitivity.value == "restricted"
    assert loaded.provenance.origin.value == "unknown"
    assert "sensitivity" not in dump_agent_ir(loaded)
    assert "provenance" not in dump_agent_ir(loaded)


def test_identifier_factories_create_valid_unique_values() -> None:
    factories = (
        new_conversation_id,
        new_request_id,
        new_call_id,
        new_output_id,
        new_parallel_group_id,
    )
    for factory in factories:
        first = factory()
        second = factory()
        assert first != second
        assert first.split("_", 1)[1].isalnum()


def test_schema_rejects_each_negative_shape_even_before_runtime_exists() -> None:
    value = _base()
    value["unknown"] = True
    assert list(SCHEMA_VALIDATOR.iter_errors(value))


@pytest.mark.parametrize("block", [
    {"type": "reasoning", "data": None},
    {"type": "extension", "namespace": "provider.block", "data": None, "required": False},
    {
        "type": "tool_call",
        "call_id": "call_1",
        "name": "read",
        "arguments_raw": "null",
        "arguments_json": None,
    },
])
def test_schema_and_loader_reject_explicit_null_block_data(block: dict[str, Any]) -> None:
    value = _base()
    value["requests"][0]["request"]["messages"] = [{"role": "assistant", "content": [block]}]
    assert list(SCHEMA_VALIDATOR.iter_errors(value))
    _assert_rejected(value)
