from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from collections.abc import Mapping

from ppmlx.agent_ir import (
    NamedToolChoice,
    Origin,
    Provenance,
    Sensitivity,
    Trust,
    Usage,
    UsageSource,
    load_agent_ir,
)
from ppmlx.protocols.base import (
    AdapterLimits,
    CallReference,
    DecodeContext,
    EncodeContext,
    NormalizationPolicy,
    ProtocolAdapterError,
)
from ppmlx.protocols.openai_responses import OpenAIResponsesAdapter
from ppmlx.protocols.sse import parse_sse


FIXTURE_DIR = Path("tests/fixtures/contracts/openai-responses/codex-0.147.0")


def _json(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def _fixture_policy(*, evidence: bool = True) -> NormalizationPolicy:
    return NormalizationPolicy(
        sensitivity=Sensitivity.PUBLIC,
        provenance=Provenance(origin=Origin.HARNESS, trust=Trust.UNTRUSTED),
        include_native_evidence=evidence,
    )


def _call_reference() -> CallReference:
    return CallReference(
        call_id="call_capture_001",
        name="exec_command",
        choice_index=0,
        output_id="fc_capture_001",
        tool_call_index=0,
        parallel_group_id="parallel_capture_001",
    )


def _initial_context(
    *,
    policy: NormalizationPolicy | None = None,
    limits: AdapterLimits | None = None,
) -> DecodeContext:
    return DecodeContext(
        request_id="req_capture_initial",
        kind="initial",
        policy=policy or _fixture_policy(),
        limits=limits or AdapterLimits(),
    )


def _continuation_context(
    *,
    prior_calls: Mapping[str, CallReference] | None = None,
) -> DecodeContext:
    reference = _call_reference()
    return DecodeContext(
        request_id="req_capture_continuation",
        kind="continuation",
        parent_request_id="req_capture_initial",
        sequence_start=4,
        result_output_ids={reference.call_id: "fco_capture_001"},
        prior_calls={reference.call_id: reference} if prior_calls is None else prior_calls,
        policy=_fixture_policy(),
    )


def _error_code(error: pytest.ExceptionInfo[ProtocolAdapterError]) -> str:
    return error.value.code


def test_decodes_exact_codex_initial_request():
    expected = _json("agent-ir.json")["requests"][0]
    decoded = OpenAIResponsesAdapter().decode_request(
        (FIXTURE_DIR / "initial-request.json").read_bytes(),
        context=_initial_context(),
    )

    assert decoded.request.model_dump(mode="json", exclude_unset=True) == expected
    assert decoded.tool_results == ()
    assert decoded.calls == ()


def test_decodes_exact_codex_tool_result_request_and_link_data():
    agent_ir = _json("agent-ir.json")
    decoded = OpenAIResponsesAdapter().decode_request(
        (FIXTURE_DIR / "tool-result-request.json").read_bytes(),
        context=_continuation_context(),
    )

    assert decoded.request.model_dump(mode="json", exclude_unset=True) == agent_ir["requests"][1]
    assert [item.model_dump(mode="json", exclude_unset=True) for item in decoded.tool_results] == [
        agent_ir["events"][4]
    ]
    assert decoded.calls == (_call_reference(),)


def test_default_policy_is_restricted_untrusted_and_has_no_native_evidence():
    decoded = OpenAIResponsesAdapter().decode_request(
        _json("initial-request.json"),
        context=DecodeContext(request_id="req_default", kind="initial"),
    )
    value = decoded.request.model_dump(mode="json", exclude_unset=True)

    assert value["sensitivity"] == "restricted"
    assert value["provenance"] == {"origin": "unknown", "trust": "untrusted"}
    assert "openai-responses.native_request" not in value["request"]["extensions"]
    assert "extensions" not in value["request"]["messages"][0]
    assert "extensions" not in value["request"]["tools"][0]
    assert value["request"]["extensions"]["openai-responses.request_options"] == {
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": False,
        "store": False,
    }


@pytest.mark.parametrize(
    ("event_slice", "response_id", "fixture_name"),
    [
        (slice(0, 4), "resp_capture_tool", "tool-call-stream.sse"),
        (slice(5, None), "resp_capture_final", "final-response.sse"),
    ],
)
def test_encodes_exact_responses_sse_without_native_stream_evidence(
    event_slice: slice,
    response_id: str,
    fixture_name: str,
):
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())
    events = [event.model_copy(update={"extensions": {}}) for event in agent_ir.events[event_slice]]

    encoded = OpenAIResponsesAdapter().encode_stream(
        events,
        context=EncodeContext(model="capture-model", created_at=0, response_id=response_id),
    )

    assert parse_sse(encoded, protocol="openai-responses") == parse_sse(
        (FIXTURE_DIR / fixture_name).read_text(),
        protocol="openai-responses",
    )


def test_decodes_named_function_choice_and_generation_scalars():
    request = _json("initial-request.json")
    request["tool_choice"] = {"type": "function", "name": "exec_command"}
    request["temperature"] = 0.2
    request["top_p"] = 0.9
    request["max_output_tokens"] = 200
    request["reasoning"] = {"effort": "high", "summary": "auto"}

    decoded = OpenAIResponsesAdapter().decode_request(
        request,
        context=_initial_context(policy=_fixture_policy(evidence=False)),
    )
    normalized = decoded.request.request

    assert isinstance(normalized.tool_choice, NamedToolChoice)
    assert normalized.tool_choice.model_dump(mode="json") == {"type": "tool", "name": "exec_command"}
    assert normalized.generation is not None
    assert normalized.generation.temperature == 0.2
    assert normalized.generation.top_p == 0.9
    assert normalized.generation.max_output_tokens == 200
    assert normalized.generation.reasoning_effort == "high"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda value: value.update({"unsupported": True}), "unsupported_field"),
        (lambda value: value["input"][0].update({"unknown": 1}), "unsupported_field"),
        (lambda value: value["input"][0]["content"][0].update({"annotations": []}), "unsupported_field"),
        (lambda value: value["input"][0].update({"type": "reasoning"}), "unsupported_item_type"),
        (lambda value: value["input"][0]["content"][0].update({"type": "output_text"}), "unsupported_content_type"),
        (lambda value: value["tools"][0].update({"type": "web_search"}), "unsupported_tool_type"),
        (lambda value: value.update({"reasoning": {"encrypted_content": "private"}}), "unsupported_field"),
    ],
)
def test_rejects_unsupported_request_data(change, code: str):
    request = _json("initial-request.json")
    change(request)

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(request, context=_initial_context())

    assert _error_code(error) == code


def test_rejects_broken_call_and_result_links():
    request = _json("tool-result-request.json")
    request["input"][1]["id"] = "fc_wrong"

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(request, context=_continuation_context())
    assert _error_code(error) == "broken_call_link"

    request = _json("tool-result-request.json")
    request["input"][2]["id"] = "fco_wrong"
    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(request, context=_continuation_context())
    assert _error_code(error) == "broken_result_link"


def test_rejects_duplicate_keys_in_function_arguments():
    request = {
        "model": "capture-model",
        "input": [
            {
                "type": "function_call",
                "id": "fc_duplicate_json",
                "call_id": "call_duplicate_json",
                "name": "exec_command",
                "arguments": '{"x":1,"x":2}',
            }
        ],
    }
    context = DecodeContext(
        request_id="req_duplicate_json",
        kind="continuation",
        parent_request_id="req_parent",
    )

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(request, context=context)

    assert _error_code(error) == "invalid_arguments"


def test_rejects_duplicate_native_item_ids():
    request = {
        "model": "capture-model",
        "input": [
            {
                "type": "function_call",
                "id": "fc_duplicate",
                "call_id": "call_one",
                "name": "exec_command",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "id": "fc_duplicate",
                "call_id": "call_two",
                "name": "exec_command",
                "arguments": "{}",
            },
        ],
    }

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            request,
            context=DecodeContext(
                request_id="req_duplicate_items",
                kind="continuation",
                parent_request_id="req_parent",
            ),
        )

    assert _error_code(error) == "duplicate_item_id"


def test_decode_enforces_the_cumulative_block_limit():
    request = {
        "model": "capture-model",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "one"},
                    {"type": "input_text", "text": "two"},
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "three"},
                    {"type": "input_text", "text": "four"},
                ],
            },
        ],
    }

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            request,
            context=DecodeContext(
                request_id="req_block_limit",
                kind="initial",
                limits=AdapterLimits(max_blocks=2),
            ),
        )

    assert _error_code(error) == "too_many_blocks"


def test_uses_injected_result_output_id_when_native_id_is_absent():
    request = _json("tool-result-request.json")
    del request["input"][2]["id"]

    decoded = OpenAIResponsesAdapter().decode_request(request, context=_continuation_context())

    assert decoded.request.request.messages[-1].id == "fco_capture_001"
    assert str(decoded.tool_results[0].output_id) == "fco_capture_001"


def test_rejects_result_before_call_and_duplicate_call_id():
    request = _json("tool-result-request.json")
    request["input"][1], request["input"][2] = request["input"][2], request["input"][1]
    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            request,
            context=_continuation_context(prior_calls={}),
        )
    assert _error_code(error) == "broken_call_link"

    request = _json("tool-result-request.json")
    request["input"].insert(2, deepcopy(request["input"][1]))
    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(request, context=_continuation_context())
    assert _error_code(error) == "duplicate_call_id"


def test_rejects_credentials_limits_duplicate_keys_and_arbitrary_input():
    request = _json("initial-request.json")
    request["client_metadata"]["authorization"] = "Bearer private-token"
    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(request, context=_initial_context())
    assert _error_code(error) == "credential_in_evidence"

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            (FIXTURE_DIR / "initial-request.json").read_bytes(),
            context=_initial_context(limits=AdapterLimits(max_request_bytes=10)),
        )
    assert _error_code(error) == "invalid_json"

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            '{"model":"a","model":"b","input":[]}',
            context=_initial_context(),
        )
    assert _error_code(error) == "invalid_json"

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            {"model": "capture-model", "input": [object()]},
            context=_initial_context(),
        )
    assert _error_code(error) == "invalid_json"


def test_rejects_invalid_request_context():
    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().decode_request(
            _json("initial-request.json"),
            context=DecodeContext(
                request_id="req_bad",
                kind="continuation",
                policy=_fixture_policy(),
            ),
        )
    assert _error_code(error) == "missing_parent_request_id"


def test_rejects_invalid_encode_lifecycles_and_unsupported_events():
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())
    adapter = OpenAIResponsesAdapter()
    context = EncodeContext(model="capture-model", response_id="resp_test")

    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream(agent_ir.events[1:4], context=context)
    assert _error_code(error) == "invalid_tool_lifecycle"

    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream([agent_ir.events[4]], context=context)
    assert _error_code(error) == "unsupported_event_type"

    wrong_arguments = agent_ir.events[2].model_copy(update={"arguments_raw": "{}"})
    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream([agent_ir.events[0], agent_ir.events[1], wrong_arguments, agent_ir.events[3]], context=context)
    assert _error_code(error) == "invalid_tool_lifecycle"

    wrong_terminal = agent_ir.events[8].model_copy(update={"finish_reason": "stop"})
    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream([*agent_ir.events[5:8], wrong_terminal], context=context)
    assert _error_code(error) == "invalid_terminal_event"


def test_completed_call_without_deltas_still_enforces_the_argument_limit():
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())
    events = [agent_ir.events[0], agent_ir.events[2], agent_ir.events[3]]

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().encode_stream(
            events,
            context=EncodeContext(
                model="capture-model",
                response_id="resp_test",
                limits=AdapterLimits(max_arguments_bytes=1),
            ),
        )

    assert _error_code(error) == "arguments_too_large"


def test_rejects_mixed_requests_non_increasing_sequences_and_credentials_in_encode_metadata():
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())
    adapter = OpenAIResponsesAdapter()
    context = EncodeContext(model="capture-model", response_id="resp_test")

    mixed = agent_ir.events[1].model_copy(update={"request_id": "req_other"})
    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream([agent_ir.events[0], mixed, agent_ir.events[2], agent_ir.events[3]], context=context)
    assert _error_code(error) == "mixed_request_ids"

    lower_sequence = agent_ir.events[1].model_copy(update={"sequence": 0})
    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream(
            [agent_ir.events[0], lower_sequence, agent_ir.events[2], agent_ir.events[3]],
            context=context,
        )
    assert _error_code(error) == "invalid_event_sequence"

    with pytest.raises(ProtocolAdapterError) as error:
        adapter.encode_stream(
            agent_ir.events[:4],
            context=EncodeContext(
                model="capture-model",
                response_id="resp_test",
                metadata={"authorization": "Bearer private-token"},
            ),
        )
    assert _error_code(error) == "credential_in_evidence"


def test_encode_rejects_non_json_metadata_and_aggregate_limits():
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())
    events = list(agent_ir.events[:4])
    tuple_metadata: dict[str, Any] = {"nested": ("Bearer private-token",)}
    cyclic_metadata: dict[str, Any] = {}
    cyclic_metadata["cycle"] = cyclic_metadata

    for metadata in (tuple_metadata, cyclic_metadata):
        with pytest.raises(ProtocolAdapterError, match="invalid_encode_metadata"):
            OpenAIResponsesAdapter().encode_stream(
                events,
                context=EncodeContext(model="model", metadata=metadata),
            )
    with pytest.raises(ProtocolAdapterError, match="too_many_events"):
        OpenAIResponsesAdapter().encode_stream(
            events,
            context=EncodeContext(
                model="model",
                limits=AdapterLimits(max_events=1),
            ),
        )
    with pytest.raises(ProtocolAdapterError, match="sse_stream_too_large"):
        OpenAIResponsesAdapter().encode_stream(
            events,
            context=EncodeContext(
                model="model",
                limits=AdapterLimits(max_sse_stream_bytes=1),
            ),
        )


def test_encode_enforces_the_sse_frame_limit():
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().encode_stream(
            agent_ir.events[:4],
            context=EncodeContext(
                model="capture-model",
                response_id="resp_test",
                limits=AdapterLimits(max_sse_frame_bytes=20),
            ),
        )

    assert _error_code(error) == "sse_frame_too_large"


def test_encode_rejects_inconsistent_usage():
    agent_ir = load_agent_ir((FIXTURE_DIR / "agent-ir.json").read_bytes())
    terminal = agent_ir.events[3].model_copy(
        update={
            "usage": Usage(
                source=UsageSource.PROVIDER,
                input_tokens=10,
                output_tokens=5,
                total_tokens=99,
            )
        }
    )

    with pytest.raises(ProtocolAdapterError) as error:
        OpenAIResponsesAdapter().encode_stream(
            [*agent_ir.events[:3], terminal],
            context=EncodeContext(model="capture-model", response_id="resp_test"),
        )

    assert _error_code(error) == "invalid_usage"
