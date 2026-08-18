"""Manifest-driven replay tests for the public protocol facades."""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from ppmlx.agent_ir import (
    AgentIR,
    Origin,
    Provenance,
    Sensitivity,
    ToolCallStartedEvent,
    ToolResultEvent,
    Trust,
    load_agent_ir,
)
from ppmlx.protocols import (
    CallReference,
    DecodeContext,
    EncodeContext,
    NormalizationPolicy,
    ProtocolAdapter,
    anthropic_messages_adapter,
    openai_chat_adapter,
    openai_responses_adapter,
)
from ppmlx.protocols.sse import parse_sse


CONTRACT_ROOT = Path(__file__).parent / "fixtures" / "contracts"
MANIFEST = json.loads((CONTRACT_ROOT / "manifest.json").read_text())
ADAPTERS: dict[str, ProtocolAdapter] = {
    "anthropic-messages": anthropic_messages_adapter,
    "openai-chat": openai_chat_adapter,
    "openai-responses": openai_responses_adapter,
}
FIXTURE_POLICY = NormalizationPolicy(
    sensitivity=Sensitivity.PUBLIC,
    provenance=Provenance(origin=Origin.HARNESS, trust=Trust.UNTRUSTED),
    include_native_evidence=True,
)


def _response_id(stream: str, *, protocol: str) -> str:
    for frame in parse_sse(stream, protocol=protocol):
        data = frame.data
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("id"), str):
            return data["id"]
        for key in ("response", "message"):
            nested = data.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]
    raise AssertionError("The fixture stream has no response identifier")


def _call_references(expected: AgentIR) -> dict[str, CallReference]:
    references: dict[str, CallReference] = {}
    for event in expected.events:
        if isinstance(event, ToolCallStartedEvent):
            call_id = str(event.call_id)
            references[call_id] = CallReference(
                call_id=call_id,
                name=event.name,
                choice_index=event.choice_index,
                output_id=str(event.output_id),
                tool_call_index=event.tool_call_index,
                parallel_group_id=(
                    str(event.parallel_group_id) if event.parallel_group_id is not None else None
                ),
            )
    return references


@pytest.mark.parametrize("fixture", MANIFEST["fixtures"], ids=lambda item: item["id"])
def test_public_adapter_replays_an_approved_harness_contract(
    fixture: dict[str, Any],
) -> None:
    directory = CONTRACT_ROOT / fixture["directory"]
    protocol = fixture["protocol"]
    adapter = ADAPTERS[protocol]
    expected = load_agent_ir((directory / "agent-ir.json").read_bytes())
    initial_request, continuation_request = expected.requests
    references = _call_references(expected)
    expected_results = [event for event in expected.events if isinstance(event, ToolResultEvent)]
    assert expected_results
    result_output_ids = {
        str(event.call_id): str(event.output_id) for event in expected_results
    }

    initial = adapter.decode_request(
        (directory / "initial-request.json").read_bytes(),
        context=DecodeContext(
            request_id=str(initial_request.request_id),
            kind="initial",
            policy=FIXTURE_POLICY,
        ),
    )
    continuation = adapter.decode_request(
        (directory / "tool-result-request.json").read_bytes(),
        context=DecodeContext(
            request_id=str(continuation_request.request_id),
            kind="continuation",
            parent_request_id=str(continuation_request.parent_request_id),
            sequence_start=min(event.sequence for event in expected_results),
            result_output_ids=MappingProxyType(result_output_ids),
            prior_calls=MappingProxyType(references),
            policy=FIXTURE_POLICY,
        ),
    )

    assert initial.request == initial_request
    assert initial.tool_results == ()
    assert continuation.request == continuation_request
    assert continuation.tool_results == tuple(expected_results)
    assert continuation.calls == tuple(references.values())

    rebuilt_events = [
        continuation.tool_results[expected_results.index(event)]
        if isinstance(event, ToolResultEvent)
        else event
        for event in expected.events
    ]
    rebuilt_data = expected.model_dump(mode="json", exclude_unset=True)
    rebuilt_data["requests"] = [
        initial.request.model_dump(mode="json", exclude_unset=True),
        continuation.request.model_dump(mode="json", exclude_unset=True),
    ]
    rebuilt_data["events"] = [
        event.model_dump(mode="json", exclude_unset=True) for event in rebuilt_events
    ]
    rebuilt = AgentIR.model_validate(rebuilt_data)
    assert rebuilt.model_dump(mode="json", exclude_unset=True) == expected.model_dump(
        mode="json", exclude_unset=True
    )

    model = str(initial.request.request.model)
    for request_id, fixture_name in (
        (str(initial_request.request_id), "tool-call-stream.sse"),
        (str(continuation_request.request_id), "final-response.sse"),
    ):
        native_stream = (directory / fixture_name).read_text()
        output_events = [
            event
            for event in expected.events
            if str(event.request_id) == request_id and not isinstance(event, ToolResultEvent)
        ]
        encoded = adapter.encode_stream(
            output_events,
            context=EncodeContext(
                model=model,
                created_at=0,
                response_id=_response_id(native_stream, protocol=protocol),
            ),
        )
        assert parse_sse(encoded, protocol=protocol) == parse_sse(
            native_stream,
            protocol=protocol,
        )
