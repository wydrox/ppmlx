"""Top-level Agent IR v1 envelope and semantic validation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ppmlx.agent_ir.base import PolicyModel, Sensitivity, Source
from ppmlx.agent_ir.events import (
    AgentEvent,
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    ResponseCancelledEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseRefusedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolResultEvent,
)
from ppmlx.agent_ir.identifiers import ConversationId
from ppmlx.agent_ir.request import RequestEnvelope


_TERMINAL_EVENTS = (
    ResponseCompletedEvent,
    ResponseRefusedEvent,
    ResponseCancelledEvent,
    ResponseFailedEvent,
)
_OUTPUT_EVENTS = (
    ContentStartedEvent,
    ContentDeltaEvent,
    ContentCompletedEvent,
    ToolCallStartedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
)
_FROZEN_REQUEST_FIELDS = ("model", "instructions", "tools", "tool_choice", "generation", "stream")
_SENSITIVITY_LEVEL = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.CONFIDENTIAL: 2,
    Sensitivity.RESTRICTED: 3,
}


def _without_native_block_evidence(block: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(block)
    extensions = normalized.get("extensions")
    if isinstance(extensions, dict):
        semantic_extensions = {
            key: value for key, value in extensions.items() if not key.endswith(".native_block")
        }
        if semantic_extensions:
            normalized["extensions"] = semantic_extensions
        else:
            normalized.pop("extensions", None)
    if normalized.get("type") == "tool_result" and isinstance(normalized.get("content"), list):
        normalized["content"] = [
            _without_native_block_evidence(item) if isinstance(item, dict) else item
            for item in normalized["content"]
        ]
    return normalized


def _semantic_messages(messages: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        semantic_message = dict(message)
        content = semantic_message.get("content")
        if isinstance(content, list):
            semantic_message["content"] = [
                _without_native_block_evidence(block) if isinstance(block, dict) else block
                for block in content
            ]
        normalized.append(semantic_message)
    return normalized


def _request_field(envelope: RequestEnvelope, name: str) -> Any:
    return envelope.request.model_dump(mode="json", exclude_unset=True).get(name)


def _validate_sensitivity_tree(value: Any, parent: Sensitivity | None = None) -> None:
    effective_parent = parent
    sensitivity = getattr(value, "sensitivity", None) if isinstance(value, BaseModel) else None
    if isinstance(sensitivity, Sensitivity):
        if parent is not None and _SENSITIVITY_LEVEL[sensitivity] < _SENSITIVITY_LEVEL[parent]:
            raise ValueError("Nested Agent IR data cannot reduce its parent sensitivity")
        effective_parent = sensitivity
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            if field_name not in {"sensitivity", "extensions"}:
                _validate_sensitivity_tree(getattr(value, field_name), effective_parent)
    elif isinstance(value, list):
        for item in value:
            _validate_sensitivity_tree(item, effective_parent)


class AgentIR(PolicyModel):
    ir_version: Literal["agent-ir/v1"]
    conversation_id: ConversationId
    source: Source
    requests: list[RequestEnvelope] = Field(min_length=1)
    events: list[AgentEvent]

    @model_validator(mode="after")
    def validate_contract(self) -> AgentIR:
        _validate_sensitivity_tree(self)
        self._validate_requests()
        self._validate_events()
        return self

    def _validate_requests(self) -> None:
        if self.requests[0].kind != "initial":
            raise ValueError("The first Agent IR request must be initial")

        by_id: dict[str, RequestEnvelope] = {}
        for index, envelope in enumerate(self.requests):
            request_id = str(envelope.request_id)
            if request_id in by_id:
                raise ValueError("Agent IR request IDs must be unique")
            if index and envelope.kind != "continuation":
                raise ValueError("Only the first Agent IR request can be initial")
            if envelope.kind == "continuation":
                parent_id = str(envelope.parent_request_id)
                parent = by_id.get(parent_id)
                if parent is None:
                    raise ValueError("A continuation parent must be an earlier request")
                self._validate_continuation(parent, envelope)
            by_id[request_id] = envelope

    @staticmethod
    def _validate_continuation(parent: RequestEnvelope, continuation: RequestEnvelope) -> None:
        for field in _FROZEN_REQUEST_FIELDS:
            if _request_field(parent, field) != _request_field(continuation, field):
                raise ValueError(f"Continuation changed frozen request field: {field}")

        parent_messages = _semantic_messages(
            parent.request.model_dump(mode="json", exclude_unset=True)["messages"]
        )
        continuation_messages = _semantic_messages(
            continuation.request.model_dump(mode="json", exclude_unset=True)["messages"]
        )
        if continuation_messages[: len(parent_messages)] != parent_messages:
            raise ValueError("Continuation removed or reordered prior message content")

        parent_extensions = parent.request.model_dump(mode="json", exclude_unset=True).get("extensions", {})
        continuation_extensions = continuation.request.model_dump(mode="json", exclude_unset=True).get(
            "extensions", {}
        )
        parent_semantic = {key: value for key, value in parent_extensions.items() if not key.endswith(".native_request")}
        continuation_semantic = {
            key: value for key, value in continuation_extensions.items() if not key.endswith(".native_request")
        }
        if continuation_semantic != parent_semantic:
            raise ValueError("Continuation changed a required request extension")
        parent_native_keys = {key for key in parent_extensions if key.endswith(".native_request")}
        continuation_native_keys = {key for key in continuation_extensions if key.endswith(".native_request")}
        if continuation_native_keys != parent_native_keys:
            raise ValueError("Continuation changed its native request evidence keys")

        parent_metadata = _request_field(parent, "metadata") or {}
        continuation_metadata = _request_field(continuation, "metadata") or {}
        for key, parent_value in parent_metadata.items():
            if key not in continuation_metadata:
                raise ValueError("Continuation removed metadata")
            if key != "client_metadata" and continuation_metadata[key] != parent_value:
                raise ValueError("Continuation changed routing or policy metadata")
        for key in continuation_metadata.keys() - parent_metadata.keys():
            if key != "client_metadata" and not key.startswith(("diagnostic.", "diagnostic_")):
                raise ValueError("Continuation added unclassified metadata")

    def _validate_events(self) -> None:
        requests_by_id = {str(item.request_id): item for item in self.requests}
        request_ids = set(requests_by_id)
        last_sequence: dict[str, int] = {}
        output_events: dict[tuple[str, int, str], list[AgentEvent]] = defaultdict(list)
        content_states: dict[tuple[str, int, str, int], dict[str, Any]] = {}
        call_states: dict[tuple[str, int, str, int], dict[str, Any]] = {}
        calls_by_id: dict[str, dict[str, Any]] = {}

        for position, event in enumerate(self.events):
            request_id = str(event.request_id)
            if request_id not in request_ids:
                raise ValueError("An event refers to an unknown request")
            previous = last_sequence.get(request_id)
            if previous is not None and event.sequence <= previous:
                raise ValueError("Event sequences must increase for each request")
            last_sequence[request_id] = event.sequence

            output_key = (request_id, event.choice_index, str(event.output_id))
            if isinstance(event, _OUTPUT_EVENTS + _TERMINAL_EVENTS):
                output_events[output_key].append(event)

            if isinstance(event, (ContentStartedEvent, ContentDeltaEvent, ContentCompletedEvent)):
                self._validate_content_event(event, content_states)
            elif isinstance(
                event,
                (ToolCallStartedEvent, ToolCallArgumentsDeltaEvent, ToolCallCompletedEvent),
            ):
                self._validate_tool_call_event(event, position, call_states, calls_by_id)
            elif isinstance(event, ToolResultEvent):
                self._validate_tool_result(event, position, calls_by_id, requests_by_id)

        self._validate_terminals(output_events)
        self._validate_completed_lifecycles(output_events, content_states, call_states)

    @staticmethod
    def _validate_content_event(
        event: ContentStartedEvent | ContentDeltaEvent | ContentCompletedEvent,
        states: dict[tuple[str, int, str, int], dict[str, Any]],
    ) -> None:
        key = (str(event.request_id), event.choice_index, str(event.output_id), event.content_index)
        state = states.setdefault(key, {"started": False, "completed": False, "content_type": None})
        if isinstance(event, ContentStartedEvent):
            if state["started"]:
                raise ValueError("A content output cannot start more than once")
            state["started"] = True
            state["content_type"] = event.content_type
            return
        if not state["started"]:
            raise ValueError("A content delta or completion requires content.started")
        if state["completed"]:
            raise ValueError("A content event cannot follow content.completed")
        if isinstance(event, ContentCompletedEvent):
            if event.content.type != state["content_type"]:
                raise ValueError("Completed content type does not match content.started")
            state["completed"] = True

    @staticmethod
    def _validate_tool_call_event(
        event: ToolCallStartedEvent | ToolCallArgumentsDeltaEvent | ToolCallCompletedEvent,
        position: int,
        states: dict[tuple[str, int, str, int], dict[str, Any]],
        calls_by_id: dict[str, dict[str, Any]],
    ) -> None:
        key = (str(event.request_id), event.choice_index, str(event.output_id), event.tool_call_index)
        state = states.get(key)
        if isinstance(event, ToolCallStartedEvent):
            if state is not None:
                raise ValueError("A tool-call identity cannot start more than once")
            call_id = str(event.call_id)
            if call_id in calls_by_id:
                raise ValueError("Tool-call IDs must be unique in one conversation")
            state = {
                "call_id": call_id,
                "request_id": str(event.request_id),
                "name": event.name,
                "parallel_group_id": event.parallel_group_id,
                "deltas": [],
                "completed": False,
                "position": position,
                "choice_index": event.choice_index,
                "tool_call_index": event.tool_call_index,
            }
            states[key] = state
            calls_by_id[call_id] = state
            return

        if state is None:
            raise ValueError("A tool-call delta or completion requires tool_call.started")
        if state["completed"]:
            raise ValueError("A tool-call event cannot follow tool_call.completed")
        if str(event.call_id) != state["call_id"]:
            raise ValueError("A tool-call lifecycle must keep one call_id")
        if event.parallel_group_id != state["parallel_group_id"]:
            raise ValueError("A tool-call lifecycle must keep one parallel_group_id")
        if isinstance(event, ToolCallArgumentsDeltaEvent):
            state["deltas"].append(event.delta)
            return
        if event.name != state["name"]:
            raise ValueError("A tool-call lifecycle must keep one tool name")
        if state["deltas"] and "".join(state["deltas"]) != event.arguments_raw:
            raise ValueError("Tool argument deltas do not assemble to arguments_raw")
        state["completed"] = True
        state["position"] = position

    @staticmethod
    def _validate_tool_result(
        event: ToolResultEvent,
        position: int,
        calls_by_id: dict[str, dict[str, Any]],
        requests_by_id: dict[str, RequestEnvelope],
    ) -> None:
        state = calls_by_id.get(str(event.call_id))
        if state is None or not state["completed"] or state["position"] >= position:
            raise ValueError("A tool result must refer to an earlier completed tool call")
        result_request = requests_by_id[str(event.request_id)]
        if result_request.kind != "continuation" or str(result_request.parent_request_id) != state["request_id"]:
            raise ValueError("A tool result must use a continuation linked to its tool-call request")
        if event.choice_index != state["choice_index"] or event.tool_call_index != state["tool_call_index"]:
            raise ValueError("A tool result must keep its call indexes")
        if event.parallel_group_id != state["parallel_group_id"]:
            raise ValueError("A tool result must keep its parallel_group_id")
        if state.get("result_received"):
            raise ValueError("A tool call cannot receive more than one result")
        state["result_received"] = True

    @staticmethod
    def _validate_terminals(output_events: dict[tuple[str, int, str], list[AgentEvent]]) -> None:
        for events in output_events.values():
            terminals = [event for event in events if isinstance(event, _TERMINAL_EVENTS)]
            if len(terminals) != 1:
                raise ValueError("Each output choice must contain exactly one terminal event")
            if events[-1] is not terminals[0]:
                raise ValueError("A terminal event must be the last event for its output choice")

    @staticmethod
    def _validate_completed_lifecycles(
        output_events: dict[tuple[str, int, str], list[AgentEvent]],
        content_states: dict[tuple[str, int, str, int], dict[str, Any]],
        call_states: dict[tuple[str, int, str, int], dict[str, Any]],
    ) -> None:
        completed_outputs = {
            key
            for key, events in output_events.items()
            if any(isinstance(event, (ResponseCompletedEvent, ResponseRefusedEvent)) for event in events)
        }
        for key, state in content_states.items():
            if key[:3] in completed_outputs and not state["completed"]:
                raise ValueError("A completed response cannot contain unfinished content")
        for key, state in call_states.items():
            if key[:3] in completed_outputs and not state["completed"]:
                raise ValueError("A completed response cannot contain an unfinished tool call")


__all__ = ["AgentIR"]
