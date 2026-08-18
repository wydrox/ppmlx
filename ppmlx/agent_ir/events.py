"""Ordered event models for Agent IR v1."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, JsonValue, StrictStr, model_validator

from ppmlx.agent_ir.base import Boolean, Error, NonEmptyString, PolicyModel, Usage
from ppmlx.agent_ir.content import ContentBlock, OutputContentBlock, RefusalBlock, _validate_parsed_arguments
from ppmlx.agent_ir.identifiers import (
    CallId,
    ChoiceIndex,
    ContentIndex,
    OutputId,
    ParallelGroupId,
    RequestId,
    Sequence,
    ToolCallIndex,
)


ContentType = Literal["text", "image", "document", "reasoning", "refusal", "extension"]


class _EventBase(PolicyModel):
    request_id: RequestId
    sequence: Sequence
    choice_index: ChoiceIndex
    output_id: OutputId


class _ContentEventBase(_EventBase):
    content_index: ContentIndex


class _ToolEventBase(_EventBase):
    tool_call_index: ToolCallIndex
    parallel_group_id: ParallelGroupId | None = None
    call_id: CallId


class _TerminalEventBase(_EventBase):
    usage: Usage | None = None


class ContentStartedEvent(_ContentEventBase):
    type: Literal["content.started"]
    content_type: ContentType


class ContentDeltaEvent(_ContentEventBase):
    type: Literal["content.delta"]
    delta: StrictStr


class ContentCompletedEvent(_ContentEventBase):
    type: Literal["content.completed"]
    content: OutputContentBlock


class ToolCallStartedEvent(_ToolEventBase):
    type: Literal["tool_call.started"]
    name: NonEmptyString


class ToolCallArgumentsDeltaEvent(_ToolEventBase):
    type: Literal["tool_call.arguments.delta"]
    delta: StrictStr


class ToolCallCompletedEvent(_ToolEventBase):
    type: Literal["tool_call.completed"]
    name: NonEmptyString
    arguments_raw: StrictStr
    arguments_json: JsonValue | None = None

    @model_validator(mode="after")
    def validate_arguments(self) -> ToolCallCompletedEvent:
        _validate_parsed_arguments(
            self.arguments_raw,
            self.arguments_json,
            present="arguments_json" in self.model_fields_set,
        )
        return self


class ToolResultEvent(_ToolEventBase):
    type: Literal["tool_result"]
    content: list[ContentBlock]
    is_error: Boolean


class ResponseRefusedEvent(_TerminalEventBase):
    type: Literal["response.refused"]
    refusal: RefusalBlock


class ResponseCompletedEvent(_TerminalEventBase):
    type: Literal["response.completed"]
    finish_reason: NonEmptyString


class ResponseCancelledEvent(_TerminalEventBase):
    type: Literal["response.cancelled"]
    reason: NonEmptyString


class ResponseFailedEvent(_TerminalEventBase):
    type: Literal["response.failed"]
    error: Error


AgentEvent = Annotated[
    Union[
        ContentStartedEvent,
        ContentDeltaEvent,
        ContentCompletedEvent,
        ToolCallStartedEvent,
        ToolCallArgumentsDeltaEvent,
        ToolCallCompletedEvent,
        ToolResultEvent,
        ResponseRefusedEvent,
        ResponseCompletedEvent,
        ResponseCancelledEvent,
        ResponseFailedEvent,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "AgentEvent",
    "ContentCompletedEvent",
    "ContentDeltaEvent",
    "ContentStartedEvent",
    "ContentType",
    "ResponseCancelledEvent",
    "ResponseCompletedEvent",
    "ResponseFailedEvent",
    "ResponseRefusedEvent",
    "ToolCallArgumentsDeltaEvent",
    "ToolCallCompletedEvent",
    "ToolCallStartedEvent",
    "ToolResultEvent",
]
