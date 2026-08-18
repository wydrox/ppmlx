"""Request models for Agent IR v1."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, JsonValue, StrictInt, StrictStr, StringConstraints, model_validator

from ppmlx.agent_ir.base import (
    Boolean,
    Extensions,
    JsonNumber,
    NonEmptyString,
    NonNegativeInt,
    PolicyModel,
    PositiveInt,
    StrictModel,
)
from ppmlx.agent_ir.content import ContentBlock
from ppmlx.agent_ir.identifiers import RequestId


InstructionRole = Literal["system", "developer", "user", "assistant", "tool", "unknown"]
MessageRole = Literal["system", "developer", "user", "assistant", "tool"]
RequestKind = Literal["initial", "continuation"]
SimpleToolChoice = Literal["auto", "none", "required"]
JsonPointer = Annotated[StrictStr, StringConstraints(pattern=r"^/")]
OpenJsonObject = dict[StrictStr, JsonValue]


class Instruction(PolicyModel):
    source_role: InstructionRole
    source_location: JsonPointer
    order: NonNegativeInt
    content: Annotated[list[ContentBlock], Field(min_length=1)]


class Message(PolicyModel):
    id: NonEmptyString | None = None
    role: MessageRole
    name: NonEmptyString | None = None
    content: list[ContentBlock]


class ToolDefinition(PolicyModel):
    name: NonEmptyString
    description: StrictStr
    input_schema: OpenJsonObject
    strict: Boolean | None = None


class NamedToolChoice(StrictModel):
    type: Literal["tool"]
    name: NonEmptyString


ToolChoice = Union[SimpleToolChoice, NamedToolChoice]


class Generation(StrictModel):
    temperature: JsonNumber | None = None
    top_p: JsonNumber | None = None
    max_output_tokens: PositiveInt | None = None
    stop: list[StrictStr] | None = None
    seed: StrictInt | None = None
    reasoning_effort: StrictStr | None = None
    extensions: Extensions = Field(default_factory=dict)


class Request(PolicyModel):
    model: NonEmptyString
    instructions: list[Instruction]
    messages: list[Message]
    tools: list[ToolDefinition]
    tool_choice: ToolChoice | None = None
    generation: Generation | None = None
    stream: Boolean | None = None
    metadata: OpenJsonObject | None = None


class RequestEnvelope(PolicyModel):
    request_id: RequestId
    kind: RequestKind
    parent_request_id: RequestId | None = None
    request: Request

    @model_validator(mode="after")
    def validate_parent_request(self) -> RequestEnvelope:
        has_parent = "parent_request_id" in self.model_fields_set
        if self.kind == "initial" and has_parent:
            raise ValueError("An initial request cannot contain parent_request_id")
        if self.kind == "continuation" and not has_parent:
            raise ValueError("A continuation request must contain parent_request_id")
        return self


__all__ = [
    "Generation",
    "Instruction",
    "InstructionRole",
    "Message",
    "MessageRole",
    "NamedToolChoice",
    "Request",
    "RequestEnvelope",
    "RequestKind",
    "SimpleToolChoice",
    "ToolChoice",
    "ToolDefinition",
]
