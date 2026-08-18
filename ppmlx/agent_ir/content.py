"""Content block models for Agent IR v1."""
from __future__ import annotations

import json
from typing import Annotated, Literal, Never, Union

from pydantic import AnyUrl, Field, JsonValue, StrictStr, TypeAdapter, model_validator

from ppmlx.agent_ir.base import (
    Boolean,
    ExtensionName,
    NonEmptyString,
    PolicyModel,
    Provenance,
    Sensitivity,
    StrictModel,
    default_provenance,
)
from ppmlx.agent_ir.identifiers import CallId


_URI_ADAPTER = TypeAdapter(AnyUrl)


def _validate_uri(value: str) -> str:
    """Validate a URI without changing its source spelling."""
    _URI_ADAPTER.validate_python(value)
    return value


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"Invalid JSON constant: {value}")


def _validate_parsed_arguments(arguments_raw: str, arguments_json: JsonValue, *, present: bool) -> None:
    if not present:
        return
    try:
        parsed = json.loads(
            arguments_raw,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("arguments_json requires valid arguments_raw JSON") from exc
    normalized_parsed = json.dumps(parsed, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    normalized_supplied = json.dumps(
        arguments_json,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if normalized_parsed != normalized_supplied:
        raise ValueError("arguments_json does not match arguments_raw")


class TextBlock(PolicyModel):
    type: Literal["text"]
    text: StrictStr


class ImageBlock(PolicyModel):
    type: Literal["image"]
    media_type: NonEmptyString
    data: StrictStr | None = None
    url: StrictStr | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> ImageBlock:
        sources = self.model_fields_set & {"data", "url"}
        if len(sources) != 1:
            raise ValueError("An image block must contain exactly one of data or url")
        if "url" in sources:
            _validate_uri(self.url or "")
        return self


class DocumentBlock(PolicyModel):
    type: Literal["document"]
    media_type: NonEmptyString
    text: StrictStr | None = None
    data: StrictStr | None = None
    url: StrictStr | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> DocumentBlock:
        sources = self.model_fields_set & {"text", "data", "url"}
        if len(sources) != 1:
            raise ValueError("A document block must contain exactly one of text, data, or url")
        if "url" in sources:
            _validate_uri(self.url or "")
        return self


class ReasoningBlock(PolicyModel):
    type: Literal["reasoning"]
    data: JsonValue


class ToolCallBlock(PolicyModel):
    type: Literal["tool_call"]
    call_id: CallId
    name: NonEmptyString
    arguments_raw: StrictStr
    arguments_json: JsonValue | None = None

    @model_validator(mode="after")
    def validate_arguments(self) -> ToolCallBlock:
        _validate_parsed_arguments(
            self.arguments_raw,
            self.arguments_json,
            present="arguments_json" in self.model_fields_set,
        )
        return self


class ToolResultBlock(PolicyModel):
    type: Literal["tool_result"]
    call_id: CallId
    content: list[ContentBlock]
    is_error: Boolean


class RefusalBlock(PolicyModel):
    type: Literal["refusal"]
    text: StrictStr


class ExtensionBlock(StrictModel):
    type: Literal["extension"]
    namespace: ExtensionName
    data: JsonValue
    required: Boolean
    sensitivity: Sensitivity = Sensitivity.RESTRICTED
    provenance: Provenance = Field(default_factory=default_provenance)


ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        DocumentBlock,
        ReasoningBlock,
        ToolCallBlock,
        ToolResultBlock,
        RefusalBlock,
        ExtensionBlock,
    ],
    Field(discriminator="type"),
]

OutputContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        DocumentBlock,
        ReasoningBlock,
        RefusalBlock,
        ExtensionBlock,
    ],
    Field(discriminator="type"),
]


ToolResultBlock.model_rebuild(_types_namespace={"ContentBlock": ContentBlock})


__all__ = [
    "ContentBlock",
    "DocumentBlock",
    "ExtensionBlock",
    "ImageBlock",
    "OutputContentBlock",
    "ReasoningBlock",
    "RefusalBlock",
    "TextBlock",
    "ToolCallBlock",
    "ToolResultBlock",
]
