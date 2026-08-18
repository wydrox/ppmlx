"""Shared types for Agent IR v1."""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Base model that rejects fields outside the Agent IR contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if isinstance(value, dict):
            null_fields = sorted(key for key, item in value.items() if item is None and key in cls.model_fields)
            if null_fields:
                joined = ", ".join(null_fields)
                raise ValueError(f"Agent IR fields cannot be null: {joined}")
        return value


class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Origin(str, Enum):
    HARNESS = "harness"
    PROVIDER = "provider"
    TOOL = "tool"
    MEMORY = "memory"
    PPMLX = "ppmlx"
    UNKNOWN = "unknown"


class Trust(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"


class Protocol(str, Enum):
    ANTHROPIC_MESSAGES = "anthropic-messages"
    OPENAI_RESPONSES = "openai-responses"
    OPENAI_CHAT = "openai-chat"


class UsageSource(str, Enum):
    PROVIDER = "provider"
    PPMLX_ESTIMATE = "ppmlx_estimate"


NonEmptyString = Annotated[StrictStr, StringConstraints(min_length=1)]
ExtensionName = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)+$"),
]
HarnessVersion = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$"),
]
Extensions = dict[ExtensionName, JsonValue]
JsonObject = dict[StrictStr, JsonValue]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
JsonNumber = StrictInt | StrictFloat
Boolean = StrictBool


class Provenance(StrictModel):
    origin: Origin
    trust: Trust
    origin_id: NonEmptyString | None = None

    @field_validator("origin", "trust", mode="before")
    @classmethod
    def reject_coerced_enum(cls, value: object) -> object:
        if type(value) is not str and not isinstance(value, (Origin, Trust)):
            raise ValueError("Provenance enum values must be strings")
        return value


def default_provenance() -> Provenance:
    return Provenance(origin=Origin.UNKNOWN, trust=Trust.UNTRUSTED)


class PolicyModel(StrictModel):
    sensitivity: Sensitivity = Sensitivity.RESTRICTED
    provenance: Provenance = Field(default_factory=default_provenance)
    extensions: Extensions = Field(default_factory=dict)

    @field_validator("sensitivity", mode="before")
    @classmethod
    def reject_coerced_sensitivity(cls, value: object) -> object:
        if type(value) is not str and not isinstance(value, Sensitivity):
            raise ValueError("Sensitivity must be a string")
        return value


class Source(StrictModel):
    harness: NonEmptyString
    harness_version: HarnessVersion
    protocol: Protocol
    protocol_version: NonEmptyString

    @field_validator("protocol", mode="before")
    @classmethod
    def reject_coerced_protocol(cls, value: object) -> object:
        if type(value) is not str and not isinstance(value, Protocol):
            raise ValueError("Protocol must be a string")
        return value


class Usage(StrictModel):
    source: UsageSource
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None

    @field_validator("source", mode="before")
    @classmethod
    def reject_coerced_source(cls, value: object) -> object:
        if type(value) is not str and not isinstance(value, UsageSource):
            raise ValueError("Usage source must be a string")
        return value


class Error(StrictModel):
    code: NonEmptyString
    category: NonEmptyString
    message: StrictStr
    retryable: Boolean
    extensions: Extensions = Field(default_factory=dict)
