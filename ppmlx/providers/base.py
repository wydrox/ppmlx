"""Protocol-neutral provider contracts for Agent IR model invocation."""
from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ppmlx.agent_ir import AgentEvent, RequestEnvelope


_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MODEL_ID_RE = re.compile(r"^\S{1,512}$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class ProviderDataPath(str, Enum):
    """Where request content leaves the gateway process."""

    LOCAL = "local"
    REMOTE = "remote"


class ProviderCredentialType(str, Enum):
    """Credential classes declared by a provider adapter."""

    NONE = "none"
    API_KEY = "api_key"
    OAUTH_SESSION = "oauth_session"


class ProviderStreamingMode(str, Enum):
    """How provider events become available to the coordinator."""

    NONE = "none"
    BUFFERED = "buffered"
    NATIVE = "native"


class ProviderHealthStatus(str, Enum):
    """Provider availability for deterministic routing decisions."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ProviderToolSupportStatus(str, Enum):
    """Evidence status for model tool-use quality."""

    NOT_EVALUATED = "not_evaluated"
    STABLE = "stable"
    PREVIEW = "preview"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


def _valid_provider_id(value: object) -> bool:
    return type(value) is str and _PROVIDER_ID_RE.fullmatch(value) is not None


def _valid_model_id(value: object) -> bool:
    return type(value) is str and _MODEL_ID_RE.fullmatch(value) is not None


def _valid_code(value: object) -> bool:
    return type(value) is str and _SAFE_CODE_RE.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Route-relevant capabilities for one provider model."""

    text: bool
    images: bool
    tools: bool
    parallel_tool_calls: bool
    reasoning: bool
    streaming: ProviderStreamingMode
    context_window: int | None
    data_path: ProviderDataPath
    credential_types: tuple[ProviderCredentialType, ...]
    tool_support_status: ProviderToolSupportStatus = (
        ProviderToolSupportStatus.NOT_EVALUATED
    )

    def __post_init__(self) -> None:
        flags = (
            self.text,
            self.images,
            self.tools,
            self.parallel_tool_calls,
            self.reasoning,
        )
        if any(type(value) is not bool for value in flags):
            raise ValueError("Provider capability flags are invalid")
        if not isinstance(self.streaming, ProviderStreamingMode):
            raise ValueError("Provider streaming mode is invalid")
        if self.context_window is not None and (
            type(self.context_window) is not int or self.context_window < 1
        ):
            raise ValueError("Provider context window is invalid")
        if not isinstance(self.data_path, ProviderDataPath):
            raise ValueError("Provider data path is invalid")
        if (
            type(self.credential_types) is not tuple
            or not self.credential_types
            or any(
                not isinstance(value, ProviderCredentialType)
                for value in self.credential_types
            )
            or len(set(self.credential_types)) != len(self.credential_types)
        ):
            raise ValueError("Provider credential types are invalid")
        if not isinstance(self.tool_support_status, ProviderToolSupportStatus):
            raise ValueError("Provider tool support status is invalid")
        if self.parallel_tool_calls and not self.tools:
            raise ValueError("Parallel tool calls require tool support")
        if not self.tools and self.tool_support_status is not ProviderToolSupportStatus.DISABLED:
            object.__setattr__(
                self,
                "tool_support_status",
                ProviderToolSupportStatus.DISABLED,
            )


@dataclass(frozen=True, slots=True)
class ProviderModel:
    """One provider-owned model identifier and its route capabilities."""

    provider_id: str
    model_id: str
    capabilities: ProviderCapabilities
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _valid_provider_id(self.provider_id):
            raise ValueError("Provider identifier is invalid")
        if not _valid_model_id(self.model_id):
            raise ValueError("Provider model identifier is invalid")
        if not isinstance(self.capabilities, ProviderCapabilities):
            raise ValueError("Provider model capabilities are invalid")
        if (
            type(self.aliases) is not tuple
            or any(not _valid_model_id(value) for value in self.aliases)
            or len(set(self.aliases)) != len(self.aliases)
            or self.model_id in self.aliases
        ):
            raise ValueError("Provider model aliases are invalid")


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Sanitized provider health used by the routing policy."""

    provider_id: str
    status: ProviderHealthStatus
    code: str
    model_count: int = 0

    def __post_init__(self) -> None:
        if not _valid_provider_id(self.provider_id):
            raise ValueError("Provider identifier is invalid")
        if not isinstance(self.status, ProviderHealthStatus):
            raise ValueError("Provider health status is invalid")
        if not _valid_code(self.code):
            raise ValueError("Provider health code is invalid")
        if type(self.model_count) is not int or self.model_count < 0:
            raise ValueError("Provider health model count is invalid")


@dataclass(frozen=True, slots=True)
class ProviderInvocation:
    """One Agent IR request bound to a provider-owned model identifier."""

    request: RequestEnvelope
    model_id: str
    output_id: str | None = None
    sequence_start: int = 0
    max_tokens_cap: int = 32_768
    enable_reasoning: bool = False
    parallel_tool_calls: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestEnvelope):
            raise ValueError("Provider request envelope is invalid")
        if not _valid_model_id(self.model_id):
            raise ValueError("Provider model identifier is invalid")
        if self.output_id is not None and not _valid_model_id(self.output_id):
            raise ValueError("Provider output identifier is invalid")
        if type(self.sequence_start) is not int or self.sequence_start < 0:
            raise ValueError("Provider sequence start is invalid")
        if type(self.max_tokens_cap) is not int or self.max_tokens_cap < 1:
            raise ValueError("Provider token cap is invalid")
        if type(self.enable_reasoning) is not bool:
            raise ValueError("Provider reasoning flag is invalid")
        if type(self.parallel_tool_calls) is not bool:
            raise ValueError("Provider parallel tool flag is invalid")


@dataclass(frozen=True, slots=True)
class ProviderCallReference:
    """Stable provider call identity needed for tool-result correlation."""

    call_id: str
    name: str
    choice_index: int
    output_id: str
    tool_call_index: int
    parallel_group_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.call_id, self.name, self.output_id):
            if not _valid_model_id(value):
                raise ValueError("Provider call reference is invalid")
        for value in (self.choice_index, self.tool_call_index):
            if type(value) is not int or value < 0:
                raise ValueError("Provider call index is invalid")
        if self.parallel_group_id is not None and not _valid_model_id(
            self.parallel_group_id
        ):
            raise ValueError("Provider parallel group is invalid")


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Canonical provider output with no external protocol objects."""

    provider_id: str
    model_id: str
    events: tuple[AgentEvent, ...]
    calls: tuple[ProviderCallReference, ...] = ()
    source_call_ids: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    streaming: ProviderStreamingMode = ProviderStreamingMode.BUFFERED

    def __post_init__(self) -> None:
        if not _valid_provider_id(self.provider_id):
            raise ValueError("Provider identifier is invalid")
        if not _valid_model_id(self.model_id):
            raise ValueError("Provider model identifier is invalid")
        if type(self.events) is not tuple:
            raise ValueError("Provider events are invalid")
        if type(self.calls) is not tuple or any(
            not isinstance(value, ProviderCallReference) for value in self.calls
        ):
            raise ValueError("Provider calls are invalid")
        if not isinstance(self.source_call_ids, Mapping) or any(
            not _valid_model_id(key) or not _valid_model_id(value)
            for key, value in self.source_call_ids.items()
        ):
            raise ValueError("Provider source call identifiers are invalid")
        if not isinstance(self.streaming, ProviderStreamingMode):
            raise ValueError("Provider streaming mode is invalid")
        object.__setattr__(
            self,
            "source_call_ids",
            MappingProxyType(dict(self.source_call_ids)),
        )


class ProviderError(ValueError):
    """A typed safe provider error that contains no request or secret data."""

    def __init__(self, *, provider_id: str, code: str) -> None:
        self.provider_id = provider_id if _valid_provider_id(provider_id) else "provider"
        self.code = code if _valid_code(code) else "provider_failed"
        super().__init__(f"{self.provider_id} provider error {self.code}")


@runtime_checkable
class Provider(Protocol):
    """Common interface for local and remote Agent IR model providers."""

    @property
    def provider_id(self) -> str: ...

    def list_models(self) -> tuple[ProviderModel, ...]: ...

    def capabilities(self, model_id: str) -> ProviderCapabilities: ...

    def invoke(self, invocation: ProviderInvocation) -> ProviderResult: ...

    def stream(self, invocation: ProviderInvocation) -> Iterator[AgentEvent]: ...

    def health(self) -> ProviderHealth: ...


__all__ = [
    "Provider",
    "ProviderCallReference",
    "ProviderCapabilities",
    "ProviderCredentialType",
    "ProviderDataPath",
    "ProviderError",
    "ProviderHealth",
    "ProviderHealthStatus",
    "ProviderInvocation",
    "ProviderModel",
    "ProviderResult",
    "ProviderStreamingMode",
    "ProviderToolSupportStatus",
]
