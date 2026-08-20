"""Pure protocol-adapter contracts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import wraps
from types import MappingProxyType
from typing import Callable, Mapping, ParamSpec, Protocol, Sequence, TypeVar

from pydantic import JsonValue

from ppmlx.agent_ir import (
    AgentEvent,
    Origin,
    Provenance,
    RequestEnvelope,
    Sensitivity,
    ToolResultEvent,
    Trust,
)


@dataclass(frozen=True, slots=True)
class AdapterLimits:
    max_request_bytes: int = 2 * 1024 * 1024
    max_sse_frame_bytes: int = 1024 * 1024
    max_sse_stream_bytes: int = 4 * 1024 * 1024
    max_events: int = 100_000
    max_json_depth: int = 64
    max_json_nodes: int = 100_000
    max_string_bytes: int = 1024 * 1024
    max_blocks: int = 4_096
    max_tools: int = 512
    max_arguments_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.max_request_bytes,
            self.max_sse_frame_bytes,
            self.max_sse_stream_bytes,
            self.max_events,
            self.max_json_depth,
            self.max_json_nodes,
            self.max_string_bytes,
            self.max_blocks,
            self.max_tools,
            self.max_arguments_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Adapter limits must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    sensitivity: Sensitivity = Sensitivity.RESTRICTED
    provenance: Provenance = field(
        default_factory=lambda: Provenance(origin=Origin.UNKNOWN, trust=Trust.UNTRUSTED)
    )
    include_native_evidence: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sensitivity, Sensitivity)
            or not isinstance(self.provenance, Provenance)
            or type(self.include_native_evidence) is not bool
        ):
            raise ValueError("Normalization policy values are invalid")


@dataclass(frozen=True, slots=True)
class CallReference:
    call_id: str
    name: str
    choice_index: int
    output_id: str
    tool_call_index: int
    parallel_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class DecodeContext:
    request_id: str
    kind: str
    parent_request_id: str | None = None
    sequence_start: int = 0
    result_output_ids: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    prior_calls: Mapping[str, CallReference] = field(
        default_factory=lambda: MappingProxyType({})
    )
    policy: NormalizationPolicy = field(default_factory=NormalizationPolicy)
    limits: AdapterLimits = field(default_factory=AdapterLimits)


@dataclass(frozen=True, slots=True)
class EncodeContext:
    model: str
    created_at: int = 0
    response_id: str | None = None
    parallel_tool_calls: bool = True
    include_usage: bool = True
    metadata: Mapping[str, JsonValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    limits: AdapterLimits = field(default_factory=AdapterLimits)


@dataclass(frozen=True, slots=True)
class DecodedRequest:
    request: RequestEnvelope
    tool_results: tuple[ToolResultEvent, ...] = ()
    calls: tuple[CallReference, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtocolCapabilities:
    request_features: frozenset[str]
    response_features: frozenset[str]
    verified_harnesses: tuple[str, ...]

    def __post_init__(self) -> None:
        features = self.request_features | self.response_features
        if not features <= _PROTOCOL_FEATURES:
            raise ValueError("Protocol capabilities use an unknown feature")
        if (
            type(self.verified_harnesses) is not tuple
            or not self.verified_harnesses
            or len(set(self.verified_harnesses)) != len(self.verified_harnesses)
            or any(type(item) is not str or not item for item in self.verified_harnesses)
        ):
            raise ValueError("Verified harness identifiers are invalid")


_PROTOCOL_FEATURES = frozenset(
    {
        "document",
        "generation",
        "image",
        "instructions",
        "metadata",
        "stream",
        "text",
        "tool_calls",
        "tool_choice",
        "tool_results",
        "tools",
        "usage",
    }
)


class ProtocolAdapterError(ValueError):
    """A safe protocol error that does not contain native request data."""

    def __init__(self, *, protocol: str, code: str, field: str | None = None) -> None:
        self.protocol = protocol
        self.code = code
        self.field = _safe_field_root(field)
        super().__init__(f"{protocol} adapter error {code}")


_SAFE_FIELD_ROOTS = {
    "client_metadata",
    "context_management",
    "generation",
    "include",
    "input",
    "instructions",
    "max_output_tokens",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "output_config",
    "reasoning",
    "service_tier",
    "stream",
    "system",
    "tool_choice",
    "tools",
}


def _safe_field_root(field: str | None) -> str | None:
    if field == "/":
        return field
    if field is None:
        return None
    root = re.split(r"[./\[]", field.lstrip("/"), maxsplit=1)[0]
    return f"/{root}" if root in _SAFE_FIELD_ROOTS else None


_P = ParamSpec("_P")
_R = TypeVar("_R")


def safe_adapter_boundary(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Remove native validation exceptions from a public adapter error."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        details: tuple[str, str, str | None] | None = None
        try:
            return function(*args, **kwargs)
        except ProtocolAdapterError as error:
            details = (error.protocol, error.code, error.field)
        except Exception:
            owner = args[0] if args else None
            protocol = getattr(owner, "protocol", "protocol")
            details = (
                protocol if type(protocol) is str else "protocol",
                "invalid_adapter_input",
                None,
            )
        protocol, code, field = details
        raise ProtocolAdapterError(protocol=protocol, code=code, field=field)

    return wrapped


class ProtocolAdapter(Protocol):
    protocol: str
    capabilities: ProtocolCapabilities

    def decode_request(
        self,
        value: str | bytes | bytearray | Mapping[str, object],
        *,
        context: DecodeContext,
    ) -> DecodedRequest: ...

    def encode_stream(
        self,
        events: Sequence[AgentEvent],
        *,
        context: EncodeContext,
    ) -> str: ...


__all__ = [
    "AdapterLimits",
    "CallReference",
    "DecodeContext",
    "DecodedRequest",
    "EncodeContext",
    "NormalizationPolicy",
    "ProtocolAdapter",
    "ProtocolAdapterError",
    "ProtocolCapabilities",
    "safe_adapter_boundary",
]
