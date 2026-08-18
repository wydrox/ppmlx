"""Lossless parsed-JSON codec for Agent IR v1."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from ppmlx.agent_ir.envelope import AgentIR


class AgentIRValidationError(ValueError):
    """The input does not conform to the Agent IR v1 contract."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Agent IR JSON cannot contain {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Agent IR JSON contains a duplicate key: {key}")
        value[key] = item
    return value


def _decode_json(value: str | bytes | bytearray) -> Any:
    text = bytes(value).decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
    return json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def load_agent_ir(value: str | bytes | bytearray | Mapping[str, Any]) -> AgentIR:
    if not isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("Agent IR input must be parsed JSON or JSON text")
    try:
        if isinstance(value, (str, bytes, bytearray)):
            return AgentIR.model_validate(_decode_json(value))
        return AgentIR.model_validate(dict(value))
    except (ValidationError, ValueError, TypeError):
        pass
    raise AgentIRValidationError("Agent IR validation failed")


def dump_agent_ir(value: AgentIR) -> dict[str, JsonValue]:
    if not isinstance(value, AgentIR):
        raise TypeError("dump_agent_ir requires an AgentIR object")
    dumped = value.model_dump(mode="json", exclude_unset=True)
    return cast(dict[str, JsonValue], dumped)


def encode_agent_ir(value: AgentIR, *, indent: int | None = None) -> str:
    return json.dumps(dump_agent_ir(value), ensure_ascii=False, allow_nan=False, indent=indent)


__all__ = ["AgentIRValidationError", "dump_agent_ir", "encode_agent_ir", "load_agent_ir"]
