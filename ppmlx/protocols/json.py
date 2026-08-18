"""Strict native JSON parsing for protocol adapters."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, Never

from ppmlx.protocols.base import AdapterLimits, ProtocolAdapterError


_CREDENTIAL_KEYS = {
    "accesstoken",
    "anthropicapikey",
    "apikey",
    "apisecret",
    "authtoken",
    "authorization",
    "bearertoken",
    "clientsecret",
    "clienttoken",
    "cookie",
    "credential",
    "credentials",
    "idtoken",
    "oauthtoken",
    "openaiapikey",
    "password",
    "privatekey",
    "proxyauthorization",
    "refreshtoken",
    "secret",
    "secretkey",
    "sessiontoken",
    "setcookie",
    "token",
    "xapikey",
}
_CREDENTIAL_KEY_SUFFIXES = (
    "accesskey",
    "apikey",
    "cookie",
    "credential",
    "credentials",
    "password",
    "privatekey",
    "secret",
    "token",
)
_CREDENTIAL_VALUE = re.compile(
    r"(?:^|\s)(?:Bearer\s+\S+|sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|xox[baprs]-\S+)",
    re.IGNORECASE,
)


def _reject_constant(value: str) -> Never:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _validate_json_value(
    value: Any,
    *,
    limits: AdapterLimits,
    depth: int = 0,
    nodes: list[int] | None = None,
) -> Any:
    counter = nodes if nodes is not None else [0]
    counter[0] += 1
    if counter[0] > limits.max_json_nodes or depth > limits.max_json_depth:
        raise ValueError("Native JSON exceeds its structural limit")
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Native JSON contains a non-finite number")
        return value
    if type(value) is str:
        if len(value.encode("utf-8")) > limits.max_string_bytes:
            raise ValueError("Native JSON string exceeds its size limit")
        return value
    if type(value) is list:
        return [
            _validate_json_value(item, limits=limits, depth=depth + 1, nodes=counter)
            for item in value
        ]
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("Native JSON object keys must be strings")
            result[key] = _validate_json_value(
                item,
                limits=limits,
                depth=depth + 1,
                nodes=counter,
            )
        return result
    raise ValueError("Native input contains a non-JSON value")


def parse_json_object(
    value: str | bytes | bytearray | Mapping[str, object],
    *,
    protocol: str,
    limits: AdapterLimits,
) -> dict[str, Any]:
    try:
        if isinstance(value, (str, bytes, bytearray)):
            raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
            if len(raw) > limits.max_request_bytes:
                raise ValueError("Native request exceeds its size limit")
            parsed = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_constant,
            )
        elif isinstance(value, Mapping):
            parsed = dict(value)
        else:
            raise TypeError("Native request must be JSON data")
        validated = _validate_json_value(parsed, limits=limits)
        if not isinstance(validated, dict):
            raise ValueError("Native request must be a JSON object")
        encoded = json.dumps(
            validated,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > limits.max_request_bytes:
            raise ValueError("Native request exceeds its size limit")
        return validated
    except Exception:
        pass
    raise ProtocolAdapterError(protocol=protocol, code="invalid_json")


def ensure_safe_evidence(value: object, *, protocol: str) -> None:
    active: set[int] = set()

    def visit(item: object) -> None:
        if isinstance(item, dict):
            identity = id(item)
            if identity in active:
                raise ProtocolAdapterError(protocol=protocol, code="invalid_evidence")
            active.add(identity)
            try:
                for key, nested in item.items():
                    normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
                    if normalized_key in _CREDENTIAL_KEYS or normalized_key.endswith(
                        _CREDENTIAL_KEY_SUFFIXES
                    ):
                        raise ProtocolAdapterError(
                            protocol=protocol,
                            code="credential_in_evidence",
                        )
                    visit(nested)
            finally:
                active.remove(identity)
        elif isinstance(item, list):
            identity = id(item)
            if identity in active:
                raise ProtocolAdapterError(protocol=protocol, code="invalid_evidence")
            active.add(identity)
            try:
                for nested in item:
                    visit(nested)
            finally:
                active.remove(identity)
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise ProtocolAdapterError(protocol=protocol, code="invalid_evidence")
        elif isinstance(item, str) and _CREDENTIAL_VALUE.search(item):
            raise ProtocolAdapterError(protocol=protocol, code="credential_in_evidence")

    visit(value)


__all__ = ["ensure_safe_evidence", "parse_json_object"]
