"""Bounded deterministic repair for local model JSON tool arguments."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn


class ToolArgumentRepairPolicy(str, Enum):
    """Versioned repair policies."""

    BOUNDED_JSON_V1 = "bounded-json-v1"


class ToolArgumentRepairKind(str, Enum):
    """Allowed transformations for one bounded repair."""

    DOUBLE_ENCODED_OBJECT = "double_encoded_object"
    TRAILING_COMMA = "trailing_comma"
    MISSING_FINAL_DELIMITER = "missing_final_delimiter"


class ToolArgumentRepairError(ValueError):
    """A safe repair error that does not retain argument content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"tool argument repair error {code}")


@dataclass(slots=True)
class ToolArgumentRepairBudget:
    """One repair allowance shared by a complete model output."""

    used: bool = False

    def __post_init__(self) -> None:
        if type(self.used) is not bool:
            raise ValueError("Repair budget state must be boolean")

    def consume(self) -> None:
        """Consume the single repair allowance."""

        if self.used:
            raise ToolArgumentRepairError("repair_exhausted")
        self.used = True


@dataclass(frozen=True, slots=True)
class ToolArgumentRepairMetadata:
    """Sanitized metadata for one accepted repair."""

    policy: ToolArgumentRepairPolicy
    kind: ToolArgumentRepairKind
    profile: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, ToolArgumentRepairPolicy):
            raise ValueError("Invalid repair policy")
        if not isinstance(self.kind, ToolArgumentRepairKind):
            raise ValueError("Invalid repair kind")
        if type(self.profile) is not str or _PROFILE_RE.fullmatch(self.profile) is None:
            raise ValueError("Invalid repair profile")


@dataclass(frozen=True, slots=True)
class ToolArgumentRepairResult:
    """Repaired argument text plus non-content metadata."""

    arguments_raw: str = field(repr=False)
    metadata: ToolArgumentRepairMetadata

    def __post_init__(self) -> None:
        if type(self.arguments_raw) is not str:
            raise ValueError("Repaired arguments must be text")


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _RepairCandidate:
    kind: ToolArgumentRepairKind
    arguments_raw: str = field(repr=False)


_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_INVALID = object()


def _reject_constant(_: str) -> NoReturn:
    raise ValueError("invalid JSON constant")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


_JSON_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_pairs,
    parse_constant=_reject_constant,
)


def _decode_complete_json(source: str) -> object:
    try:
        value, end = _JSON_DECODER.raw_decode(source)
        if source[end:].strip():
            return _INVALID
        return value
    except (_DuplicateJsonKey, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return _INVALID


def _is_strict_object(source: str) -> bool:
    return isinstance(_decode_complete_json(source), dict)


def _double_encoded_candidate(source: str) -> _RepairCandidate | None:
    value = _decode_complete_json(source)
    if type(value) is not str:
        return None
    return _RepairCandidate(ToolArgumentRepairKind.DOUBLE_ENCODED_OBJECT, value)


def _trailing_comma_candidate(source: str) -> _RepairCandidate | None:
    positions: list[int] = []
    in_string = False
    escaped = False

    for index, character in enumerate(source):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            continue
        if character != ",":
            continue

        next_index = index + 1
        while next_index < len(source) and source[next_index] in " \t\r\n":
            next_index += 1
        if next_index < len(source) and source[next_index] in "}]":
            positions.append(index)

    if in_string:
        return None
    if len(positions) > 1:
        raise ToolArgumentRepairError("repair_ambiguous")
    if not positions:
        return None

    position = positions[0]
    return _RepairCandidate(
        ToolArgumentRepairKind.TRAILING_COMMA,
        source[:position] + source[position + 1 :],
    )


def _missing_final_delimiter_candidate(source: str) -> _RepairCandidate | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}

    for character in source:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack[-1] != matching[character]:
                return None
            stack.pop()

    if in_string or len(stack) != 1:
        return None

    closing = "}" if stack[0] == "{" else "]"
    return _RepairCandidate(
        ToolArgumentRepairKind.MISSING_FINAL_DELIMITER,
        source + closing,
    )


def _selected_policy(policy: ToolArgumentRepairPolicy | str) -> ToolArgumentRepairPolicy:
    if isinstance(policy, ToolArgumentRepairPolicy):
        return policy
    if type(policy) is not str:
        raise ToolArgumentRepairError("repair_unavailable")
    try:
        return ToolArgumentRepairPolicy(policy)
    except ValueError:
        raise ToolArgumentRepairError("repair_unavailable") from None


def _validate_profile(profile: str) -> str:
    if type(profile) is not str or _PROFILE_RE.fullmatch(profile) is None:
        raise ToolArgumentRepairError("invalid_repair_profile")
    return profile


def repair_json_object(
    source: str,
    *,
    profile: str,
    policy: ToolArgumentRepairPolicy | str = ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
    budget: ToolArgumentRepairBudget | None = None,
    max_bytes: int = 1024 * 1024,
) -> ToolArgumentRepairResult:
    """Apply one allowlisted repair to one invalid JSON object."""

    try:
        if type(source) is not str:
            raise ToolArgumentRepairError("invalid_arguments_type")
        selected_profile = _validate_profile(profile)
        selected_policy = _selected_policy(policy)
        if type(max_bytes) is not int or max_bytes < 0:
            raise ToolArgumentRepairError("invalid_repair_limit")
        if len(source.encode("utf-8")) > max_bytes:
            raise ToolArgumentRepairError("arguments_limit_exceeded")
        if budget is not None and type(budget) is not ToolArgumentRepairBudget:
            raise ToolArgumentRepairError("invalid_repair_budget")
        if _is_strict_object(source):
            raise ToolArgumentRepairError("repair_not_required")

        selected_budget = budget or ToolArgumentRepairBudget()
        selected_budget.consume()

        candidates = [
            _double_encoded_candidate(source),
            _trailing_comma_candidate(source),
            _missing_final_delimiter_candidate(source),
        ]
        accepted: list[_RepairCandidate] = []
        for candidate in candidates:
            if candidate is None:
                continue
            if len(candidate.arguments_raw.encode("utf-8")) > max_bytes:
                raise ToolArgumentRepairError("arguments_limit_exceeded")
            if _is_strict_object(candidate.arguments_raw):
                accepted.append(candidate)

        if not accepted:
            raise ToolArgumentRepairError("repair_ineligible")
        if len(accepted) != 1:
            raise ToolArgumentRepairError("repair_ambiguous")

        repaired = accepted[0]
        return ToolArgumentRepairResult(
            arguments_raw=repaired.arguments_raw,
            metadata=ToolArgumentRepairMetadata(
                policy=selected_policy,
                kind=repaired.kind,
                profile=selected_profile,
            ),
        )
    except ToolArgumentRepairError:
        raise
    except Exception:
        raise ToolArgumentRepairError("repair_failed") from None


__all__ = [
    "ToolArgumentRepairBudget",
    "ToolArgumentRepairError",
    "ToolArgumentRepairKind",
    "ToolArgumentRepairMetadata",
    "ToolArgumentRepairPolicy",
    "ToolArgumentRepairResult",
    "repair_json_object",
]
