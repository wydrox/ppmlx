"""Strict normalization of local model tool output.

Each profile accepts one documented format. The normalizer does not infer a
profile or execute a tool. An explicit profile policy can permit one bounded
argument repair before the normalized value is accepted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, NoReturn

from .tool_argument_repair import (
    ToolArgumentRepairBudget,
    ToolArgumentRepairError,
    ToolArgumentRepairMetadata,
    ToolArgumentRepairPolicy,
    repair_json_object,
)


class NormalizationProfile(str, Enum):
    """Versioned local model output formats."""

    GROK_OPENAI_CHAT_V1 = "grok-openai-chat-v1"
    KIMI_K2_V1 = "kimi-k2-v1"
    DEEPSEEK_V3_V1 = "deepseek-v3-v1"
    QWEN_JSON_V1 = "qwen-json-v1"
    GEMMA4_V1 = "gemma4-v1"
    LFM25_V1 = "lfm25-v1"


@dataclass(frozen=True, slots=True)
class ToolOutputLimits:
    """Resource limits for one model output."""

    max_output_bytes: int = 2 * 1024 * 1024
    max_calls: int = 128
    max_arguments_bytes: int = 1024 * 1024
    max_name_bytes: int = 256
    max_call_id_bytes: int = 512
    max_json_depth: int = 64
    max_json_nodes: int = 100_000
    max_json_string_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.max_output_bytes,
            self.max_calls,
            self.max_arguments_bytes,
            self.max_name_bytes,
            self.max_call_id_bytes,
            self.max_json_depth,
            self.max_json_nodes,
            self.max_json_string_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Tool output limits must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    """A model tool call with accepted argument text."""

    index: int
    name: str
    arguments_raw: str
    arguments_json: dict[str, object]
    call_id: str | None = None
    repair: ToolArgumentRepairMetadata | None = None


@dataclass(frozen=True, slots=True)
class NormalizedToolOutput:
    """Normalized calls and text outside the accepted tool envelope."""

    profile: NormalizationProfile
    tool_calls: tuple[NormalizedToolCall, ...]
    remaining_text: str


class ToolNormalizationError(ValueError):
    """A safe error that does not include model output."""

    def __init__(self, *, profile: str, code: str) -> None:
        self.profile = profile
        self.code = code
        super().__init__(f"local tool normalization error {code}")


class _DuplicateJsonKey(ValueError):
    pass


_NAME_RE = re.compile(r"^[A-Za-z0-9_$][A-Za-z0-9_$./:-]*$")
_CALL_ID_RE = re.compile(r"^\S+$")
_KIMI_ID_RE = re.compile(r"^functions\.([A-Za-z0-9_$][A-Za-z0-9_$./-]*):([0-9]+)$")

_QWEN_START = "<tool_call>"
_QWEN_END = "</tool_call>"

_KIMI_SECTION_START = "<|tool_calls_section_begin|>"
_KIMI_SECTION_END = "<|tool_calls_section_end|>"
_KIMI_CALL_START = "<|tool_call_begin|>"
_KIMI_CALL_END = "<|tool_call_end|>"
_KIMI_ARGUMENT_START = "<|tool_call_argument_begin|>"

_DEEPSEEK_SECTION_START = "<｜tool▁calls▁begin｜>"
_DEEPSEEK_SECTION_END = "<｜tool▁calls▁end｜>"
_DEEPSEEK_CALL_START = "<｜tool▁call▁begin｜>"
_DEEPSEEK_CALL_END = "<｜tool▁call▁end｜>"
_DEEPSEEK_SEPARATOR = "<｜tool▁sep｜>"

_GEMMA_CALL_START = "<|tool_call>"
_GEMMA_CALL_PREFIX = "call:"

_LFM_SECTION_START = "<|tool_call_start|>"
_LFM_SECTION_END = "<|tool_call_end|>"

_LITERAL_MAX_DEPTH = 64


def normalize_tool_output(
    text: str,
    *,
    profile: NormalizationProfile | str,
    limits: ToolOutputLimits = ToolOutputLimits(),
    repair_policy: ToolArgumentRepairPolicy | str | None = None,
) -> NormalizedToolOutput:
    """Normalize one output with an explicit, versioned profile."""

    profile_name = profile.value if isinstance(profile, NormalizationProfile) else "local-tool-output"
    try:
        selected = profile if isinstance(profile, NormalizationProfile) else NormalizationProfile(profile)
        if type(text) is not str:
            _raise(selected.value, "invalid_output_type")
        if not isinstance(limits, ToolOutputLimits):
            _raise(selected.value, "invalid_limits")
        if len(text.encode("utf-8")) > limits.max_output_bytes:
            _raise(selected.value, "output_limit_exceeded")
        selected_policy = _select_repair_policy(repair_policy, profile=selected.value)
        parser = _PARSERS[selected]
        return parser(
            text,
            limits,
            selected_policy,
            ToolArgumentRepairBudget(),
        )
    except ToolNormalizationError as error:
        details = (error.profile, error.code)
    except ToolArgumentRepairError as error:
        details = (profile_name, error.code)
    except (LookupError, TypeError, ValueError):
        details = (profile_name, "invalid_tool_output")
    except Exception:
        details = (profile_name, "invalid_tool_output")
    raise ToolNormalizationError(profile=details[0], code=details[1]) from None


def _select_repair_policy(
    policy: ToolArgumentRepairPolicy | str | None,
    *,
    profile: str,
) -> ToolArgumentRepairPolicy | None:
    if policy is None:
        return None
    if isinstance(policy, ToolArgumentRepairPolicy):
        return policy
    if type(policy) is not str:
        _raise(profile, "repair_unavailable")
    try:
        return ToolArgumentRepairPolicy(policy)
    except ValueError:
        _raise(profile, "repair_unavailable")


def _raise(profile: str, code: str) -> NoReturn:
    raise ToolNormalizationError(profile=profile, code=code)


def _reject_constant(_: str) -> None:
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


def _parse_json(source: str, *, profile: str, limits: ToolOutputLimits) -> object:
    try:
        value, end = _JSON_DECODER.raw_decode(source)
        if source[end:].strip():
            _raise(profile, "malformed_arguments")
    except ToolNormalizationError:
        raise
    except _DuplicateJsonKey:
        _raise(profile, "duplicate_json_key")
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        _raise(profile, "malformed_arguments")
    _check_json_limits(value, profile=profile, limits=limits)
    return value


def _check_json_limits(value: object, *, profile: str, limits: ToolOutputLimits) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_json_nodes:
            _raise(profile, "json_node_limit_exceeded")
        if depth > limits.max_json_depth:
            _raise(profile, "json_depth_limit_exceeded")
        if isinstance(item, str):
            if len(item.encode("utf-8")) > limits.max_json_string_bytes:
                _raise(profile, "json_string_limit_exceeded")
        elif isinstance(item, dict):
            for key, child in item.items():
                if len(key.encode("utf-8")) > limits.max_json_string_bytes:
                    _raise(profile, "json_string_limit_exceeded")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _parse_json_object_with_fields(
    source: str,
    *,
    profile: str,
    limits: ToolOutputLimits,
) -> tuple[dict[str, object], dict[str, str]]:
    """Parse one JSON object and retain each top-level value lexeme."""

    value = _parse_json(source, profile=profile, limits=limits)
    if not isinstance(value, dict):
        _raise(profile, "arguments_not_object")

    fields: dict[str, str] = {}
    position = _skip_space(source, 0)
    if position >= len(source) or source[position] != "{":
        _raise(profile, "malformed_arguments")
    position = _skip_space(source, position + 1)
    if position < len(source) and source[position] == "}":
        return value, fields

    while position < len(source):
        try:
            key, key_end = json.JSONDecoder().raw_decode(source, position)
        except (json.JSONDecodeError, ValueError):
            _raise(profile, "malformed_arguments")
        if type(key) is not str:
            _raise(profile, "malformed_arguments")
        position = _skip_space(source, key_end)
        if position >= len(source) or source[position] != ":":
            _raise(profile, "malformed_arguments")
        value_start = _skip_space(source, position + 1)
        try:
            _, value_end = _JSON_DECODER.raw_decode(source, value_start)
        except (_DuplicateJsonKey, json.JSONDecodeError, TypeError, ValueError, RecursionError):
            _raise(profile, "malformed_arguments")
        fields[key] = source[value_start:value_end]
        position = _skip_space(source, value_end)
        if position < len(source) and source[position] == ",":
            position = _skip_space(source, position + 1)
            continue
        if position < len(source) and source[position] == "}":
            position = _skip_space(source, position + 1)
            if position != len(source):
                _raise(profile, "malformed_arguments")
            return value, fields
        _raise(profile, "malformed_arguments")
    _raise(profile, "malformed_arguments")


def _parse_qwen_repair_fields(
    source: str,
    *,
    profile: str,
) -> tuple[str, str]:
    """Isolate the documented Qwen argument field without repairing its envelope."""

    position = _skip_space(source, 0)
    if position >= len(source) or source[position] != "{":
        _raise(profile, "invalid_tool_call_shape")
    position = _skip_space(source, position + 1)

    try:
        first_key, position = _JSON_DECODER.raw_decode(source, position)
    except (_DuplicateJsonKey, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        _raise(profile, "invalid_tool_call_shape")
    if first_key != "name":
        _raise(profile, "invalid_tool_call_shape")
    position = _skip_space(source, position)
    if position >= len(source) or source[position] != ":":
        _raise(profile, "invalid_tool_call_shape")
    position = _skip_space(source, position + 1)

    try:
        name, position = _JSON_DECODER.raw_decode(source, position)
    except (_DuplicateJsonKey, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        _raise(profile, "invalid_tool_call_shape")
    if type(name) is not str:
        _raise(profile, "invalid_tool_call_shape")
    position = _skip_space(source, position)
    if position >= len(source) or source[position] != ",":
        _raise(profile, "invalid_tool_call_shape")
    position = _skip_space(source, position + 1)

    try:
        second_key, position = _JSON_DECODER.raw_decode(source, position)
    except (_DuplicateJsonKey, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        _raise(profile, "invalid_tool_call_shape")
    if second_key != "arguments":
        _raise(profile, "invalid_tool_call_shape")
    position = _skip_space(source, position)
    if position >= len(source) or source[position] != ":":
        _raise(profile, "invalid_tool_call_shape")
    arguments_start = _skip_space(source, position + 1)

    outer_end = len(source)
    while outer_end > arguments_start and source[outer_end - 1] in " \t\r\n":
        outer_end -= 1
    if outer_end <= arguments_start:
        _raise(profile, "invalid_tool_call_shape")
    if source[outer_end - 1] == "}":
        # The documented envelope keeps one closing brace for the call object.
        # It is never part of the argument value.
        arguments_raw = source[arguments_start : outer_end - 1].rstrip()
        if not arguments_raw:
            _raise(profile, "invalid_tool_call_shape")
        return name, arguments_raw
    if source[outer_end - 1] == "]":
        # The envelope brace is missing and the argument value holds the final
        # delimiter; the repair surface stays inside the argument value.
        arguments_raw = source[arguments_start:outer_end].rstrip()
        if not arguments_raw:
            _raise(profile, "invalid_tool_call_shape")
        return name, arguments_raw
    _raise(profile, "invalid_tool_call_shape")


def _skip_space(source: str, position: int) -> int:
    while position < len(source) and source[position] in " \t\r\n":
        position += 1
    return position


def _validate_name(name: object, *, profile: str, limits: ToolOutputLimits) -> str:
    if (
        type(name) is not str
        or not name
        or len(name.encode("utf-8")) > limits.max_name_bytes
        or _NAME_RE.fullmatch(name) is None
    ):
        _raise(profile, "invalid_tool_name")
    return name


def _validate_call_id(call_id: object, *, profile: str, limits: ToolOutputLimits) -> str:
    if (
        type(call_id) is not str
        or not call_id
        or len(call_id.encode("utf-8")) > limits.max_call_id_bytes
        or _CALL_ID_RE.fullmatch(call_id) is None
    ):
        _raise(profile, "invalid_call_id")
    return call_id


def _repair_arguments(
    raw: str,
    *,
    profile: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy,
    repair_budget: ToolArgumentRepairBudget,
) -> tuple[str, dict[str, object], ToolArgumentRepairMetadata]:
    try:
        repaired = repair_json_object(
            raw,
            profile=profile,
            policy=repair_policy,
            budget=repair_budget,
            max_bytes=limits.max_arguments_bytes,
        )
    except ToolArgumentRepairError as error:
        _raise(profile, error.code)
    value = _parse_json(repaired.arguments_raw, profile=profile, limits=limits)
    if not isinstance(value, dict):
        _raise(profile, "repair_failed")
    return repaired.arguments_raw, value, repaired.metadata


def _arguments(
    raw: str,
    *,
    profile: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> tuple[str, dict[str, object], ToolArgumentRepairMetadata | None]:
    if len(raw.encode("utf-8")) > limits.max_arguments_bytes:
        _raise(profile, "arguments_limit_exceeded")
    try:
        value = _parse_json(raw, profile=profile, limits=limits)
    except ToolNormalizationError as error:
        if repair_policy is None or error.code != "malformed_arguments":
            raise
        return _repair_arguments(
            raw,
            profile=profile,
            limits=limits,
            repair_policy=repair_policy,
            repair_budget=repair_budget,
        )
    if isinstance(value, dict):
        return raw, value, None
    if repair_policy is not None and type(value) is str:
        return _repair_arguments(
            raw,
            profile=profile,
            limits=limits,
            repair_policy=repair_policy,
            repair_budget=repair_budget,
        )
    _raise(profile, "arguments_not_object")


def _call(
    *,
    index: int,
    name: object,
    arguments_raw: str,
    profile: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
    call_id: object | None = None,
) -> NormalizedToolCall:
    normalized_name = _validate_name(name, profile=profile, limits=limits)
    normalized_call_id = None
    if call_id is not None:
        normalized_call_id = _validate_call_id(call_id, profile=profile, limits=limits)
    accepted_raw, accepted_json, repair = _arguments(
        arguments_raw,
        profile=profile,
        limits=limits,
        repair_policy=repair_policy,
        repair_budget=repair_budget,
    )
    return NormalizedToolCall(
        index=index,
        name=normalized_name,
        arguments_raw=accepted_raw,
        arguments_json=accepted_json,
        call_id=normalized_call_id,
        repair=repair,
    )


def _check_calls(
    calls: list[NormalizedToolCall],
    *,
    profile: str,
    limits: ToolOutputLimits,
) -> tuple[NormalizedToolCall, ...]:
    if not calls:
        _raise(profile, "empty_tool_section")
    if len(calls) > limits.max_calls:
        _raise(profile, "call_limit_exceeded")
    identifiers = [call.call_id for call in calls if call.call_id is not None]
    if len(identifiers) != len(set(identifiers)):
        _raise(profile, "duplicate_call_id")
    return tuple(calls)


def _single_section(
    text: str,
    *,
    profile: str,
    start: str,
    end: str,
) -> tuple[str, str]:
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count == 0 and end_count == 0:
        return text, ""
    if start_count != 1 or end_count != 1:
        _raise(profile, "ambiguous_tool_section")
    before, section_and_after = text.split(start, 1)
    section, after = section_and_after.split(end, 1)
    if after.strip():
        _raise(profile, "text_after_tool_section")
    return before, section


def _parse_delimited_calls(
    section: str,
    *,
    profile: str,
    call_start: str,
    call_end: str,
    limits: ToolOutputLimits,
    parse_body: Callable[[str, int], NormalizedToolCall],
) -> tuple[NormalizedToolCall, ...]:
    calls: list[NormalizedToolCall] = []
    rest = section
    while rest.strip():
        rest = rest.lstrip()
        if not rest.startswith(call_start):
            _raise(profile, "malformed_tool_section")
        after_start = rest[len(call_start) :]
        end_position = after_start.find(call_end)
        if end_position < 0:
            _raise(profile, "unterminated_tool_call")
        body = after_start[:end_position]
        if call_start in body:
            _raise(profile, "nested_tool_call")
        calls.append(parse_body(body, len(calls)))
        if len(calls) > limits.max_calls:
            _raise(profile, "call_limit_exceeded")
        rest = after_start[end_position + len(call_end) :]
    return _check_calls(calls, profile=profile, limits=limits)


def _parse_qwen(
    text: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolOutput:
    profile = NormalizationProfile.QWEN_JSON_V1.value
    if _QWEN_START not in text and _QWEN_END not in text:
        return NormalizedToolOutput(NormalizationProfile.QWEN_JSON_V1, (), text)
    if text.count(_QWEN_START) != text.count(_QWEN_END):
        _raise(profile, "unterminated_tool_call")

    first = text.find(_QWEN_START)
    remaining_text = text[:first]
    rest = text[first:]

    def parse_body(body: str, index: int) -> NormalizedToolCall:
        source = body.strip()
        try:
            value, fields = _parse_json_object_with_fields(
                source,
                profile=profile,
                limits=limits,
            )
        except ToolNormalizationError as error:
            if repair_policy is None or error.code != "malformed_arguments":
                raise
            name, repairable_arguments = _parse_qwen_repair_fields(
                source,
                profile=profile,
            )
            call = _call(
                index=index,
                name=name,
                arguments_raw=repairable_arguments,
                profile=profile,
                limits=limits,
                repair_policy=repair_policy,
                repair_budget=repair_budget,
            )
            return call
        if set(value) != {"name", "arguments"}:
            _raise(profile, "invalid_tool_call_shape")
        if "arguments" not in fields:
            _raise(profile, "invalid_tool_call_shape")
        return _call(
            index=index,
            name=value["name"],
            arguments_raw=fields["arguments"],
            profile=profile,
            limits=limits,
            repair_policy=repair_policy,
            repair_budget=repair_budget,
        )

    calls = _parse_delimited_calls(
        rest,
        profile=profile,
        call_start=_QWEN_START,
        call_end=_QWEN_END,
        limits=limits,
        parse_body=parse_body,
    )
    return NormalizedToolOutput(NormalizationProfile.QWEN_JSON_V1, calls, remaining_text)


def _parse_kimi(
    text: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolOutput:
    profile = NormalizationProfile.KIMI_K2_V1.value
    has_section_marker = _KIMI_SECTION_START in text or _KIMI_SECTION_END in text
    remaining_text, section = _single_section(
        text,
        profile=profile,
        start=_KIMI_SECTION_START,
        end=_KIMI_SECTION_END,
    )
    if not has_section_marker:
        return NormalizedToolOutput(NormalizationProfile.KIMI_K2_V1, (), remaining_text)

    def parse_body(body: str, index: int) -> NormalizedToolCall:
        if body.count(_KIMI_ARGUMENT_START) != 1:
            _raise(profile, "invalid_tool_call_shape")
        call_id, arguments_raw = body.split(_KIMI_ARGUMENT_START, 1)
        match = _KIMI_ID_RE.fullmatch(call_id)
        if match is None:
            _raise(profile, "invalid_call_id")
        name = match.group(1)
        return _call(
            index=index,
            name=name,
            arguments_raw=arguments_raw,
            profile=profile,
            limits=limits,
            repair_policy=repair_policy,
            repair_budget=repair_budget,
            call_id=call_id,
        )

    calls = _parse_delimited_calls(
        section,
        profile=profile,
        call_start=_KIMI_CALL_START,
        call_end=_KIMI_CALL_END,
        limits=limits,
        parse_body=parse_body,
    )
    return NormalizedToolOutput(NormalizationProfile.KIMI_K2_V1, calls, remaining_text)


def _parse_deepseek(
    text: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolOutput:
    profile = NormalizationProfile.DEEPSEEK_V3_V1.value
    has_section_marker = _DEEPSEEK_SECTION_START in text or _DEEPSEEK_SECTION_END in text
    remaining_text, section = _single_section(
        text,
        profile=profile,
        start=_DEEPSEEK_SECTION_START,
        end=_DEEPSEEK_SECTION_END,
    )
    if not has_section_marker:
        return NormalizedToolOutput(NormalizationProfile.DEEPSEEK_V3_V1, (), remaining_text)

    def parse_body(body: str, index: int) -> NormalizedToolCall:
        prefix = "function" + _DEEPSEEK_SEPARATOR
        if not body.startswith(prefix):
            _raise(profile, "invalid_tool_call_type")
        payload = body[len(prefix) :]
        fence = "\n```json\n"
        if payload.count(fence) != 1 or not payload.endswith("\n```"):
            _raise(profile, "invalid_tool_call_shape")
        name, fenced_arguments = payload.split(fence, 1)
        arguments_raw = fenced_arguments[:-4]
        return _call(
            index=index,
            name=name,
            arguments_raw=arguments_raw,
            profile=profile,
            limits=limits,
            repair_policy=repair_policy,
            repair_budget=repair_budget,
        )

    calls = _parse_delimited_calls(
        section,
        profile=profile,
        call_start=_DEEPSEEK_CALL_START,
        call_end=_DEEPSEEK_CALL_END,
        limits=limits,
        parse_body=parse_body,
    )
    return NormalizedToolOutput(NormalizationProfile.DEEPSEEK_V3_V1, calls, remaining_text)


def _parse_literal_value(
    source: str,
    position: int,
    *,
    profile: str,
    depth: int,
) -> tuple[object, int]:
    """Parse one JSON-compatible literal at ``position``.

    Accepts JSON strings, numbers, ``true``/``false``/``null``, and the
    array/object containers used by the Gemma 4 and LFM2.5 argument
    renderings. Object keys may be bare identifiers or JSON strings.
    """

    if depth > _LITERAL_MAX_DEPTH:
        _raise(profile, "json_depth_limit_exceeded")
    if position >= len(source):
        _raise(profile, "malformed_arguments")
    head = source[position]
    if head in "[{":
        return _parse_literal_container(source, position, profile=profile, depth=depth)
    try:
        value, end = _JSON_DECODER.raw_decode(source, position)
    except (_DuplicateJsonKey, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        _raise(profile, "malformed_arguments")
    return value, end


def _parse_literal_container(
    source: str,
    position: int,
    *,
    profile: str,
    depth: int,
) -> tuple[object, int]:
    opening = source[position]
    closing = "]" if opening == "[" else "}"
    position = _skip_space(source, position + 1)
    items: list[object] = []
    fields: dict[str, object] = {}
    if position < len(source) and source[position] == closing:
        return (items if opening == "[" else fields), position + 1
    while True:
        if position >= len(source):
            _raise(profile, "malformed_arguments")
        if opening == "{":
            key: object
            if source[position] == '"':
                try:
                    key, position = _JSON_DECODER.raw_decode(source, position)
                except (
                    _DuplicateJsonKey,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    RecursionError,
                ):
                    _raise(profile, "malformed_arguments")
            else:
                key_end = position
                while key_end < len(source) and source[key_end] not in ":{}[], \t\r\n":
                    key_end += 1
                if key_end == position:
                    _raise(profile, "malformed_arguments")
                key = source[position:key_end]
                position = key_end
            if type(key) is not str:
                _raise(profile, "malformed_arguments")
            position = _skip_space(source, position)
            if position >= len(source) or source[position] != ":":
                _raise(profile, "malformed_arguments")
            value, position = _parse_literal_value(
                source,
                _skip_space(source, position + 1),
                profile=profile,
                depth=depth + 1,
            )
            if key in fields:
                raise _DuplicateJsonKey
            fields[key] = value
        else:
            value, position = _parse_literal_value(
                source,
                position,
                profile=profile,
                depth=depth + 1,
            )
            items.append(value)
        position = _skip_space(source, position)
        if position < len(source) and source[position] == ",":
            position = _skip_space(source, position + 1)
            continue
        if position < len(source) and source[position] == closing:
            return (items if opening == "[" else fields), position + 1
        _raise(profile, "malformed_arguments")


def _split_top_level(source: str, *, separator: str) -> list[str]:
    """Split on commas that are outside quotes, parentheses, and brackets."""

    parts: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    start = 0
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
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == separator and depth == 0:
            parts.append(source[start:index])
            start = index + 1
    parts.append(source[start:])
    return parts


def _literal_arguments(
    value: object,
    *,
    index: int,
    name: str,
    profile: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolCall:
    """Build a call from already-decoded literal arguments."""

    if not isinstance(value, dict):
        _raise(profile, "arguments_not_object")
    return _call(
        index=index,
        name=name,
        arguments_raw=json.dumps(value),
        profile=profile,
        limits=limits,
        repair_policy=repair_policy,
        repair_budget=repair_budget,
    )


def _parse_gemma4(
    text: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolOutput:
    profile = NormalizationProfile.GEMMA4_V1.value
    if _GEMMA_CALL_START not in text:
        return NormalizedToolOutput(NormalizationProfile.GEMMA4_V1, (), text)

    first = text.find(_GEMMA_CALL_START)
    remaining_text = text[:first]
    rest = text[first:]
    calls: list[NormalizedToolCall] = []
    while True:
        rest = rest.lstrip()
        if not rest:
            break
        if not rest.startswith(_GEMMA_CALL_START):
            _raise(profile, "text_after_tool_section")
        after_start = rest[len(_GEMMA_CALL_START) :].lstrip()
        if not after_start.startswith(_GEMMA_CALL_PREFIX):
            _raise(profile, "invalid_tool_call_shape")
        after_prefix = after_start[len(_GEMMA_CALL_PREFIX) :]
        brace = after_prefix.find("{")
        if brace < 0:
            _raise(profile, "invalid_tool_call_shape")
        name = after_prefix[:brace].strip()
        try:
            arguments, end_position = _parse_literal_container(
                after_prefix,
                brace,
                profile=profile,
                depth=0,
            )
        except _DuplicateJsonKey:
            _raise(profile, "duplicate_json_key")
        except RecursionError:
            _raise(profile, "json_depth_limit_exceeded")
        _check_json_limits(arguments, profile=profile, limits=limits)
        calls.append(
            _literal_arguments(
                arguments,
                index=len(calls),
                name=name.strip(),
                profile=profile,
                limits=limits,
                repair_policy=repair_policy,
                repair_budget=repair_budget,
            )
        )
        if len(calls) > limits.max_calls:
            _raise(profile, "call_limit_exceeded")
        rest = after_prefix[end_position:]
    return NormalizedToolOutput(
        NormalizationProfile.GEMMA4_V1,
        _check_calls(calls, profile=profile, limits=limits),
        remaining_text,
    )


_LFM_NAME_END_RE = re.compile(r"^[A-Za-z0-9_$][A-Za-z0-9_$.]*$")


def _lfm_call_element(element: str, *, index: int, profile: str) -> tuple[str, dict[str, object]]:
    open_position = element.find("(")
    if open_position < 0:
        _raise(profile, "invalid_tool_call_shape")
    name = element[:open_position].strip()
    if _LFM_NAME_END_RE.fullmatch(name) is None:
        _raise(profile, "invalid_tool_name")
    payload = element[open_position + 1 :].rstrip()
    if not payload.endswith(")"):
        _raise(profile, "invalid_tool_call_shape")
    payload = payload[:-1].strip()
    if not payload:
        return name, {}
    arguments: dict[str, object] = {}
    for assignment in _split_top_level(payload, separator=","):
        stripped = assignment.strip()
        if not stripped:
            _raise(profile, "malformed_arguments")
        equals = stripped.find("=")
        if equals <= 0:
            _raise(profile, "malformed_arguments")
        key = stripped[:equals].strip()
        if _LFM_NAME_END_RE.fullmatch(key) is None:
            _raise(profile, "malformed_arguments")
        if key in arguments:
            raise _DuplicateJsonKey
        value, end_position = _parse_literal_value(
            stripped,
            _skip_space(stripped, equals + 1),
            profile=profile,
            depth=0,
        )
        if stripped[end_position:].strip():
            _raise(profile, "malformed_arguments")
        arguments[key] = value
    return name, arguments


def _parse_lfm25(
    text: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolOutput:
    profile = NormalizationProfile.LFM25_V1.value
    has_section_marker = _LFM_SECTION_START in text or _LFM_SECTION_END in text
    remaining_text, section = _single_section(
        text,
        profile=profile,
        start=_LFM_SECTION_START,
        end=_LFM_SECTION_END,
    )
    if not has_section_marker:
        return NormalizedToolOutput(NormalizationProfile.LFM25_V1, (), remaining_text)

    section = section.strip()
    if not section.startswith("[") or not section.endswith("]"):
        _raise(profile, "invalid_tool_call_shape")
    inner = section[1:-1]
    if not inner.strip():
        _raise(profile, "empty_tool_section")

    calls: list[NormalizedToolCall] = []
    for element in _split_top_level(inner, separator=","):
        stripped = element.strip()
        if not stripped:
            _raise(profile, "malformed_tool_section")
        try:
            name, arguments = _lfm_call_element(stripped, index=len(calls), profile=profile)
        except _DuplicateJsonKey:
            _raise(profile, "duplicate_json_key")
        except RecursionError:
            _raise(profile, "json_depth_limit_exceeded")
        _check_json_limits(arguments, profile=profile, limits=limits)
        calls.append(
            _literal_arguments(
                arguments,
                index=len(calls),
                name=name,
                profile=profile,
                limits=limits,
                repair_policy=repair_policy,
                repair_budget=repair_budget,
            )
        )
        if len(calls) > limits.max_calls:
            _raise(profile, "call_limit_exceeded")
    return NormalizedToolOutput(
        NormalizationProfile.LFM25_V1,
        _check_calls(calls, profile=profile, limits=limits),
        remaining_text,
    )


def _parse_grok(
    text: str,
    limits: ToolOutputLimits,
    repair_policy: ToolArgumentRepairPolicy | None,
    repair_budget: ToolArgumentRepairBudget,
) -> NormalizedToolOutput:
    profile = NormalizationProfile.GROK_OPENAI_CHAT_V1.value
    if not text.lstrip().startswith("{"):
        return NormalizedToolOutput(NormalizationProfile.GROK_OPENAI_CHAT_V1, (), text)
    value = _parse_json(text, profile=profile, limits=limits)
    if not isinstance(value, dict) or not {"role", "content", "tool_calls"} <= set(value):
        _raise(profile, "invalid_tool_call_shape")
    if set(value) - {"role", "content", "tool_calls"}:
        _raise(profile, "invalid_tool_call_shape")
    if value["role"] != "assistant":
        _raise(profile, "ambiguous_tool_output")
    native_calls = value["tool_calls"]
    if not isinstance(native_calls, list):
        _raise(profile, "invalid_tool_call_shape")
    if len(native_calls) > limits.max_calls:
        _raise(profile, "call_limit_exceeded")
    if not native_calls:
        content = value["content"]
        if type(content) is not str or not content:
            _raise(profile, "ambiguous_tool_output")
        return NormalizedToolOutput(
            NormalizationProfile.GROK_OPENAI_CHAT_V1,
            (),
            content,
        )
    if value["content"] not in (None, ""):
        _raise(profile, "ambiguous_tool_output")

    calls: list[NormalizedToolCall] = []
    for index, native_call in enumerate(native_calls):
        if not isinstance(native_call, dict) or set(native_call) != {"id", "type", "function"}:
            _raise(profile, "invalid_tool_call_shape")
        if native_call["type"] != "function" or not isinstance(native_call["function"], dict):
            _raise(profile, "invalid_tool_call_type")
        function = native_call["function"]
        if set(function) != {"name", "arguments"} or type(function["arguments"]) is not str:
            _raise(profile, "invalid_tool_call_shape")
        calls.append(
            _call(
                index=index,
                name=function["name"],
                arguments_raw=function["arguments"],
                profile=profile,
                limits=limits,
                repair_policy=repair_policy,
                repair_budget=repair_budget,
                call_id=native_call["id"],
            )
        )
    normalized = _check_calls(calls, profile=profile, limits=limits)
    return NormalizedToolOutput(NormalizationProfile.GROK_OPENAI_CHAT_V1, normalized, "")


_Parser = Callable[
    [
        str,
        ToolOutputLimits,
        ToolArgumentRepairPolicy | None,
        ToolArgumentRepairBudget,
    ],
    NormalizedToolOutput,
]

_PARSERS: dict[NormalizationProfile, _Parser] = {
    NormalizationProfile.GROK_OPENAI_CHAT_V1: _parse_grok,
    NormalizationProfile.KIMI_K2_V1: _parse_kimi,
    NormalizationProfile.DEEPSEEK_V3_V1: _parse_deepseek,
    NormalizationProfile.QWEN_JSON_V1: _parse_qwen,
    NormalizationProfile.GEMMA4_V1: _parse_gemma4,
    NormalizationProfile.LFM25_V1: _parse_lfm25,
}


# Exact reviewed model repositories. A model identifier selects a profile only
# when it names one reviewed repository; a family-name match does not create a
# capability claim (docs/capabilities/tool-profiles.md).
_REVIEWED_MODEL_PROFILES: dict[str, NormalizationProfile] = {}


def select_normalization_profile(model: str) -> NormalizationProfile | None:
    """Select a profile only for an exact reviewed model repository.

    A family-name or substring match does not create a capability claim.
    Unknown models select no profile and stay strict.
    """

    if type(model) is not str:
        return None
    return _REVIEWED_MODEL_PROFILES.get(model)


__all__ = [
    "NormalizationProfile",
    "NormalizedToolCall",
    "NormalizedToolOutput",
    "ToolNormalizationError",
    "ToolOutputLimits",
    "normalize_tool_output",
    "select_normalization_profile",
]
