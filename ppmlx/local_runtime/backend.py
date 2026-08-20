"""Protocol-neutral local inference that emits Agent IR events."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, cast

from jsonschema import Draft202012Validator, SchemaError, ValidationError, validators
from pydantic import JsonValue
import regex

from ppmlx.agent_ir import (
    AgentEvent,
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    Instruction,
    Message,
    Origin,
    Provenance,
    RequestEnvelope,
    ResponseCompletedEvent,
    Sensitivity,
    TextBlock,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolResultBlock,
    Trust,
    Usage,
    UsageSource,
    new_call_id,
    new_output_id,
    new_parallel_group_id,
)
from ppmlx.protocols import CallReference

from .normalization import (
    NormalizationProfile,
    ToolOutputLimits,
    normalize_tool_output,
)
from .tool_profiles import get_tool_profile_contract


class LocalRuntimeError(ValueError):
    """A safe local-runtime error that does not contain request or model text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Local runtime error {code}")


@dataclass(frozen=True, slots=True)
class LocalGeneration:
    """The visible result of one local model generation."""

    text: str
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise LocalRuntimeError("invalid_generation")
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 0:
            raise LocalRuntimeError("invalid_usage")
        if type(self.completion_tokens) is not int or self.completion_tokens < 0:
            raise LocalRuntimeError("invalid_usage")


@dataclass(frozen=True, slots=True)
class LocalEngineRequest:
    """The small request surface that the MLX engine consumes."""

    model: str
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...]
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    stop: tuple[str, ...] | None
    seed: int | None
    enable_thinking: bool


class LocalGenerator(Protocol):
    """A local engine port with no dependency on an API protocol."""

    def __call__(self, request: LocalEngineRequest) -> LocalGeneration: ...


@dataclass(frozen=True, slots=True)
class TerminalReasons:
    """Native terminal values required by the selected egress adapter."""

    text: str
    tool_calls: str

    def __post_init__(self) -> None:
        if type(self.text) is not str or not self.text:
            raise LocalRuntimeError("invalid_terminal_reason")
        if type(self.tool_calls) is not str or not self.tool_calls:
            raise LocalRuntimeError("invalid_terminal_reason")


@dataclass(frozen=True, slots=True)
class LocalExecution:
    """Agent IR events and stable call references from one generation."""

    events: tuple[AgentEvent, ...]
    calls: tuple[CallReference, ...]
    source_call_ids: Mapping[str, str]


def _policy(sensitivity: Sensitivity, model: str) -> dict[str, object]:
    return {
        "sensitivity": sensitivity,
        "provenance": Provenance(
            origin=Origin.PROVIDER,
            trust=Trust.UNTRUSTED,
            origin_id=model,
        ),
    }


def _text_content(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "refusal":
            parts.append(block.text)
        elif block.type == "extension" and not block.required:
            continue
        else:
            raise LocalRuntimeError("unsupported_message_content")
    return "\n".join(parts)


def _engine_message(message: Message) -> Mapping[str, object]:
    if message.role == "assistant":
        text: list[str] = []
        tool_calls: list[dict[str, object]] = []
        for block in message.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_call":
                tool_calls.append(
                    {
                        "id": str(block.call_id),
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.arguments_raw,
                        },
                    }
                )
            elif block.type == "extension" and not block.required:
                continue
            else:
                raise LocalRuntimeError("unsupported_message_content")
        value: dict[str, object] = {
            "role": "assistant",
            "content": "\n".join(text) if text else None,
        }
        if tool_calls:
            value["tool_calls"] = tool_calls
        return value

    if message.role == "tool":
        if len(message.content) != 1 or message.content[0].type != "tool_result":
            raise LocalRuntimeError("invalid_tool_result_message")
        result = message.content[0]
        content_parts: list[str] = []
        for block in result.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "extension" and not block.required:
                continue
            else:
                raise LocalRuntimeError("unsupported_tool_result_content")
        value = {
            "role": "tool",
            "tool_call_id": str(result.call_id),
            "content": "\n".join(content_parts),
        }
        if message.name is not None:
            value["name"] = message.name
        return value

    value = {"role": message.role, "content": _text_content(message)}
    if message.name is not None:
        value["name"] = message.name
    return value


def _tool_result_message(block: ToolResultBlock, *, name: str | None) -> Mapping[str, object]:
    content_parts: list[str] = []
    for content in block.content:
        if content.type == "text":
            content_parts.append(content.text)
        elif content.type == "extension" and not content.required:
            continue
        else:
            raise LocalRuntimeError("unsupported_tool_result_content")
    value: dict[str, object] = {
        "role": "tool",
        "tool_call_id": str(block.call_id),
        "content": "\n".join(content_parts),
    }
    if name is not None:
        value["name"] = name
    return value


def _engine_messages(message: Message) -> tuple[Mapping[str, object], ...]:
    if message.role != "user" or not any(block.type == "tool_result" for block in message.content):
        return (_engine_message(message),)

    result: list[Mapping[str, object]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            value: dict[str, object] = {"role": "user", "content": "\n".join(text_parts)}
            if message.name is not None:
                value["name"] = message.name
            result.append(value)
            text_parts.clear()

    for block in message.content:
        if block.type == "tool_result":
            flush_text()
            result.append(_tool_result_message(block, name=message.name))
        elif block.type == "text":
            text_parts.append(block.text)
        elif block.type == "refusal":
            text_parts.append(block.text)
        elif block.type == "extension" and not block.required:
            continue
        else:
            raise LocalRuntimeError("unsupported_message_content")
    flush_text()
    return tuple(result)


def _instruction_message(instruction: Instruction) -> Mapping[str, object]:
    if instruction.source_role not in {"system", "developer", "user", "assistant"}:
        raise LocalRuntimeError("unsupported_instruction_role")
    parts: list[str] = []
    for block in instruction.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "refusal":
            parts.append(block.text)
        elif block.type == "extension" and not block.required:
            continue
        else:
            raise LocalRuntimeError("unsupported_instruction_content")
    return {"role": instruction.source_role, "content": "\n".join(parts)}


_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$ref",
        "allOf",
        "anyOf",
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "maxContains",
        "minContains",
        "not",
        "oneOf",
        "patternProperties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
)


def _unsafe_schema_keyword(value: object) -> str | None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            unsupported = _UNSUPPORTED_SCHEMA_KEYWORDS.intersection(item)
            if unsupported:
                return min(unsupported)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def _safe_pattern(validator, pattern: object, instance: object, schema: object):
    del validator, schema
    if not isinstance(instance, str) or not isinstance(pattern, str):
        return
    try:
        matched = regex.search(pattern, instance, timeout=0.05)
    except (TimeoutError, regex.error):
        matched = None
    if matched is None:
        yield ValidationError("The string does not match the required pattern")


_BoundedToolValidator = validators.extend(
    Draft202012Validator,
    validators={"pattern": _safe_pattern},
)


def _validate_tool_schema(schema: Mapping[str, object]) -> None:
    if _unsafe_schema_keyword(schema) is not None:
        raise LocalRuntimeError("complex_tool_schema_unsupported")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise LocalRuntimeError("invalid_tool_schema") from None
    except Exception:
        raise LocalRuntimeError("tool_schema_validation_failed") from None


def _validate_tool_arguments(arguments: object, schema: Mapping[str, object]) -> None:
    """Validate one call without resolving an external schema reference."""

    _validate_tool_schema(schema)
    try:
        _BoundedToolValidator(schema).validate(arguments)
    except ValidationError:
        raise LocalRuntimeError("tool_arguments_schema_mismatch") from None
    except Exception:
        raise LocalRuntimeError("tool_schema_validation_failed") from None


def prepare_local_request(
    envelope: RequestEnvelope,
    *,
    model: str,
    max_tokens_cap: int = 32_768,
    enable_thinking: bool = False,
) -> LocalEngineRequest:
    """Map one validated Agent IR request to the local MLX engine port."""

    request = envelope.request
    if type(model) is not str or not model:
        raise LocalRuntimeError("invalid_model")
    if type(max_tokens_cap) is not int or max_tokens_cap < 1:
        raise LocalRuntimeError("invalid_max_tokens_cap")
    if type(enable_thinking) is not bool:
        raise LocalRuntimeError("invalid_thinking_mode")
    if any(tool.strict is True for tool in request.tools):
        raise LocalRuntimeError("strict_tool_schema_unsupported")
    for tool in request.tools:
        _validate_tool_schema(tool.input_schema)
    tools = tuple(
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": cast(dict[str, JsonValue], tool.input_schema),
                **({"strict": tool.strict} if tool.strict is not None else {}),
            },
        }
        for tool in request.tools
    )
    generation = request.generation
    reasoning_effort = generation.reasoning_effort if generation is not None else None
    instructions: list[Mapping[str, object]] = []
    for index, instruction in enumerate(request.instructions):
        if instruction.order != index:
            raise LocalRuntimeError("invalid_instruction_order")
        if not instruction.source_location.startswith("/messages/"):
            instructions.append(_instruction_message(instruction))
    messages = [*instructions]
    for message in request.messages:
        messages.extend(_engine_messages(message))
    requested_max_tokens = generation.max_output_tokens if generation else None
    if requested_max_tokens is not None and requested_max_tokens > max_tokens_cap:
        raise LocalRuntimeError("max_tokens_exceeded")
    return LocalEngineRequest(
        model=model,
        messages=tuple(messages),
        tools=tools,
        temperature=(float(generation.temperature) if generation and generation.temperature is not None else None),
        top_p=(float(generation.top_p) if generation and generation.top_p is not None else None),
        max_tokens=requested_max_tokens or max_tokens_cap,
        stop=(tuple(generation.stop) if generation and generation.stop is not None else None),
        seed=(generation.seed if generation else None),
        enable_thinking=(enable_thinking or (not tools and reasoning_effort != "none")),
    )


def execute_local_request(
    envelope: RequestEnvelope,
    *,
    model: str,
    generate: LocalGenerator,
    profile: NormalizationProfile | None,
    terminal_reasons: TerminalReasons,
    output_id: str | None = None,
    call_id_factory: Callable[[], str] = new_call_id,
    output_id_factory: Callable[[], str] = new_output_id,
    parallel_group_factory: Callable[[], str] = new_parallel_group_id,
    limits: ToolOutputLimits = ToolOutputLimits(),
    sequence_start: int = 0,
    max_tokens_cap: int = 32_768,
    enable_thinking: bool = False,
    parallel_tool_calls: bool = True,
) -> LocalExecution:
    """Run one local generation and return a complete Agent IR output."""

    engine_request = prepare_local_request(
        envelope,
        model=model,
        max_tokens_cap=max_tokens_cap,
        enable_thinking=enable_thinking,
    )
    if type(parallel_tool_calls) is not bool:
        raise LocalRuntimeError("invalid_parallel_tool_calls")
    if engine_request.tools and profile is None:
        raise LocalRuntimeError("tool_profile_required")
    try:
        generation = generate(engine_request)
    except LocalRuntimeError:
        raise
    except Exception:
        raise LocalRuntimeError("generation_failed") from None
    if not isinstance(generation, LocalGeneration):
        raise LocalRuntimeError("invalid_generation")

    if engine_request.tools:
        assert profile is not None
        profile_contract = get_tool_profile_contract(profile)
        if profile_contract is None:
            raise LocalRuntimeError("tool_profile_required")
        normalized = normalize_tool_output(
            generation.text,
            profile=profile,
            limits=limits,
            repair_policy=profile_contract.repair_policy,
        )
    else:
        normalized = None

    request_id = str(envelope.request_id)
    selected_output_id = output_id or output_id_factory()
    if type(selected_output_id) is not str or not selected_output_id:
        raise LocalRuntimeError("invalid_output_id")
    policy = _policy(envelope.sensitivity, model)
    events: list[AgentEvent] = []
    calls: list[CallReference] = []
    source_call_ids: dict[str, str] = {}
    if type(sequence_start) is not int or sequence_start < 0:
        raise LocalRuntimeError("invalid_sequence_start")
    sequence = sequence_start
    tool_calls = () if normalized is None else normalized.tool_calls
    remaining_text = generation.text if normalized is None else normalized.remaining_text

    if tool_calls and remaining_text.strip():
        raise LocalRuntimeError("mixed_text_and_tool_calls")
    if len(tool_calls) > 1 and not parallel_tool_calls:
        raise LocalRuntimeError("parallel_tool_calls_disabled")
    if tool_calls and envelope.request.tool_choice == "none":
        raise LocalRuntimeError("tool_call_forbidden")
    if not tool_calls and envelope.request.tool_choice == "required":
        raise LocalRuntimeError("required_tool_missing")

    named_choice = envelope.request.tool_choice
    allowed_tools = {tool.name: tool for tool in envelope.request.tools}
    parallel_group_id = parallel_group_factory() if len(tool_calls) > 1 else None
    for index, tool_call in enumerate(tool_calls):
        definition = allowed_tools.get(tool_call.name)
        if definition is None:
            raise LocalRuntimeError("unknown_tool")
        if not isinstance(named_choice, str) and named_choice is not None and tool_call.name != named_choice.name:
            raise LocalRuntimeError("wrong_named_tool")
        call_id = call_id_factory()
        if type(call_id) is not str or not call_id:
            raise LocalRuntimeError("invalid_call_id")
        if any(reference.call_id == call_id for reference in calls):
            raise LocalRuntimeError("duplicate_call_id")
        if tool_call.call_id is not None:
            source_call_ids[call_id] = tool_call.call_id
        _validate_tool_arguments(tool_call.arguments_json, definition.input_schema)
        base: dict[str, object] = {
            "request_id": request_id,
            "choice_index": 0,
            "output_id": selected_output_id,
            "tool_call_index": index,
            "call_id": call_id,
            **policy,
        }
        if parallel_group_id is not None:
            base["parallel_group_id"] = parallel_group_id
        events.append(
            ToolCallStartedEvent.model_validate(
                {**base, "sequence": sequence, "type": "tool_call.started", "name": tool_call.name}
            )
        )
        sequence += 1
        events.append(
            ToolCallArgumentsDeltaEvent.model_validate(
                {
                    **base,
                    "sequence": sequence,
                    "type": "tool_call.arguments.delta",
                    "delta": tool_call.arguments_raw,
                }
            )
        )
        sequence += 1
        completed: dict[str, object] = {
            **base,
            "sequence": sequence,
            "type": "tool_call.completed",
            "name": tool_call.name,
            "arguments_raw": tool_call.arguments_raw,
        }
        if tool_call.arguments_json is not None:
            completed["arguments_json"] = tool_call.arguments_json
        if tool_call.repair is not None:
            completed["extensions"] = {
                "ppmlx.tool_argument_repair": {
                    "policy": tool_call.repair.policy.value,
                    "kind": tool_call.repair.kind.value,
                    "profile": tool_call.repair.profile,
                }
            }
        events.append(ToolCallCompletedEvent.model_validate(completed))
        sequence += 1
        calls.append(
            CallReference(
                call_id=call_id,
                name=tool_call.name,
                choice_index=0,
                output_id=selected_output_id,
                tool_call_index=index,
                parallel_group_id=parallel_group_id,
            )
        )

    if not tool_calls:
        events.append(
            ContentStartedEvent.model_validate(
                {
                    "request_id": request_id,
                    "sequence": sequence,
                    "type": "content.started",
                    "choice_index": 0,
                    "output_id": selected_output_id,
                    "content_index": 0,
                    "content_type": "text",
                    **policy,
                }
            )
        )
        sequence += 1
        events.append(
            ContentDeltaEvent.model_validate(
                {
                    "request_id": request_id,
                    "sequence": sequence,
                    "type": "content.delta",
                    "choice_index": 0,
                    "output_id": selected_output_id,
                    "content_index": 0,
                    "delta": remaining_text,
                    **policy,
                }
            )
        )
        sequence += 1
        events.append(
            ContentCompletedEvent.model_validate(
                {
                    "request_id": request_id,
                    "sequence": sequence,
                    "type": "content.completed",
                    "choice_index": 0,
                    "output_id": selected_output_id,
                    "content_index": 0,
                    "content": TextBlock.model_validate(
                        {"type": "text", "text": remaining_text, **policy}
                    ),
                    **policy,
                }
            )
        )
        sequence += 1

    usage = Usage(
        source=UsageSource.PPMLX_ESTIMATE,
        input_tokens=generation.prompt_tokens,
        output_tokens=generation.completion_tokens,
        total_tokens=generation.prompt_tokens + generation.completion_tokens,
    )
    events.append(
        ResponseCompletedEvent.model_validate(
            {
                "request_id": request_id,
                "sequence": sequence,
                "type": "response.completed",
                "choice_index": 0,
                "output_id": selected_output_id,
                "finish_reason": (
                    terminal_reasons.tool_calls if tool_calls else terminal_reasons.text
                ),
                "usage": usage,
                **policy,
            }
        )
    )
    return LocalExecution(
        events=tuple(events),
        calls=tuple(calls),
        source_call_ids=MappingProxyType(source_call_ids),
    )


__all__ = [
    "LocalEngineRequest",
    "LocalExecution",
    "LocalGeneration",
    "LocalGenerator",
    "LocalRuntimeError",
    "TerminalReasons",
    "execute_local_request",
    "prepare_local_request",
]
