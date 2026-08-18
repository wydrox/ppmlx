"""Pure OpenAI Responses protocol adapter."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, cast

from pydantic import JsonValue, ValidationError

from ppmlx.agent_ir import (
    AgentEvent,
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    Generation,
    Instruction,
    Message,
    Origin,
    Provenance,
    Request,
    RequestEnvelope,
    ResponseCompletedEvent,
    Sensitivity,
    TextBlock,
    ToolCallArgumentsDeltaEvent,
    ToolCallBlock,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolDefinition,
    ToolResultBlock,
    ToolResultEvent,
    Usage,
)
from ppmlx.protocols.base import (
    CallReference,
    DecodeContext,
    DecodedRequest,
    EncodeContext,
    ProtocolAdapterError,
    ProtocolCapabilities,
    safe_adapter_boundary,
)
from ppmlx.protocols.json import ensure_safe_evidence, parse_json_object
from ppmlx.protocols.sse import SSEFrame, encode_sse


_PROTOCOL = "openai-responses"
_REQUEST_FIELDS = {
    "client_metadata",
    "include",
    "input",
    "instructions",
    "max_output_tokens",
    "model",
    "parallel_tool_calls",
    "prompt_cache_key",
    "reasoning",
    "store",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
}


class _PolicyFields(TypedDict):
    sensitivity: Sensitivity
    provenance: Provenance


def _error(code: str, field: str | None = None) -> ProtocolAdapterError:
    return ProtocolAdapterError(protocol=_PROTOCOL, code=code, field=field)


def _require_exact_fields(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    field: str,
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise _error("missing_field", f"{field}.{sorted(missing)[0]}")
    if extra:
        raise _error("unsupported_field", field)


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise _error("invalid_field", field)
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise _error("invalid_field", field)
    return value


def _number(value: object, field: str) -> int | float:
    if type(value) not in {int, float}:
        raise _error("invalid_field", field)
    return cast(int | float, value)


def _object(value: object, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _error("invalid_field", field)
    return cast(dict[str, Any], value)


def _array(value: object, field: str) -> list[Any]:
    if type(value) is not list:
        raise _error("invalid_field", field)
    return cast(list[Any], value)


def _provenance(context: DecodeContext, origin: Origin | None = None) -> Provenance:
    configured = context.policy.provenance
    if origin is None or configured.origin == origin:
        return configured
    return Provenance(origin=origin, trust=configured.trust)


def _policy_fields(context: DecodeContext, origin: Origin | None = None) -> _PolicyFields:
    return {
        "sensitivity": context.policy.sensitivity,
        "provenance": _provenance(context, origin),
    }


def _extensions(
    context: DecodeContext,
    key: str,
    evidence: JsonValue,
) -> dict[str, JsonValue]:
    if not context.policy.include_native_evidence:
        return {}
    ensure_safe_evidence(evidence, protocol=_PROTOCOL)
    return {key: evidence}


def _parse_arguments(value: object, *, field: str, context: DecodeContext) -> tuple[str, dict[str, JsonValue]]:
    raw = _string(value, field, allow_empty=True)
    if len(raw.encode("utf-8")) > context.limits.max_arguments_bytes:
        raise _error("arguments_too_large", field)
    try:
        parsed = parse_json_object(raw, protocol=_PROTOCOL, limits=context.limits)
    except ProtocolAdapterError:
        raise _error("invalid_arguments", field) from None
    return raw, cast(dict[str, JsonValue], parsed)


def _decode_instruction(native: dict[str, Any], context: DecodeContext) -> list[Instruction]:
    if "instructions" not in native:
        return []
    text = _string(native["instructions"], "instructions", allow_empty=True)
    return [
        Instruction(
            source_role="developer",
            source_location="/instructions",
            order=0,
            content=[TextBlock(type="text", text=text, **_policy_fields(context))],
            **_policy_fields(context),
        )
    ]


def _decode_message(item: dict[str, Any], *, index: int, context: DecodeContext) -> Message:
    field = f"input[{index}]"
    _require_exact_fields(
        item,
        required={"type", "role", "content"},
        optional={"id"},
        field=field,
    )
    if item["type"] != "message":
        raise _error("unsupported_item_type", f"{field}.type")
    role = _string(item["role"], f"{field}.role")
    if role not in {"system", "developer", "user", "assistant"}:
        raise _error("unsupported_role", f"{field}.role")
    content = _array(item["content"], f"{field}.content")
    if len(content) > context.limits.max_blocks:
        raise _error("too_many_blocks", f"{field}.content")
    blocks: list[TextBlock] = []
    for content_index, raw_block in enumerate(content):
        block_field = f"{field}.content[{content_index}]"
        block = _object(raw_block, block_field)
        _require_exact_fields(
            block,
            required={"type", "text"},
            optional=set(),
            field=block_field,
        )
        if block["type"] != "input_text":
            raise _error("unsupported_content_type", f"{block_field}.type")
        blocks.append(
            TextBlock(
                type="text",
                text=_string(block["text"], f"{block_field}.text", allow_empty=True),
                **_policy_fields(context),
            )
        )
    values: dict[str, object] = {
        "role": role,
        "content": blocks,
        **_policy_fields(context),
    }
    if "id" in item:
        values["id"] = _string(item["id"], f"{field}.id")
    native_fields = _extensions(
        context,
        "openai-responses.native_message_fields",
        {"type": "message"},
    )
    if native_fields:
        values["extensions"] = native_fields
    try:
        return Message.model_validate(values)
    except ValidationError:
        raise _error("invalid_message", field) from None


def _resolve_call_reference(
    *,
    item: dict[str, Any],
    index: int,
    call_index: int,
    context: DecodeContext,
) -> tuple[Message, CallReference]:
    field = f"input[{index}]"
    _require_exact_fields(
        item,
        required={"type", "id", "call_id", "name", "arguments"},
        optional=set(),
        field=field,
    )
    item_id = _string(item["id"], f"{field}.id")
    call_id = _string(item["call_id"], f"{field}.call_id")
    name = _string(item["name"], f"{field}.name")
    arguments_raw, arguments_json = _parse_arguments(
        item["arguments"],
        field=f"{field}.arguments",
        context=context,
    )
    supplied = context.prior_calls.get(call_id)
    if supplied is not None:
        if supplied.name != name or supplied.output_id != item_id:
            raise _error("broken_call_link", field)
        reference = supplied
    else:
        reference = CallReference(
            call_id=call_id,
            name=name,
            choice_index=0,
            output_id=item_id,
            tool_call_index=call_index,
        )
    block_values: dict[str, object] = {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "arguments_raw": arguments_raw,
        "arguments_json": arguments_json,
        **_policy_fields(context, Origin.PROVIDER),
    }
    block_extensions = _extensions(
        context,
        "openai-responses.native_function_call",
        {"type": "function_call", "id": item_id},
    )
    if block_extensions:
        block_values["extensions"] = block_extensions
    message = Message(
        id=item_id,
        role="assistant",
        content=[ToolCallBlock.model_validate(block_values)],
        **_policy_fields(context, Origin.PROVIDER),
    )
    return message, reference


def _decode_tool_result(
    *,
    item: dict[str, Any],
    index: int,
    calls: Mapping[str, CallReference],
    context: DecodeContext,
    sequence: int,
) -> tuple[Message, ToolResultEvent]:
    field = f"input[{index}]"
    _require_exact_fields(
        item,
        required={"type", "call_id", "output"},
        optional={"id"},
        field=field,
    )
    call_id = _string(item["call_id"], f"{field}.call_id")
    reference = calls.get(call_id) or context.prior_calls.get(call_id)
    if reference is None:
        raise _error("broken_call_link", f"{field}.call_id")
    output = _string(item["output"], f"{field}.output", allow_empty=True)
    supplied_output_id = context.result_output_ids.get(call_id)
    if "id" in item:
        output_id = _string(item["id"], f"{field}.id")
        if supplied_output_id is not None and supplied_output_id != output_id:
            raise _error("broken_result_link", f"{field}.id")
    elif supplied_output_id is not None:
        output_id = supplied_output_id
    else:
        raise _error("missing_result_output_id", f"{field}.id")
    evidence = dict(item)
    text_block = TextBlock(
        type="text",
        text=output,
        **_policy_fields(context, Origin.TOOL),
    )
    result_values: dict[str, object] = {
        "type": "tool_result",
        "call_id": call_id,
        "content": [text_block],
        "is_error": False,
        **_policy_fields(context, Origin.TOOL),
    }
    result_extensions = _extensions(
        context,
        "openai-responses.native_tool_result",
        cast(JsonValue, evidence),
    )
    if result_extensions:
        result_values["extensions"] = result_extensions
    result_block = ToolResultBlock.model_validate(result_values)
    message = Message(
        id=output_id,
        role="tool",
        content=[result_block],
        **_policy_fields(context, Origin.TOOL),
    )
    event_values: dict[str, object] = {
        "request_id": context.request_id,
        "sequence": sequence,
        "type": "tool_result",
        "choice_index": reference.choice_index,
        "output_id": output_id,
        "tool_call_index": reference.tool_call_index,
        "call_id": call_id,
        "content": [text_block],
        "is_error": False,
        **_policy_fields(context, Origin.TOOL),
    }
    if reference.parallel_group_id is not None:
        event_values["parallel_group_id"] = reference.parallel_group_id
    if result_extensions:
        event_values["extensions"] = result_extensions
    return message, ToolResultEvent.model_validate(event_values)


def _decode_tools(native: dict[str, Any], context: DecodeContext) -> list[ToolDefinition]:
    if "tools" not in native:
        return []
    tools = _array(native["tools"], "tools")
    if len(tools) > context.limits.max_tools:
        raise _error("too_many_tools", "tools")
    decoded: list[ToolDefinition] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(tools):
        field = f"tools[{index}]"
        tool = _object(raw_tool, field)
        _require_exact_fields(
            tool,
            required={"type", "name", "description", "parameters"},
            optional={"strict"},
            field=field,
        )
        if tool["type"] != "function":
            raise _error("unsupported_tool_type", f"{field}.type")
        name = _string(tool["name"], f"{field}.name")
        if name in names:
            raise _error("duplicate_tool", f"{field}.name")
        names.add(name)
        values: dict[str, object] = {
            "name": name,
            "description": _string(tool["description"], f"{field}.description", allow_empty=True),
            "input_schema": _object(tool["parameters"], f"{field}.parameters"),
            **_policy_fields(context),
        }
        if "strict" in tool:
            values["strict"] = _boolean(tool["strict"], f"{field}.strict")
        native_fields = _extensions(
            context,
            "openai-responses.native_tool_fields",
            {"type": "function"},
        )
        if native_fields:
            values["extensions"] = native_fields
        try:
            decoded.append(ToolDefinition.model_validate(values))
        except ValidationError:
            raise _error("invalid_tool", field) from None
    return decoded


def _decode_tool_choice(value: object) -> object:
    if type(value) is str:
        if value not in {"auto", "none", "required"}:
            raise _error("unsupported_tool_choice", "tool_choice")
        return value
    choice = _object(value, "tool_choice")
    _require_exact_fields(choice, required={"type", "name"}, optional=set(), field="tool_choice")
    if choice["type"] != "function":
        raise _error("unsupported_tool_choice", "tool_choice.type")
    return {"type": "tool", "name": _string(choice["name"], "tool_choice.name")}


def _decode_generation(native: dict[str, Any], context: DecodeContext) -> Generation | None:
    values: dict[str, object] = {}
    if "temperature" in native:
        values["temperature"] = _number(native["temperature"], "temperature")
    if "top_p" in native:
        values["top_p"] = _number(native["top_p"], "top_p")
    if "max_output_tokens" in native:
        maximum = native["max_output_tokens"]
        if type(maximum) is not int or maximum < 1:
            raise _error("invalid_field", "max_output_tokens")
        values["max_output_tokens"] = maximum
    if "reasoning" in native:
        reasoning = _object(native["reasoning"], "reasoning")
        _require_exact_fields(reasoning, required=set(), optional={"effort", "summary"}, field="reasoning")
        if "effort" in reasoning:
            values["reasoning_effort"] = _string(reasoning["effort"], "reasoning.effort")
        values["extensions"] = {
            "openai-responses.generation": {"reasoning": cast(JsonValue, reasoning)}
        }
    if not values:
        return None
    try:
        return Generation.model_validate(values)
    except ValidationError:
        raise _error("invalid_generation") from None


def _decode_metadata(native: dict[str, Any]) -> dict[str, JsonValue] | None:
    values: dict[str, JsonValue] = {}
    if "client_metadata" in native:
        values["client_metadata"] = cast(JsonValue, _object(native["client_metadata"], "client_metadata"))
    if "prompt_cache_key" in native:
        values["prompt_cache_key"] = _string(native["prompt_cache_key"], "prompt_cache_key")
    return values or None


class OpenAIResponsesAdapter:
    """Convert strict Responses requests and Agent IR output streams."""

    protocol = _PROTOCOL
    capabilities = ProtocolCapabilities(
        request_features=frozenset(
            {
                "generation",
                "instructions",
                "metadata",
                "stream",
                "text",
                "tool_calls",
                "tool_choice",
                "tool_results",
                "tools",
            }
        ),
        response_features=frozenset({"stream", "text", "tool_calls", "usage"}),
        verified_harnesses=("codex-0.147.0",),
    )

    @safe_adapter_boundary
    def decode_request(
        self,
        value: str | bytes | bytearray | Mapping[str, object],
        *,
        context: DecodeContext,
    ) -> DecodedRequest:
        try:
            return self._decode_request(value, context=context)
        except ProtocolAdapterError:
            raise
        except (TypeError, ValueError, ValidationError):
            raise _error("invalid_request") from None

    def _decode_request(
        self,
        value: str | bytes | bytearray | Mapping[str, object],
        *,
        context: DecodeContext,
    ) -> DecodedRequest:
        if context.kind not in {"initial", "continuation"}:
            raise _error("invalid_request_kind")
        if context.kind == "initial" and context.parent_request_id is not None:
            raise _error("unexpected_parent_request_id")
        if context.kind == "continuation" and context.parent_request_id is None:
            raise _error("missing_parent_request_id")
        native = parse_json_object(value, protocol=_PROTOCOL, limits=context.limits)
        ensure_safe_evidence(native, protocol=_PROTOCOL)
        extra = native.keys() - _REQUEST_FIELDS
        if extra:
            raise _error("unsupported_field", "/")
        for required in ("model", "input"):
            if required not in native:
                raise _error("missing_field", required)

        raw_input = _array(native["input"], "input")
        if len(raw_input) > context.limits.max_blocks:
            raise _error("too_many_input_items", "input")
        messages: list[Message] = []
        calls: list[CallReference] = []
        calls_by_id: dict[str, CallReference] = {}
        tool_results: list[ToolResultEvent] = []
        result_call_ids: set[str] = set()
        item_ids: set[str] = set()
        for index, raw_item in enumerate(raw_input):
            item = _object(raw_item, f"input[{index}]")
            item_type = item.get("type")
            if item_type == "message":
                message = _decode_message(item, index=index, context=context)
                if message.id is not None:
                    item_id = str(message.id)
                    if item_id in item_ids:
                        raise _error("duplicate_item_id", f"input[{index}].id")
                    item_ids.add(item_id)
                messages.append(message)
            elif item_type == "function_call":
                if context.kind != "continuation":
                    raise _error("unexpected_function_call", f"input[{index}]")
                message, reference = _resolve_call_reference(
                    item=item,
                    index=index,
                    call_index=len(calls),
                    context=context,
                )
                if reference.call_id in calls_by_id:
                    raise _error("duplicate_call_id", f"input[{index}].call_id")
                if reference.output_id in item_ids:
                    raise _error("duplicate_item_id", f"input[{index}].id")
                item_ids.add(reference.output_id)
                calls_by_id[reference.call_id] = reference
                calls.append(reference)
                messages.append(message)
            elif item_type == "function_call_output":
                if context.kind != "continuation":
                    raise _error("unexpected_tool_result", f"input[{index}]")
                message, event = _decode_tool_result(
                    item=item,
                    index=index,
                    calls=calls_by_id,
                    context=context,
                    sequence=context.sequence_start + len(tool_results),
                )
                call_id = str(event.call_id)
                if call_id in result_call_ids:
                    raise _error("duplicate_tool_result", f"input[{index}].call_id")
                result_output_id = str(event.output_id)
                if result_output_id in item_ids:
                    raise _error("duplicate_item_id", f"input[{index}].id")
                item_ids.add(result_output_id)
                result_call_ids.add(call_id)
                messages.append(message)
                tool_results.append(event)
            else:
                raise _error("unsupported_item_type", f"input[{index}].type")

        if sum(len(message.content) for message in messages) > context.limits.max_blocks:
            raise _error("too_many_blocks", "input")

        request_values: dict[str, object] = {
            "model": _string(native["model"], "model"),
            "instructions": _decode_instruction(native, context),
            "messages": messages,
            "tools": _decode_tools(native, context),
            **_policy_fields(context),
        }
        if "tool_choice" in native:
            request_values["tool_choice"] = _decode_tool_choice(native["tool_choice"])
        generation = _decode_generation(native, context)
        if generation is not None:
            request_values["generation"] = generation
        if "stream" in native:
            request_values["stream"] = _boolean(native["stream"], "stream")
        metadata = _decode_metadata(native)
        if metadata is not None:
            request_values["metadata"] = metadata
        request_extensions: dict[str, JsonValue] = {}
        options: dict[str, JsonValue] = {}
        if "include" in native:
            include = _array(native["include"], "include")
            normalized_include = [
                _string(item, f"include[{index}]")
                for index, item in enumerate(include)
            ]
            if len(set(normalized_include)) != len(normalized_include):
                raise _error("duplicate_include", "include")
            options["include"] = cast(JsonValue, normalized_include)
        for key in ("parallel_tool_calls", "store"):
            if key in native:
                options[key] = _boolean(native[key], key)
        if options:
            request_extensions["openai-responses.request_options"] = options
        request_extensions.update(
            _extensions(
                context,
                "openai-responses.native_request",
                cast(JsonValue, native),
            )
        )
        if request_extensions:
            request_values["extensions"] = request_extensions
        envelope_values: dict[str, object] = {
            "request_id": context.request_id,
            "kind": context.kind,
            "request": Request.model_validate(request_values),
            **_policy_fields(context),
        }
        if context.parent_request_id is not None:
            envelope_values["parent_request_id"] = context.parent_request_id
        try:
            envelope = RequestEnvelope.model_validate(envelope_values)
        except ValidationError:
            raise _error("invalid_request") from None
        return DecodedRequest(
            request=envelope,
            tool_results=tuple(tool_results),
            calls=tuple(calls),
        )

    @safe_adapter_boundary
    def encode_stream(
        self,
        events: Sequence[AgentEvent],
        *,
        context: EncodeContext,
    ) -> str:
        try:
            return _encode_stream(events, context=context)
        except ProtocolAdapterError:
            raise
        except (TypeError, ValueError):
            raise _error("invalid_encode_input") from None


def _response(
    *,
    context: EncodeContext,
    status: Literal["in_progress", "completed"],
    output: list[dict[str, JsonValue]],
    usage: dict[str, JsonValue] | None,
    metadata: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "id": context.response_id or "resp_ppmlx",
        "object": "response",
        "created_at": context.created_at,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": context.model,
        "output": cast(JsonValue, output),
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": usage,
        "user": None,
        "metadata": cast(JsonValue, metadata),
    }


def _native_usage(value: Usage | None) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    input_tokens = value.input_tokens or 0
    output_tokens = value.output_tokens or 0
    total_tokens = value.total_tokens
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    elif value.input_tokens is not None and value.output_tokens is not None and total_tokens != input_tokens + output_tokens:
        raise _error("invalid_usage")
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }


def _encode_stream(events: Sequence[AgentEvent], *, context: EncodeContext) -> str:
    if not events:
        raise _error("empty_event_stream")
    if len(events) > context.limits.max_events:
        raise _error("too_many_events")
    try:
        metadata = cast(
            dict[str, JsonValue],
            parse_json_object(
                context.metadata,
                protocol=_PROTOCOL,
                limits=context.limits,
            ),
        )
    except ProtocolAdapterError:
        raise _error("invalid_encode_metadata") from None
    ensure_safe_evidence(metadata, protocol=_PROTOCOL)
    if not context.model or type(context.model) is not str:
        raise _error("invalid_encode_model")
    if len(context.model.encode("utf-8")) > context.limits.max_string_bytes:
        raise _error("encode_model_too_large")
    if len(context.model.encode("utf-8")) > context.limits.max_sse_stream_bytes:
        raise _error("sse_stream_too_large")
    if type(context.created_at) is not int or context.created_at < 0:
        raise _error("invalid_created_at")
    if context.response_id is not None and (type(context.response_id) is not str or not context.response_id):
        raise _error("invalid_response_id")
    if (
        context.response_id is not None
        and len(context.response_id.encode("utf-8")) > context.limits.max_string_bytes
    ):
        raise _error("response_id_too_large")
    if type(context.parallel_tool_calls) is not bool:
        raise _error("invalid_parallel_tool_calls")

    frames: list[SSEFrame] = []
    sequence_number = 0

    def emit(data: dict[str, JsonValue]) -> None:
        nonlocal sequence_number
        payload = dict(data)
        payload["sequence_number"] = sequence_number
        event_type = cast(str, payload["type"])
        frames.append(SSEFrame(event=event_type, data=payload))
        sequence_number += 1

    emit(
        {
            "type": "response.created",
            "response": _response(
                context=context,
                status="in_progress",
                output=[],
                usage=None,
                metadata=metadata,
            ),
        }
    )
    first_request_id = str(events[0].request_id) if isinstance(events[0], tuple(_SUPPORTED_EVENTS)) else ""
    last_ir_sequence = -1
    states: dict[str, dict[str, Any]] = {}
    output_indexes: dict[int, str] = {}
    call_ids: set[str] = set()
    completed_items: list[tuple[int, dict[str, JsonValue]]] = []
    terminals: list[ResponseCompletedEvent] = []

    for event in events:
        if not isinstance(event, tuple(_SUPPORTED_EVENTS)):
            raise _error("unsupported_event_type")
        if str(event.request_id) != first_request_id:
            raise _error("mixed_request_ids")
        if event.sequence <= last_ir_sequence:
            raise _error("invalid_event_sequence")
        last_ir_sequence = event.sequence
        output_id = str(event.output_id)

        if isinstance(event, ToolCallStartedEvent):
            if output_id in states:
                raise _error("duplicate_output_id")
            if str(event.call_id) in call_ids:
                raise _error("duplicate_call_id")
            call_ids.add(str(event.call_id))
            output_index = event.tool_call_index
            if output_index in output_indexes:
                raise _error("duplicate_output_index")
            output_indexes[output_index] = output_id
            item: dict[str, JsonValue] = {
                "id": output_id,
                "type": "function_call",
                "status": "in_progress",
                "arguments": "",
                "call_id": str(event.call_id),
                "name": event.name,
            }
            states[output_id] = {
                "kind": "tool",
                "output_index": output_index,
                "call_id": str(event.call_id),
                "name": event.name,
                "choice_index": event.choice_index,
                "parallel_group_id": event.parallel_group_id,
                "arguments": [],
                "argument_bytes": 0,
                "completed": False,
                "terminal": False,
            }
            emit(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                }
            )
            continue

        if isinstance(event, ToolCallArgumentsDeltaEvent):
            state = states.get(output_id)
            if (
                state is None
                or state["kind"] != "tool"
                or state["completed"]
                or state["call_id"] != str(event.call_id)
                or state["choice_index"] != event.choice_index
                or state["parallel_group_id"] != event.parallel_group_id
            ):
                raise _error("invalid_tool_lifecycle")
            argument_bytes = cast(int, state["argument_bytes"]) + len(event.delta.encode("utf-8"))
            if argument_bytes > context.limits.max_arguments_bytes:
                raise _error("arguments_too_large")
            state["argument_bytes"] = argument_bytes
            cast(list[str], state["arguments"]).append(event.delta)
            emit(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": output_id,
                    "output_index": state["output_index"],
                    "delta": event.delta,
                }
            )
            continue

        if isinstance(event, ToolCallCompletedEvent):
            state = states.get(output_id)
            arguments = "" if state is None else "".join(cast(list[str], state["arguments"]))
            if (
                state is None
                or state["kind"] != "tool"
                or state["completed"]
                or state["call_id"] != str(event.call_id)
                or state["name"] != event.name
                or state["choice_index"] != event.choice_index
                or state["parallel_group_id"] != event.parallel_group_id
                or (arguments and arguments != event.arguments_raw)
            ):
                raise _error("invalid_tool_lifecycle")
            if len(event.arguments_raw.encode("utf-8")) > context.limits.max_arguments_bytes:
                raise _error("arguments_too_large")
            try:
                parse_json_object(
                    event.arguments_raw,
                    protocol=_PROTOCOL,
                    limits=context.limits,
                )
            except ProtocolAdapterError:
                raise _error("invalid_arguments") from None
            state["arguments"] = event.arguments_raw
            state["completed"] = True
            emit(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": output_id,
                    "output_index": state["output_index"],
                    "arguments": event.arguments_raw,
                }
            )
            item = {
                "id": output_id,
                "type": "function_call",
                "status": "completed",
                "arguments": event.arguments_raw,
                "call_id": state["call_id"],
                "name": state["name"],
            }
            emit(
                {
                    "type": "response.output_item.done",
                    "output_index": state["output_index"],
                    "item": item,
                }
            )
            completed_items.append((state["output_index"], item))
            continue

        if isinstance(event, ContentStartedEvent):
            if event.content_type != "text" or output_id in states:
                raise _error("unsupported_content_type")
            output_index = event.choice_index
            if output_index in output_indexes:
                raise _error("duplicate_output_index")
            output_indexes[output_index] = output_id
            states[output_id] = {
                "kind": "text",
                "output_index": output_index,
                "content_index": event.content_index,
                "choice_index": event.choice_index,
                "text": [],
                "text_bytes": 0,
                "completed": False,
                "terminal": False,
            }
            emit(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": output_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }
            )
            emit(
                {
                    "type": "response.content_part.added",
                    "item_id": output_id,
                    "output_index": output_index,
                    "content_index": event.content_index,
                    "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""},
                }
            )
            continue

        if isinstance(event, ContentDeltaEvent):
            state = states.get(output_id)
            if (
                state is None
                or state["kind"] != "text"
                or state["completed"]
                or state["content_index"] != event.content_index
                or state["choice_index"] != event.choice_index
            ):
                raise _error("invalid_content_lifecycle")
            text_bytes = cast(int, state["text_bytes"]) + len(event.delta.encode("utf-8"))
            if text_bytes > context.limits.max_string_bytes:
                raise _error("content_too_large")
            state["text_bytes"] = text_bytes
            cast(list[str], state["text"]).append(event.delta)
            emit(
                {
                    "type": "response.output_text.delta",
                    "item_id": output_id,
                    "output_index": state["output_index"],
                    "content_index": event.content_index,
                    "delta": event.delta,
                    "logprobs": [],
                }
            )
            continue

        if isinstance(event, ContentCompletedEvent):
            state = states.get(output_id)
            text = "" if state is None else "".join(cast(list[str], state["text"]))
            if (
                isinstance(event.content, TextBlock)
                and len(event.content.text.encode("utf-8")) > context.limits.max_string_bytes
            ):
                raise _error("content_too_large")
            if (
                state is None
                or state["kind"] != "text"
                or state["completed"]
                or state["content_index"] != event.content_index
                or state["choice_index"] != event.choice_index
                or not isinstance(event.content, TextBlock)
                or (text and text != event.content.text)
            ):
                raise _error("invalid_content_lifecycle")
            state["text"] = event.content.text
            state["completed"] = True
            part: dict[str, JsonValue] = {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": event.content.text,
            }
            emit(
                {
                    "type": "response.output_text.done",
                    "item_id": output_id,
                    "output_index": state["output_index"],
                    "content_index": event.content_index,
                    "text": event.content.text,
                    "logprobs": [],
                }
            )
            emit(
                {
                    "type": "response.content_part.done",
                    "item_id": output_id,
                    "output_index": state["output_index"],
                    "content_index": event.content_index,
                    "part": part,
                }
            )
            continue

        if isinstance(event, ResponseCompletedEvent):
            state = states.get(output_id)
            if (
                state is None
                or not state["completed"]
                or state["terminal"]
                or state["choice_index"] != event.choice_index
                or event.finish_reason != "completed"
            ):
                raise _error("invalid_terminal_event")
            state["terminal"] = True
            terminals.append(event)
            if state["kind"] == "text":
                part = {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": state["text"],
                }
                item = {
                    "id": output_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [part],
                }
                emit(
                    {
                        "type": "response.output_item.done",
                        "output_index": state["output_index"],
                        "item": item,
                    }
                )
                completed_items.append((state["output_index"], item))

    if not states or any(not state["completed"] or not state["terminal"] for state in states.values()):
        raise _error("incomplete_event_stream")
    if len(terminals) != len(states):
        raise _error("invalid_terminal_count")
    indexes = sorted(output_indexes)
    if indexes != list(range(len(indexes))):
        raise _error("non_contiguous_output_indexes")
    usage_values = {_usage_key(event.usage) for event in terminals}
    if len(usage_values) != 1:
        raise _error("inconsistent_usage")
    completed_items.sort(key=lambda item: item[0])
    native_output = [item for _, item in completed_items]
    usage = _native_usage(terminals[-1].usage)
    emit(
        {
            "type": "response.completed",
            "response": _response(
                context=context,
                status="completed",
                output=native_output,
                usage=usage,
                metadata=metadata,
            ),
        }
    )
    frames.append(SSEFrame(event=None, data="[DONE]"))
    return encode_sse(frames, protocol=_PROTOCOL, limits=context.limits)


def _usage_key(value: Usage | None) -> tuple[object, ...]:
    if value is None:
        return (None,)
    return (
        value.source,
        value.input_tokens,
        value.output_tokens,
        value.total_tokens,
    )


_SUPPORTED_EVENTS = (
    ContentStartedEvent,
    ContentDeltaEvent,
    ContentCompletedEvent,
    ToolCallStartedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ResponseCompletedEvent,
)


openai_responses_adapter = OpenAIResponsesAdapter()


__all__ = ["OpenAIResponsesAdapter", "openai_responses_adapter"]
