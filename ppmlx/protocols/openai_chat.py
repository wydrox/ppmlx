"""Pure OpenAI Chat Completions protocol adapter."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import ValidationError

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


_PROTOCOL = "openai-chat"
_KNOWN_REQUEST_FIELDS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
}
_REQUEST_OPTION_FIELDS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "service_tier",
    "store",
    "stream_options",
    "top_logprobs",
    "user",
}


def _error(code: str, field: str | None = None) -> ProtocolAdapterError:
    return ProtocolAdapterError(protocol=_PROTOCOL, code=code, field=field)


def _policy(context: DecodeContext) -> dict[str, object]:
    return {
        "sensitivity": context.policy.sensitivity,
        "provenance": context.policy.provenance,
    }


def _derived_policy(context: DecodeContext, origin: Origin) -> dict[str, object]:
    return {
        "sensitivity": context.policy.sensitivity,
        "provenance": Provenance(
            origin=origin,
            trust=context.policy.provenance.trust,
        ),
    }


def _with_evidence(
    base: dict[str, object],
    *,
    context: DecodeContext,
    namespace: str,
    evidence: object,
) -> dict[str, object]:
    if context.policy.include_native_evidence:
        ensure_safe_evidence(evidence, protocol=_PROTOCOL)
        base["extensions"] = {namespace: evidence}
    return base


def _require_string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise _error("invalid_shape", field)
    return value


def _require_object(value: object, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _error("invalid_shape", field)
    return cast(dict[str, Any], value)


def _require_list(value: object, field: str) -> list[Any]:
    if type(value) is not list:
        raise _error("invalid_shape", field)
    return cast(list[Any], value)


def _parse_arguments(value: str, *, context: DecodeContext) -> object | None:
    try:
        return parse_json_object(value, protocol=_PROTOCOL, limits=context.limits)
    except ProtocolAdapterError:
        return None


def _text_block(
    text: str,
    *,
    context: DecodeContext,
    origin: Origin | None = None,
    native_fields: dict[str, Any] | None = None,
) -> TextBlock:
    policy = _policy(context) if origin is None else _derived_policy(context, origin)
    if native_fields:
        ensure_safe_evidence(native_fields, protocol=_PROTOCOL)
        policy["extensions"] = {"openai-chat.block_options": native_fields}
        _with_evidence(
            policy,
            context=context,
            namespace="openai-chat.native_block",
            evidence=native_fields,
        )
    return TextBlock.model_validate({"type": "text", "text": text, **policy})


def _decode_text_content(
    content: object,
    *,
    context: DecodeContext,
    origin: Origin | None,
    field: str,
) -> list[TextBlock]:
    if type(content) is str:
        return [_text_block(cast(str, content), context=context, origin=origin)]
    blocks = _require_list(content, field)
    if len(blocks) > context.limits.max_blocks:
        raise _error("too_many_blocks", field)
    decoded: list[TextBlock] = []
    for index, item in enumerate(blocks):
        block_field = f"{field}/{index}"
        block = _require_object(item, block_field)
        if block.get("type") not in {"text", "input_text", "output_text"}:
            raise _error("unsupported_content", block_field)
        if set(block) != {"type", "text"}:
            raise _error("unsupported_content", block_field)
        text = _require_string(block.get("text"), f"{block_field}/text", allow_empty=True)
        decoded.append(
            _text_block(
                text,
                context=context,
                origin=origin,
            )
        )
    return decoded


def _decode_tool_call(
    value: object,
    *,
    context: DecodeContext,
    field: str,
) -> ToolCallBlock:
    native = _require_object(value, field)
    if set(native) - {"id", "type", "function"}:
        raise _error("unsupported_tool_call", field)
    if native.get("type") != "function":
        raise _error("unsupported_tool_call", f"{field}/type")
    call_id = _require_string(native.get("id"), f"{field}/id")
    function = _require_object(native.get("function"), f"{field}/function")
    if set(function) - {"name", "arguments"}:
        raise _error("unsupported_tool_call", f"{field}/function")
    name = _require_string(function.get("name"), f"{field}/function/name")
    arguments_raw = _require_string(
        function.get("arguments"),
        f"{field}/function/arguments",
        allow_empty=True,
    )
    if len(arguments_raw.encode("utf-8")) > context.limits.max_arguments_bytes:
        raise _error("arguments_too_large", f"{field}/function/arguments")
    block: dict[str, object] = {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "arguments_raw": arguments_raw,
        **_derived_policy(context, Origin.PROVIDER),
    }
    arguments_json = _parse_arguments(arguments_raw, context=context)
    if arguments_json is not None:
        block["arguments_json"] = arguments_json
    _with_evidence(
        block,
        context=context,
        namespace="openai-chat.native_tool_call",
        evidence={"type": "function"},
    )
    return ToolCallBlock.model_validate(block)


def _decode_tool_result(
    native: dict[str, Any],
    *,
    context: DecodeContext,
    field: str,
    prior_calls: Mapping[str, CallReference],
) -> tuple[ToolResultBlock, ToolResultEvent]:
    if set(native) - {"role", "content", "tool_call_id", "name"}:
        raise _error("unsupported_message", field)
    call_id = _require_string(native.get("tool_call_id"), f"{field}/tool_call_id")
    reference = prior_calls.get(call_id)
    if reference is None:
        raise _error("broken_call_link", f"{field}/tool_call_id")
    if "name" in native and native["name"] != reference.name:
        raise _error("broken_call_link", f"{field}/name")
    content_value = native.get("content")
    if content_value is None:
        raise _error("invalid_shape", f"{field}/content")
    content = _decode_text_content(
        content_value,
        context=context,
        origin=Origin.TOOL,
        field=f"{field}/content",
    )
    policy = _derived_policy(context, Origin.TOOL)
    evidence = dict(native)
    block_data: dict[str, object] = {
        "type": "tool_result",
        "call_id": call_id,
        "content": content,
        "is_error": False,
        **policy,
    }
    _with_evidence(
        block_data,
        context=context,
        namespace="openai-chat.native_tool_result",
        evidence=evidence,
    )
    output_id = context.result_output_ids.get(call_id)
    if not output_id:
        raise _error("missing_result_output_id", f"{field}/tool_call_id")
    event_data: dict[str, object] = {
        "request_id": context.request_id,
        "sequence": context.sequence_start,
        "type": "tool_result",
        "choice_index": reference.choice_index,
        "output_id": output_id,
        "tool_call_index": reference.tool_call_index,
        "call_id": call_id,
        "content": content,
        "is_error": False,
        **policy,
    }
    if reference.parallel_group_id is not None:
        event_data["parallel_group_id"] = reference.parallel_group_id
    _with_evidence(
        event_data,
        context=context,
        namespace="openai-chat.native_tool_result",
        evidence=evidence,
    )
    return ToolResultBlock.model_validate(block_data), ToolResultEvent.model_validate(event_data)


def _decode_messages(
    values: object,
    *,
    context: DecodeContext,
) -> tuple[
    list[Instruction],
    list[Message],
    tuple[ToolResultEvent, ...],
    tuple[CallReference, ...],
]:
    native_messages = _require_list(values, "/messages")
    if len(native_messages) > context.limits.max_blocks:
        raise _error("too_many_messages", "/messages")
    instructions: list[Instruction] = []
    messages: list[Message] = []
    tool_results: list[ToolResultEvent] = []
    result_call_ids: set[str] = set()
    assistant_calls: dict[str, ToolCallBlock] = {}
    call_references: list[CallReference] = []
    for index, item in enumerate(native_messages):
        field = f"/messages/{index}"
        native = _require_object(item, field)
        role = native.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise _error("unsupported_role", f"{field}/role")
        if role == "tool":
            result_block, result_event = _decode_tool_result(
                native,
                context=context,
                field=field,
                prior_calls=context.prior_calls,
            )
            result_call_id = str(result_event.call_id)
            if result_call_id in result_call_ids:
                raise _error("duplicate_tool_result", field)
            result_call_ids.add(result_call_id)
            message_data: dict[str, object] = {
                "role": "tool",
                "content": [result_block],
                **_derived_policy(context, Origin.TOOL),
            }
            if "name" in native:
                message_data["name"] = native["name"]
            messages.append(Message.model_validate(message_data))
            tool_results.append(
                result_event.model_copy(update={"sequence": context.sequence_start + len(tool_results)})
            )
            continue

        allowed = {"role", "content", "name"}
        if role == "assistant":
            allowed.add("tool_calls")
        unknown = set(native) - allowed
        if unknown:
            raise _error("unsupported_message", field)
        origin = Origin.PROVIDER if role == "assistant" else None
        content_value = native.get("content")
        content: list[object] = []
        if content_value is not None:
            content.extend(
                _decode_text_content(
                    content_value,
                    context=context,
                    origin=origin,
                    field=f"{field}/content",
                )
            )
        elif role != "assistant":
            raise _error("invalid_shape", f"{field}/content")
        for call_index, call_value in enumerate(native.get("tool_calls", [])):
            block = _decode_tool_call(
                call_value,
                context=context,
                field=f"{field}/tool_calls/{call_index}",
            )
            if block.call_id in assistant_calls:
                raise _error("duplicate_call_id", f"{field}/tool_calls/{call_index}/id")
            assistant_calls[str(block.call_id)] = block
            reference = context.prior_calls.get(str(block.call_id))
            if context.kind == "continuation" and (
                reference is None or reference.name != block.name
            ):
                raise _error("broken_call_link", f"{field}/tool_calls/{call_index}/id")
            if reference is not None:
                call_references.append(reference)
            content.append(block)
        message_policy = _derived_policy(context, Origin.PROVIDER) if role == "assistant" else _policy(context)
        message_data = {"role": role, "content": content, **message_policy}
        if "name" in native:
            message_data["name"] = native["name"]
        messages.append(Message.model_validate(message_data))
        if role in {"system", "developer"}:
            instructions.append(
                Instruction.model_validate(
                    {
                        "source_role": role,
                        "source_location": f"/messages/{index}/content",
                        "order": len(instructions),
                        "content": content,
                        **_policy(context),
                    }
                )
            )

    for result in tool_results:
        if str(result.call_id) not in assistant_calls:
            raise _error("broken_call_link", "/messages")
    if sum(len(message.content) for message in messages) > context.limits.max_blocks:
        raise _error("too_many_blocks", "/messages")
    return instructions, messages, tuple(tool_results), tuple(call_references)


def _decode_tools(value: object, *, context: DecodeContext) -> list[ToolDefinition]:
    native_tools = _require_list(value, "/tools")
    if len(native_tools) > context.limits.max_tools:
        raise _error("too_many_tools", "/tools")
    tools: list[ToolDefinition] = []
    names: set[str] = set()
    for index, item in enumerate(native_tools):
        field = f"/tools/{index}"
        native = _require_object(item, field)
        if set(native) - {"type", "function"} or native.get("type") != "function":
            raise _error("unsupported_tool", field)
        function = _require_object(native.get("function"), f"{field}/function")
        if set(function) - {"name", "description", "parameters", "strict"}:
            raise _error("unsupported_tool", f"{field}/function")
        name = _require_string(function.get("name"), f"{field}/function/name")
        if name in names:
            raise _error("duplicate_tool_name", f"{field}/function/name")
        names.add(name)
        description = _require_string(
            function.get("description", ""),
            f"{field}/function/description",
            allow_empty=True,
        )
        parameters = _require_object(function.get("parameters"), f"{field}/function/parameters")
        tool_data: dict[str, object] = {
            "name": name,
            "description": description,
            "input_schema": parameters,
            **_policy(context),
        }
        if "strict" in function:
            if type(function["strict"]) is not bool:
                raise _error("invalid_shape", f"{field}/function/strict")
            tool_data["strict"] = function["strict"]
        _with_evidence(
            tool_data,
            context=context,
            namespace="openai-chat.native_tool_fields",
            evidence={"type": "function"},
        )
        tools.append(ToolDefinition.model_validate(tool_data))
    return tools


def _decode_generation(native: dict[str, Any]) -> Generation | None:
    fields: dict[str, object] = {}
    max_fields = [key for key in ("max_completion_tokens", "max_tokens") if key in native]
    if len(max_fields) > 1:
        raise _error("conflicting_fields", "/max_tokens")
    if max_fields:
        fields["max_output_tokens"] = native[max_fields[0]]
    for key in ("temperature", "top_p", "stop", "seed", "reasoning_effort"):
        if key not in native:
            continue
        value = native[key]
        if key == "stop" and type(value) is str:
            value = [value]
        fields[key] = value
    if not fields:
        return None
    try:
        return Generation.model_validate(fields)
    except ValidationError:
        raise _error("invalid_generation") from None


def _decode_tool_choice(value: object) -> object:
    if type(value) is str and value in {"auto", "none", "required"}:
        return value
    native = _require_object(value, "/tool_choice")
    if set(native) != {"type", "function"} or native.get("type") != "function":
        raise _error("unsupported_tool_choice", "/tool_choice")
    function = _require_object(native["function"], "/tool_choice/function")
    if set(function) != {"name"}:
        raise _error("unsupported_tool_choice", "/tool_choice/function")
    return {"type": "tool", "name": _require_string(function["name"], "/tool_choice/function/name")}


class OpenAIChatAdapter:
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
        response_features=frozenset({"text", "tool_calls", "usage", "stream"}),
        verified_harnesses=("opencode-1.18.18", "pi-0.84.2"),
    )

    @safe_adapter_boundary
    def decode_request(
        self,
        value: str | bytes | bytearray | Mapping[str, object],
        *,
        context: DecodeContext,
    ) -> DecodedRequest:
        native = parse_json_object(value, protocol=self.protocol, limits=context.limits)
        ensure_safe_evidence(native, protocol=self.protocol)
        unknown = set(native) - _KNOWN_REQUEST_FIELDS
        if unknown:
            raise _error("unsupported_request_field", "/")
        model = _require_string(native.get("model"), "/model")
        if "messages" not in native:
            raise _error("missing_field", "/messages")
        try:
            instructions, messages, tool_results, calls = _decode_messages(
                native["messages"], context=context
            )
            tools = _decode_tools(native.get("tools", []), context=context)
            request_data: dict[str, object] = {
                "model": model,
                "instructions": instructions,
                "messages": messages,
                "tools": tools,
                **_policy(context),
            }
            if "tool_choice" in native:
                request_data["tool_choice"] = _decode_tool_choice(native["tool_choice"])
            generation = _decode_generation(native)
            if generation is not None:
                request_data["generation"] = generation
            if "stream" in native:
                if type(native["stream"]) is not bool:
                    raise _error("invalid_shape", "/stream")
                request_data["stream"] = native["stream"]
            if "metadata" in native:
                request_data["metadata"] = _require_object(native["metadata"], "/metadata")
            extensions: dict[str, object] = {}
            options = {key: native[key] for key in _REQUEST_OPTION_FIELDS if key in native}
            if options:
                extensions["openai-chat.request_options"] = options
            if context.policy.include_native_evidence:
                extensions["openai-chat.native_request"] = native
            if extensions:
                request_data["extensions"] = extensions
            request = Request.model_validate(request_data)
            envelope_data: dict[str, object] = {
                "request_id": context.request_id,
                "kind": context.kind,
                "request": request,
                **_policy(context),
            }
            if context.kind == "continuation":
                if context.parent_request_id is None:
                    raise _error("missing_parent_request_id")
                envelope_data["parent_request_id"] = context.parent_request_id
            elif context.kind != "initial" or context.parent_request_id is not None:
                raise _error("invalid_request_kind")
            envelope = RequestEnvelope.model_validate(envelope_data)
            return DecodedRequest(request=envelope, tool_results=tool_results, calls=calls)
        except ProtocolAdapterError:
            raise
        except (ValidationError, TypeError, ValueError):
            raise _error("invalid_shape") from None

    @safe_adapter_boundary
    def encode_stream(
        self,
        events: Sequence[AgentEvent],
        *,
        context: EncodeContext,
    ) -> str:
        return _encode_stream(events, context=context)


def _chunk(
    *,
    response_id: str,
    context: EncodeContext,
    choice_index: int,
    delta: dict[str, object],
    finish_reason: str | None = None,
    usage: Usage | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": context.created_at,
        "model": context.model,
        "choices": [
            {
                "index": choice_index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        usage_data = usage.model_dump(mode="json", exclude_unset=True)
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        payload["usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": usage_data.get("total_tokens", input_tokens + output_tokens),
        }
    return payload


def _encode_stream(events: Sequence[AgentEvent], *, context: EncodeContext) -> str:
    if isinstance(events, (str, bytes, bytearray)) or not isinstance(events, Sequence):
        raise _error("invalid_event_stream")
    if not events:
        raise _error("empty_event_stream")
    if len(events) > context.limits.max_events:
        raise _error("too_many_events")
    if type(context.model) is not str or not context.model:
        raise _error("invalid_encode_model")
    if len(context.model.encode("utf-8")) > context.limits.max_string_bytes:
        raise _error("encode_model_too_large")
    if len(context.model.encode("utf-8")) > context.limits.max_sse_stream_bytes:
        raise _error("sse_stream_too_large")
    if type(context.created_at) is not int or context.created_at < 0:
        raise _error("invalid_created_at")
    if context.response_id is not None and (
        type(context.response_id) is not str or not context.response_id
    ):
        raise _error("invalid_response_id")
    if (
        context.response_id is not None
        and len(context.response_id.encode("utf-8")) > context.limits.max_string_bytes
    ):
        raise _error("response_id_too_large")
    if type(context.parallel_tool_calls) is not bool:
        raise _error("invalid_parallel_tool_calls")
    if context.metadata:
        raise _error("unsupported_encode_metadata")
    if any(
        not isinstance(
            event,
            (
                ContentStartedEvent,
                ContentDeltaEvent,
                ContentCompletedEvent,
                ToolCallStartedEvent,
                ToolCallArgumentsDeltaEvent,
                ToolCallCompletedEvent,
                ResponseCompletedEvent,
            ),
        )
        for event in events
    ):
        raise _error("unsupported_event")
    first = events[0]
    request_id = str(first.request_id)
    output_id = str(first.output_id)
    choice_index = first.choice_index
    if any(
        str(event.request_id) != request_id
        or str(event.output_id) != output_id
        or event.choice_index != choice_index
        for event in events
    ):
        raise _error("mixed_output_stream")
    if any(events[index].sequence >= events[index + 1].sequence for index in range(len(events) - 1)):
        raise _error("invalid_sequence")
    if not isinstance(events[-1], ResponseCompletedEvent):
        raise _error("missing_terminal")
    if sum(isinstance(event, ResponseCompletedEvent) for event in events) != 1:
        raise _error("invalid_terminal")

    response_id = context.response_id or output_id
    frames: list[SSEFrame] = [
        SSEFrame(
            event=None,
            data=_chunk(
                response_id=response_id,
                context=context,
                choice_index=choice_index,
                delta={"role": "assistant", "content": None},
            ),
        )
    ]
    content_started = False
    content_completed = False
    content_fragments: list[str] = []
    content_bytes = 0
    calls: dict[int, dict[str, object]] = {}

    for event in events[:-1]:
        if isinstance(event, ContentStartedEvent):
            if event.content_type == "reasoning":
                raise _error("reasoning_leakage")
            if event.content_type != "text" or content_started or calls:
                raise _error("unsupported_content_event")
            content_started = True
        elif isinstance(event, ContentDeltaEvent):
            if not content_started or content_completed:
                raise _error("invalid_content_lifecycle")
            content_bytes += len(event.delta.encode("utf-8"))
            if content_bytes > context.limits.max_string_bytes:
                raise _error("content_too_large")
            content_fragments.append(event.delta)
            if event.delta:
                frames.append(
                    SSEFrame(
                        event=None,
                        data=_chunk(
                            response_id=response_id,
                            context=context,
                            choice_index=choice_index,
                            delta={"content": event.delta},
                        ),
                    )
                )
        elif isinstance(event, ContentCompletedEvent):
            if (
                not content_started
                or content_completed
                or event.content.type != "text"
                or "".join(content_fragments) != event.content.text
            ):
                raise _error("invalid_content_lifecycle")
            content_completed = True
        elif isinstance(event, ToolCallStartedEvent):
            if content_started or event.tool_call_index in calls:
                raise _error("invalid_tool_lifecycle")
            if calls and not context.parallel_tool_calls:
                raise _error("parallel_tool_calls_disabled")
            calls[event.tool_call_index] = {
                "call_id": str(event.call_id),
                "name": event.name,
                "parallel_group_id": event.parallel_group_id,
                "fragments": [],
                "argument_bytes": 0,
                "completed": False,
            }
            frames.append(
                SSEFrame(
                    event=None,
                    data=_chunk(
                        response_id=response_id,
                        context=context,
                        choice_index=choice_index,
                        delta={
                            "tool_calls": [
                                {
                                    "index": event.tool_call_index,
                                    "id": str(event.call_id),
                                    "type": "function",
                                    "function": {"name": event.name, "arguments": ""},
                                }
                            ]
                        },
                    ),
                )
            )
        elif isinstance(event, ToolCallArgumentsDeltaEvent):
            state = calls.get(event.tool_call_index)
            if (
                state is None
                or state["completed"]
                or str(event.call_id) != state["call_id"]
                or event.parallel_group_id != state["parallel_group_id"]
            ):
                raise _error("invalid_tool_lifecycle")
            fragment = event.delta
            argument_bytes = cast(int, state["argument_bytes"]) + len(fragment.encode("utf-8"))
            if argument_bytes > context.limits.max_arguments_bytes:
                raise _error("arguments_too_large")
            state["argument_bytes"] = argument_bytes
            cast(list[str], state["fragments"]).append(fragment)
            if fragment:
                frames.append(
                    SSEFrame(
                        event=None,
                        data=_chunk(
                            response_id=response_id,
                            context=context,
                            choice_index=choice_index,
                            delta={
                                "tool_calls": [
                                    {
                                        "index": event.tool_call_index,
                                        "function": {"arguments": fragment},
                                    }
                                ]
                            },
                        ),
                    )
                )
        elif isinstance(event, ToolCallCompletedEvent):
            state = calls.get(event.tool_call_index)
            if (
                state is None
                or state["completed"]
                or str(event.call_id) != state["call_id"]
                or event.name != state["name"]
                or event.parallel_group_id != state["parallel_group_id"]
                or "".join(cast(list[str], state["fragments"])) != event.arguments_raw
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
            state["completed"] = True

    terminal = cast(ResponseCompletedEvent, events[-1])
    if content_started:
        if not content_completed or calls or terminal.finish_reason == "tool_calls":
            raise _error("invalid_terminal")
        frames[0] = SSEFrame(
            event=None,
            data=_chunk(
                response_id=response_id,
                context=context,
                choice_index=choice_index,
                delta={"role": "assistant", "content": ""},
            ),
        )
    elif calls:
        if any(not state["completed"] for state in calls.values()) or terminal.finish_reason != "tool_calls":
            raise _error("invalid_terminal")
    else:
        raise _error("empty_output")

    frames.append(
        SSEFrame(
            event=None,
            data=_chunk(
                response_id=response_id,
                context=context,
                choice_index=choice_index,
                delta={},
                finish_reason=terminal.finish_reason,
                usage=terminal.usage,
            ),
        )
    )
    frames.append(SSEFrame(event=None, data="[DONE]"))
    return encode_sse(frames, protocol=_PROTOCOL, limits=context.limits)


openai_chat_adapter = OpenAIChatAdapter()


__all__ = ["OpenAIChatAdapter", "openai_chat_adapter"]
