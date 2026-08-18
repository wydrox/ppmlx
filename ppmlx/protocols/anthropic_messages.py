"""Pure Anthropic Messages protocol adapter."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import ValidationError

from ppmlx.agent_ir import (
    AgentEvent,
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    DocumentBlock,
    Generation,
    ImageBlock,
    Instruction,
    Message,
    NamedToolChoice,
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
    UsageSource,
)
from ppmlx.agent_ir.content import ContentBlock
from ppmlx.protocols.base import (
    CallReference,
    DecodeContext,
    DecodedRequest,
    EncodeContext,
    NormalizationPolicy,
    ProtocolAdapterError,
    ProtocolCapabilities,
    safe_adapter_boundary,
)
from ppmlx.protocols.json import ensure_safe_evidence, parse_json_object
from ppmlx.protocols.sse import SSEFrame, encode_sse


_PROTOCOL = "anthropic-messages"
_NATIVE_BLOCK = f"{_PROTOCOL}.native_block"
_NATIVE_REQUEST = f"{_PROTOCOL}.native_request"
_NATIVE_TOOL_RESULT = f"{_PROTOCOL}.native_tool_result"
_CACHE_CONTROL = f"{_PROTOCOL}.cache_control"
_GENERATION = f"{_PROTOCOL}.generation"
_REQUEST_OPTIONS = f"{_PROTOCOL}.request_options"

_REQUEST_FIELDS = {
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "stream",
    "metadata",
    "thinking",
    "context_management",
    "output_config",
    "service_tier",
}
_FINISH_REASONS = {
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
}
_SUPPORTED_EVENTS = (
    ContentStartedEvent,
    ContentDeltaEvent,
    ContentCompletedEvent,
    ToolCallStartedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ResponseCompletedEvent,
)


def _error(code: str, field: str | None = None) -> ProtocolAdapterError:
    return ProtocolAdapterError(protocol=_PROTOCOL, code=code, field=field)


def _require_dict(value: object, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _error("invalid_type", field)
    return cast(dict[str, Any], value)


def _require_list(value: object, field: str) -> list[Any]:
    if type(value) is not list:
        raise _error("invalid_type", field)
    return cast(list[Any], value)


def _require_string(value: object, field: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise _error("invalid_type", field)
    return value


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise _error("invalid_type", field)
    return value


def _require_keys(
    value: Mapping[str, object],
    *,
    allowed: set[str],
    required: set[str],
    field: str,
) -> None:
    if not required <= value.keys():
        raise _error("missing_field", field)
    unexpected = set(value) - allowed
    if unexpected:
        raise _error("unsupported_field", field)


def _provenance(policy: NormalizationPolicy, origin: Origin | None = None) -> Provenance:
    if origin is None:
        return policy.provenance
    values: dict[str, object] = {"origin": origin, "trust": policy.provenance.trust}
    if policy.provenance.origin_id is not None:
        values["origin_id"] = policy.provenance.origin_id
    return Provenance.model_validate(values)


def _policy_values(
    policy: NormalizationPolicy,
    *,
    origin: Origin | None = None,
    extensions: Mapping[str, object] | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "sensitivity": policy.sensitivity,
        "provenance": _provenance(policy, origin),
    }
    if extensions:
        values["extensions"] = dict(extensions)
    return values


def _native_block_extension(
    block: Mapping[str, object],
    *,
    policy: NormalizationPolicy,
) -> dict[str, object]:
    if "cache_control" not in block:
        return {}
    cache_control = _require_dict(block["cache_control"], "/cache_control")
    _require_keys(
        cache_control,
        allowed={"type", "ttl"},
        required={"type"},
        field="/cache_control",
    )
    if cache_control["type"] != "ephemeral":
        raise _error("unsupported_cache_control", "/cache_control/type")
    if "ttl" in cache_control:
        _require_string(cache_control["ttl"], "/cache_control/ttl")
    evidence = {"cache_control": cache_control}
    ensure_safe_evidence(evidence, protocol=_PROTOCOL)
    if policy.include_native_evidence:
        return {_NATIVE_BLOCK: evidence}
    return {_CACHE_CONTROL: evidence}


def _compact_json(value: object, field: str, max_bytes: int) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise _error("invalid_json", field) from None
    if len(raw.encode("utf-8")) > max_bytes:
        raise _error("arguments_too_large", field)
    return raw


class AnthropicMessagesAdapter:
    """Convert Anthropic Messages requests and Agent IR response events."""

    protocol = _PROTOCOL
    capabilities = ProtocolCapabilities(
        request_features=frozenset(
            {
                "instructions",
                "text",
                "image",
                "document",
                "stream",
                "tool_calls",
                "tool_results",
                "tools",
                "tool_choice",
                "generation",
                "metadata",
            }
        ),
        response_features=frozenset({"stream", "text", "tool_calls", "usage"}),
        verified_harnesses=("claude-code-2.1.231",),
    )

    @safe_adapter_boundary
    def decode_request(
        self,
        value: str | bytes | bytearray | Mapping[str, object],
        *,
        context: DecodeContext,
    ) -> DecodedRequest:
        native = parse_json_object(value, protocol=self.protocol, limits=context.limits)
        try:
            return self._decode_native(native, context=context)
        except ProtocolAdapterError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError):
            raise _error("invalid_request") from None

    def _decode_native(self, native: dict[str, Any], *, context: DecodeContext) -> DecodedRequest:
        _require_keys(native, allowed=_REQUEST_FIELDS, required={"model", "messages"}, field="/")
        if context.kind not in {"initial", "continuation"}:
            raise _error("invalid_context", "/kind")
        if context.kind == "initial" and context.parent_request_id is not None:
            raise _error("invalid_context", "/parent_request_id")
        if context.kind == "continuation" and context.parent_request_id is None:
            raise _error("invalid_context", "/parent_request_id")
        ensure_safe_evidence(native, protocol=self.protocol)

        block_count = [0]
        instructions = self._decode_system(native.get("system", []), context=context, count=block_count)
        messages, tool_results, calls = self._decode_messages(
            native["messages"], context=context, count=block_count
        )
        tools = self._decode_tools(native.get("tools", []), context=context)
        tool_choice, choice_options = self._decode_tool_choice(native.get("tool_choice"))
        generation = self._decode_generation(native)
        metadata = native.get("metadata")
        if metadata is not None:
            _require_dict(metadata, "/metadata")

        request_extensions: dict[str, object] = {}
        if context.policy.include_native_evidence:
            request_extensions[_NATIVE_REQUEST] = native
        options: dict[str, object] = {}
        options.update(choice_options)
        if "service_tier" in native:
            options["service_tier"] = native["service_tier"]
        if options:
            request_extensions[_REQUEST_OPTIONS] = options

        request_values: dict[str, object] = {
            "model": _require_string(native["model"], "/model"),
            "instructions": instructions,
            "messages": messages,
            "tools": tools,
            **_policy_values(context.policy, extensions=request_extensions),
        }
        if tool_choice is not None:
            request_values["tool_choice"] = tool_choice
        if generation is not None:
            request_values["generation"] = generation
        if "stream" in native:
            request_values["stream"] = _require_bool(native["stream"], "/stream")
        if metadata is not None:
            request_values["metadata"] = metadata

        envelope_values: dict[str, object] = {
            "request_id": context.request_id,
            "kind": context.kind,
            "request": Request.model_validate(request_values),
            **_policy_values(context.policy),
        }
        if context.parent_request_id is not None:
            envelope_values["parent_request_id"] = context.parent_request_id
        return DecodedRequest(
            request=RequestEnvelope.model_validate(envelope_values),
            tool_results=tuple(tool_results),
            calls=tuple(calls),
        )

    def _decode_system(
        self,
        value: object,
        *,
        context: DecodeContext,
        count: list[int],
    ) -> list[Instruction]:
        if type(value) is str:
            blocks: list[Any] = [{"type": "text", "text": value}]
        else:
            blocks = _require_list(value, "/system")
        instructions: list[Instruction] = []
        for index, item in enumerate(blocks):
            block = _require_dict(item, f"/system/{index}")
            _require_keys(
                block,
                allowed={"type", "text", "cache_control"},
                required={"type", "text"},
                field=f"/system/{index}",
            )
            if block["type"] != "text":
                raise _error("unsupported_content", f"/system/{index}/type")
            self._add_block(count, context, f"/system/{index}")
            content = self._text_block(
                block,
                context=context,
                origin=Origin.HARNESS,
                field=f"/system/{index}",
            )
            instructions.append(
                Instruction.model_validate(
                    {
                        "source_role": "system",
                        "source_location": f"/system/{index}",
                        "order": index,
                        "content": [content],
                        **_policy_values(context.policy, origin=Origin.HARNESS),
                    }
                )
            )
        return instructions

    def _decode_messages(
        self,
        value: object,
        *,
        context: DecodeContext,
        count: list[int],
    ) -> tuple[list[Message], list[ToolResultEvent], list[CallReference]]:
        native_messages = _require_list(value, "/messages")
        messages: list[Message] = []
        results: list[ToolResultEvent] = []
        result_call_ids: set[str] = set()
        calls: list[CallReference] = []
        call_ids: set[str] = set()
        sequence = context.sequence_start
        for message_index, item in enumerate(native_messages):
            field = f"/messages/{message_index}"
            message = _require_dict(item, field)
            _require_keys(message, allowed={"role", "content"}, required={"role", "content"}, field=field)
            role = _require_string(message["role"], f"{field}/role")
            if role not in {"user", "assistant"}:
                raise _error("unsupported_role", f"{field}/role")
            native_content = message["content"]
            if type(native_content) is str:
                native_blocks: list[Any] = [{"type": "text", "text": native_content}]
            else:
                native_blocks = _require_list(native_content, f"{field}/content")
            content: list[ContentBlock] = []
            for block_index, native_item in enumerate(native_blocks):
                block_field = f"{field}/content/{block_index}"
                block = _require_dict(native_item, block_field)
                block_type = block.get("type")
                if block_type == "text":
                    content.append(
                        self._text_block(block, context=context, origin=Origin.HARNESS, field=block_field)
                    )
                elif block_type == "image" and role == "user":
                    content.append(self._image_block(block, context=context, field=block_field))
                elif block_type == "document" and role == "user":
                    content.append(self._document_block(block, context=context, field=block_field))
                elif block_type == "tool_use" and role == "assistant":
                    tool_call, reference = self._tool_call_block(
                        block, context=context, field=block_field, fallback_index=len(calls)
                    )
                    if reference.call_id in call_ids:
                        raise _error("duplicate_call_id", block_field)
                    call_ids.add(reference.call_id)
                    content.append(tool_call)
                    if reference not in calls:
                        calls.append(reference)
                elif block_type == "tool_result" and role == "user":
                    tool_result, event, reference = self._tool_result_block(
                        block,
                        context=context,
                        field=block_field,
                        sequence=sequence,
                        count=count,
                    )
                    result_call_id = str(event.call_id)
                    if result_call_id in result_call_ids:
                        raise _error("duplicate_tool_result", block_field)
                    result_call_ids.add(result_call_id)
                    sequence += 1
                    content.append(tool_result)
                    results.append(event)
                    if reference not in calls:
                        calls.append(reference)
                else:
                    raise _error("unsupported_content", f"{block_field}/type")
                self._add_block(count, context, block_field)
            origin = Origin.PROVIDER if role == "assistant" else Origin.HARNESS
            messages.append(
                Message.model_validate(
                    {
                        "role": cast(Literal["user", "assistant"], role),
                        "content": content,
                        **_policy_values(context.policy, origin=origin),
                    }
                )
            )
        return messages, results, calls

    def _text_block(
        self,
        block: dict[str, Any],
        *,
        context: DecodeContext,
        origin: Origin,
        field: str,
    ) -> TextBlock:
        _require_keys(
            block,
            allowed={"type", "text", "cache_control"},
            required={"type", "text"},
            field=field,
        )
        return TextBlock.model_validate(
            {
                "type": "text",
                "text": _require_string(block["text"], f"{field}/text", empty=True),
                **_policy_values(
                    context.policy,
                    origin=origin,
                    extensions=_native_block_extension(block, policy=context.policy),
                ),
            }
        )

    def _image_block(
        self, block: dict[str, Any], *, context: DecodeContext, field: str
    ) -> ImageBlock:
        _require_keys(
            block,
            allowed={"type", "source", "cache_control"},
            required={"type", "source"},
            field=field,
        )
        source = _require_dict(block["source"], f"{field}/source")
        source_type = source.get("type")
        if source_type == "base64":
            _require_keys(
                source,
                allowed={"type", "media_type", "data"},
                required={"type", "media_type", "data"},
                field=f"{field}/source",
            )
            source_value = {"data": _require_string(source["data"], f"{field}/source/data")}
        elif source_type == "url":
            _require_keys(
                source,
                allowed={"type", "media_type", "url"},
                required={"type", "media_type", "url"},
                field=f"{field}/source",
            )
            source_value = {"url": _require_string(source["url"], f"{field}/source/url")}
        else:
            raise _error("unsupported_content", f"{field}/source/type")
        return ImageBlock.model_validate(
            {
                "type": "image",
                "media_type": _require_string(source["media_type"], f"{field}/source/media_type"),
                **source_value,
                **_policy_values(
                    context.policy,
                    origin=Origin.HARNESS,
                    extensions=_native_block_extension(block, policy=context.policy),
                ),
            }
        )

    def _document_block(
        self, block: dict[str, Any], *, context: DecodeContext, field: str
    ) -> DocumentBlock:
        _require_keys(
            block,
            allowed={"type", "source", "cache_control"},
            required={"type", "source"},
            field=field,
        )
        source = _require_dict(block["source"], f"{field}/source")
        source_type = source.get("type")
        if source_type == "base64":
            _require_keys(
                source,
                allowed={"type", "media_type", "data"},
                required={"type", "media_type", "data"},
                field=f"{field}/source",
            )
            source_value = {"data": _require_string(source["data"], f"{field}/source/data")}
        elif source_type == "text":
            _require_keys(
                source,
                allowed={"type", "media_type", "data"},
                required={"type", "media_type", "data"},
                field=f"{field}/source",
            )
            source_value = {"text": _require_string(source["data"], f"{field}/source/data", empty=True)}
        elif source_type == "url":
            _require_keys(
                source,
                allowed={"type", "media_type", "url"},
                required={"type", "media_type", "url"},
                field=f"{field}/source",
            )
            source_value = {"url": _require_string(source["url"], f"{field}/source/url")}
        else:
            raise _error("unsupported_content", f"{field}/source/type")
        return DocumentBlock.model_validate(
            {
                "type": "document",
                "media_type": _require_string(source["media_type"], f"{field}/source/media_type"),
                **source_value,
                **_policy_values(
                    context.policy,
                    origin=Origin.HARNESS,
                    extensions=_native_block_extension(block, policy=context.policy),
                ),
            }
        )

    def _tool_call_block(
        self,
        block: dict[str, Any],
        *,
        context: DecodeContext,
        field: str,
        fallback_index: int,
    ) -> tuple[ToolCallBlock, CallReference]:
        _require_keys(
            block,
            allowed={"type", "id", "name", "input", "cache_control"},
            required={"type", "id", "name", "input"},
            field=field,
        )
        call_id = _require_string(block["id"], f"{field}/id")
        name = _require_string(block["name"], f"{field}/name")
        arguments = _require_dict(block["input"], f"{field}/input")
        raw = _compact_json(arguments, f"{field}/input", context.limits.max_arguments_bytes)
        reference = context.prior_calls.get(call_id)
        if reference is None:
            reference = CallReference(
                call_id=call_id,
                name=name,
                choice_index=0,
                output_id=f"native_request_{context.request_id}",
                tool_call_index=fallback_index,
            )
        elif reference.name != name:
            raise _error("broken_tool_link", f"{field}/name")
        return (
            ToolCallBlock.model_validate(
                {
                    "type": "tool_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments_raw": raw,
                    "arguments_json": arguments,
                    **_policy_values(
                        context.policy,
                        origin=Origin.PROVIDER,
                        extensions=_native_block_extension(block, policy=context.policy),
                    ),
                }
            ),
            reference,
        )

    def _tool_result_block(
        self,
        block: dict[str, Any],
        *,
        context: DecodeContext,
        field: str,
        sequence: int,
        count: list[int],
    ) -> tuple[ToolResultBlock, ToolResultEvent, CallReference]:
        _require_keys(
            block,
            allowed={"type", "tool_use_id", "content", "is_error", "cache_control"},
            required={"type", "tool_use_id", "content"},
            field=field,
        )
        call_id = _require_string(block["tool_use_id"], f"{field}/tool_use_id")
        reference = context.prior_calls.get(call_id)
        if reference is None:
            raise _error("broken_tool_link", f"{field}/tool_use_id")
        result_output_id = context.result_output_ids.get(call_id)
        if result_output_id is None:
            raise _error("missing_result_output_id", f"{field}/tool_use_id")
        is_error = False
        if "is_error" in block:
            is_error = _require_bool(block["is_error"], f"{field}/is_error")
        content = self._decode_tool_result_content(
            block["content"], context=context, field=f"{field}/content", count=count
        )
        block_extensions = _native_block_extension(block, policy=context.policy)
        result_block = ToolResultBlock.model_validate(
            {
                "type": "tool_result",
                "call_id": call_id,
                "content": content,
                "is_error": is_error,
                **_policy_values(context.policy, origin=Origin.TOOL, extensions=block_extensions),
            }
        )
        event_extensions: dict[str, object] = {}
        if context.policy.include_native_evidence:
            event_extensions[_NATIVE_TOOL_RESULT] = block
        event = ToolResultEvent.model_validate(
            {
                "request_id": context.request_id,
                "sequence": sequence,
                "type": "tool_result",
                "choice_index": reference.choice_index,
                "output_id": result_output_id,
                "tool_call_index": reference.tool_call_index,
                "call_id": call_id,
                "content": content,
                "is_error": is_error,
                **(
                    {"parallel_group_id": reference.parallel_group_id}
                    if reference.parallel_group_id is not None
                    else {}
                ),
                **_policy_values(context.policy, origin=Origin.TOOL, extensions=event_extensions),
            }
        )
        return result_block, event, reference

    def _decode_tool_result_content(
        self,
        value: object,
        *,
        context: DecodeContext,
        field: str,
        count: list[int],
    ) -> list[ContentBlock]:
        if type(value) is str:
            return [
                TextBlock.model_validate(
                    {
                        "type": "text",
                        "text": value,
                        **_policy_values(context.policy, origin=Origin.TOOL),
                    }
                )
            ]
        native_blocks = _require_list(value, field)
        content: list[ContentBlock] = []
        for index, item in enumerate(native_blocks):
            block_field = f"{field}/{index}"
            block = _require_dict(item, block_field)
            if block.get("type") != "text":
                raise _error("unsupported_content", f"{block_field}/type")
            self._add_block(count, context, block_field)
            content.append(
                self._text_block(block, context=context, origin=Origin.TOOL, field=block_field)
            )
        return content

    def _decode_tools(self, value: object, *, context: DecodeContext) -> list[ToolDefinition]:
        tools = _require_list(value, "/tools")
        if len(tools) > context.limits.max_tools:
            raise _error("too_many_tools", "/tools")
        result: list[ToolDefinition] = []
        names: set[str] = set()
        for index, item in enumerate(tools):
            field = f"/tools/{index}"
            tool = _require_dict(item, field)
            _require_keys(
                tool,
                allowed={"name", "description", "input_schema", "cache_control"},
                required={"name", "description", "input_schema"},
                field=field,
            )
            extensions = _native_block_extension(tool, policy=context.policy)
            name = _require_string(tool["name"], f"{field}/name")
            if name in names:
                raise _error("duplicate_tool_name", f"{field}/name")
            names.add(name)
            result.append(
                ToolDefinition.model_validate(
                    {
                        "name": name,
                        "description": _require_string(
                            tool["description"], f"{field}/description", empty=True
                        ),
                        "input_schema": _require_dict(
                            tool["input_schema"], f"{field}/input_schema"
                        ),
                        **_policy_values(
                            context.policy, origin=Origin.HARNESS, extensions=extensions
                        ),
                    }
                )
            )
        return result

    def _decode_tool_choice(
        self, value: object
    ) -> tuple[str | NamedToolChoice | None, dict[str, object]]:
        if value is None:
            return None, {}
        choice = _require_dict(value, "/tool_choice")
        choice_type = choice.get("type")
        allowed = {"type", "disable_parallel_tool_use"}
        if choice_type == "tool":
            allowed.add("name")
        _require_keys(choice, allowed=allowed, required={"type"}, field="/tool_choice")
        if choice_type == "auto":
            normalized: str | NamedToolChoice = "auto"
        elif choice_type == "any":
            normalized = "required"
        elif choice_type == "tool":
            normalized = NamedToolChoice(
                type="tool", name=_require_string(choice.get("name"), "/tool_choice/name")
            )
        elif choice_type == "none":
            normalized = "none"
        else:
            raise _error("unsupported_tool_choice", "/tool_choice/type")
        options: dict[str, object] = {}
        if "disable_parallel_tool_use" in choice:
            options["disable_parallel_tool_use"] = _require_bool(
                choice["disable_parallel_tool_use"], "/tool_choice/disable_parallel_tool_use"
            )
        return normalized, options

    def _decode_generation(self, native: Mapping[str, object]) -> Generation | None:
        values: dict[str, object] = {}
        if "max_tokens" in native:
            value = native["max_tokens"]
            if type(value) is not int or value < 1:
                raise _error("invalid_type", "/max_tokens")
            values["max_output_tokens"] = value
        for source, target in (("temperature", "temperature"), ("top_p", "top_p")):
            if source in native:
                value = native[source]
                if type(value) not in {int, float}:
                    raise _error("invalid_type", f"/{source}")
                values[target] = value
        if "stop_sequences" in native:
            stop = _require_list(native["stop_sequences"], "/stop_sequences")
            values["stop"] = [
                _require_string(item, f"/stop_sequences/{index}", empty=True)
                for index, item in enumerate(stop)
            ]
        extension = {
            key: native[key]
            for key in ("thinking", "context_management", "output_config")
            if key in native
        }
        if extension:
            values["extensions"] = {_GENERATION: extension}
        return Generation.model_validate(values) if values else None

    @staticmethod
    def _add_block(count: list[int], context: DecodeContext, field: str) -> None:
        count[0] += 1
        if count[0] > context.limits.max_blocks:
            raise _error("too_many_blocks", field)

    @safe_adapter_boundary
    def encode_stream(self, events: Sequence[AgentEvent], *, context: EncodeContext) -> str:
        try:
            return self._encode_events(events, context=context)
        except ProtocolAdapterError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError):
            raise _error("invalid_events") from None

    def _encode_events(self, events: Sequence[AgentEvent], *, context: EncodeContext) -> str:
        if isinstance(events, (str, bytes, bytearray)) or not isinstance(events, Sequence):
            raise _error("invalid_events")
        if not events:
            raise _error("empty_events")
        if len(events) > context.limits.max_events:
            raise _error("too_many_events")
        if any(not isinstance(event, _SUPPORTED_EVENTS) for event in events):
            raise _error("unsupported_event")
        if context.metadata:
            raise _error("unsupported_encode_metadata")
        if type(context.model) is not str or not context.model:
            raise _error("invalid_context", "/model")
        if len(context.model.encode("utf-8")) > context.limits.max_string_bytes:
            raise _error("encode_model_too_large")
        if len(context.model.encode("utf-8")) > context.limits.max_sse_stream_bytes:
            raise _error("sse_stream_too_large")
        if context.response_id is not None and (
            type(context.response_id) is not str or not context.response_id
        ):
            raise _error("invalid_response_id")
        if (
            context.response_id is not None
            and len(context.response_id.encode("utf-8")) > context.limits.max_string_bytes
        ):
            raise _error("response_id_too_large")

        first = events[0]
        request_id = first.request_id
        output_id = first.output_id
        previous_sequence = -1
        terminal: ResponseCompletedEvent | None = None
        frames: list[SSEFrame] = []
        block_indices: dict[tuple[str, int], int] = {}
        text_fragments: dict[tuple[str, int], list[str]] = {}
        text_bytes: dict[tuple[str, int], int] = {}
        argument_fragments: dict[str, list[str]] = {}
        argument_bytes: dict[str, int] = {}
        tool_state: dict[tuple[str, int], tuple[str, str, str | None]] = {}
        open_blocks: set[tuple[str, int]] = set()

        for event in events:
            if event.request_id != request_id or event.output_id != output_id:
                raise _error("mixed_output")
            if event.choice_index != 0:
                raise _error("unsupported_choice")
            if event.sequence <= previous_sequence:
                raise _error("invalid_sequence")
            previous_sequence = event.sequence
            if terminal is not None:
                raise _error("event_after_terminal")

            if isinstance(event, ContentStartedEvent):
                if event.content_type != "text":
                    raise _error("unsupported_content")
                key = ("content", event.content_index)
                if key in block_indices:
                    raise _error("invalid_lifecycle")
                native_index = len(block_indices)
                block_indices[key] = native_index
                text_fragments[key] = []
                text_bytes[key] = 0
                open_blocks.add(key)
                frames.append(
                    SSEFrame(
                        event="content_block_start",
                        data={
                            "type": "content_block_start",
                            "index": native_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
            elif isinstance(event, ContentDeltaEvent):
                key = ("content", event.content_index)
                if key not in open_blocks:
                    raise _error("invalid_lifecycle")
                text_bytes[key] += len(event.delta.encode("utf-8"))
                if text_bytes[key] > context.limits.max_string_bytes:
                    raise _error("content_too_large")
                text_fragments[key].append(event.delta)
                frames.append(
                    SSEFrame(
                        event="content_block_delta",
                        data={
                            "type": "content_block_delta",
                            "index": block_indices[key],
                            "delta": {"type": "text_delta", "text": event.delta},
                        },
                    )
                )
            elif isinstance(event, ContentCompletedEvent):
                key = ("content", event.content_index)
                if key not in open_blocks or not isinstance(event.content, TextBlock):
                    raise _error("invalid_lifecycle")
                if "".join(text_fragments[key]) != event.content.text:
                    raise _error("content_mismatch")
                open_blocks.remove(key)
                frames.append(
                    SSEFrame(
                        event="content_block_stop",
                        data={"type": "content_block_stop", "index": block_indices[key]},
                    )
                )
            elif isinstance(event, ToolCallStartedEvent):
                key = ("tool", event.tool_call_index)
                if key in block_indices or event.call_id in argument_fragments:
                    raise _error("invalid_lifecycle")
                native_index = len(block_indices)
                block_indices[key] = native_index
                argument_fragments[event.call_id] = []
                argument_bytes[event.call_id] = 0
                tool_state[key] = (event.call_id, event.name, event.parallel_group_id)
                open_blocks.add(key)
                frames.append(
                    SSEFrame(
                        event="content_block_start",
                        data={
                            "type": "content_block_start",
                            "index": native_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": event.call_id,
                                "name": event.name,
                                "input": {},
                            },
                        },
                    )
                )
            elif isinstance(event, ToolCallArgumentsDeltaEvent):
                key = ("tool", event.tool_call_index)
                state = tool_state.get(key)
                if (
                    key not in open_blocks
                    or event.call_id not in argument_fragments
                    or state is None
                    or state[0] != event.call_id
                    or state[2] != event.parallel_group_id
                ):
                    raise _error("invalid_lifecycle")
                argument_bytes[event.call_id] += len(event.delta.encode("utf-8"))
                if argument_bytes[event.call_id] > context.limits.max_arguments_bytes:
                    raise _error("arguments_too_large")
                argument_fragments[event.call_id].append(event.delta)
                frames.append(
                    SSEFrame(
                        event="content_block_delta",
                        data={
                            "type": "content_block_delta",
                            "index": block_indices[key],
                            "delta": {"type": "input_json_delta", "partial_json": event.delta},
                        },
                    )
                )
            elif isinstance(event, ToolCallCompletedEvent):
                key = ("tool", event.tool_call_index)
                if (
                    key not in open_blocks
                    or event.call_id not in argument_fragments
                    or tool_state.get(key) != (event.call_id, event.name, event.parallel_group_id)
                ):
                    raise _error("invalid_lifecycle")
                if "".join(argument_fragments[event.call_id]) != event.arguments_raw:
                    raise _error("arguments_mismatch")
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
                open_blocks.remove(key)
                frames.append(
                    SSEFrame(
                        event="content_block_stop",
                        data={"type": "content_block_stop", "index": block_indices[key]},
                    )
                )
            elif isinstance(event, ResponseCompletedEvent):
                if event.finish_reason not in _FINISH_REASONS:
                    raise _error("unsupported_finish_reason")
                if open_blocks:
                    raise _error("incomplete_content")
                terminal = event
            else:
                raise _error("unsupported_event")

        if terminal is None:
            raise _error("missing_terminal")
        if context.response_id is not None and context.response_id != output_id:
            raise _error("response_id_mismatch")
        has_tool_calls = any(key[0] == "tool" for key in block_indices)
        if (terminal.finish_reason == "tool_use") != has_tool_calls:
            raise _error("invalid_terminal")
        usage = terminal.usage
        input_tokens = usage.input_tokens if usage is not None and usage.input_tokens is not None else 0
        output_tokens = usage.output_tokens if usage is not None and usage.output_tokens is not None else 0
        if usage is not None and usage.source not in {UsageSource.PROVIDER, UsageSource.PPMLX_ESTIMATE}:
            raise _error("unsupported_usage")
        if (
            usage is not None
            and usage.input_tokens is not None
            and usage.output_tokens is not None
            and usage.total_tokens is not None
            and usage.total_tokens != usage.input_tokens + usage.output_tokens
        ):
            raise _error("invalid_usage")

        start = SSEFrame(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": output_id,
                    "type": "message",
                    "role": "assistant",
                    "model": context.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )
        finish = [
            SSEFrame(
                event="message_delta",
                data={
                    "type": "message_delta",
                    "delta": {"stop_reason": terminal.finish_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            ),
            SSEFrame(event="message_stop", data={"type": "message_stop"}),
        ]
        return encode_sse(
            [start, *frames, *finish],
            protocol=_PROTOCOL,
            limits=context.limits,
        )


anthropic_messages_adapter = AnthropicMessagesAdapter()


__all__ = ["AnthropicMessagesAdapter", "anthropic_messages_adapter"]
