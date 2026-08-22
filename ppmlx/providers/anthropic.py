"""Remote Anthropic Messages implementation of the provider interface."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import httpx

from ppmlx.agent_ir import (
    AgentEvent,
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    NamedToolChoice,
    ResponseCompletedEvent,
    TextBlock,
    ToolCallArgumentsDeltaEvent,
    ToolCallBlock,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolDefinition,
    ToolResultBlock,
    Usage,
    UsageSource,
    new_call_id,
    new_output_id,
    new_parallel_group_id,
)

from .base import (
    ProviderCallReference,
    ProviderCapabilities,
    ProviderCancellationHandle,
    ProviderCancelledError,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealth,
    ProviderHealthStatus,
    ProviderInvocation,
    ProviderModel,
    ProviderResult,
    ProviderStreamingMode,
    ProviderToolSupportStatus,
)

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ENV_KEY = "ANTHROPIC_API_KEY"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL_CATALOG = ("claude-sonnet-4-5", "claude-opus-4-1")
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024

_IdFactory = Callable[[], str]

# Typed error codes for HTTP failures. Response bodies are never embedded in
# errors because they may echo credentials or request content.
_STATUS_ERROR_CODES: Mapping[int, str] = {
    401: "auth_failed",
    403: "auth_failed",
    429: "rate_limited",
}

# Anthropic stop_reason values this adapter understands; unknown reasons fall
# back to a derived finish reason instead of leaking through.
_KNOWN_STOP_REASONS = frozenset(
    {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"}
)


def _text_of(blocks: Sequence[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "".join(parts)


def _safe_source_call_id(value: Any) -> bool:
    return (
        type(value) is str
        and bool(value)
        and not any(character.isspace() for character in value)
    )


def _arguments_input(arguments_raw: str) -> dict[str, Any]:
    """Decode staged argument JSON back into the Messages ``input`` object."""
    if not isinstance(arguments_raw, str):
        raise ProviderError(provider_id="anthropic", code="unsupported_content")
    if not arguments_raw.strip():
        return {}
    try:
        decoded = json.loads(arguments_raw)
    except json.JSONDecodeError:
        raise ProviderError(provider_id="anthropic", code="unsupported_content") from None
    if not isinstance(decoded, Mapping):
        raise ProviderError(provider_id="anthropic", code="unsupported_content")
    return dict(decoded)


class AnthropicProvider:
    """Adapter for the remote Anthropic Messages API.

    Data path is REMOTE and credentials are ``api_key`` per ADR 0004. The API
    key is resolved from the environment at request time and sent via the
    ``x-api-key`` header with the ``anthropic-version`` header pinned to a
    known snapshot. It is never logged, embedded in exceptions, or exposed
    through reprs.

    Streaming iterates the SSE response live so cancellation is observed at
    chunk boundaries as a typed :class:`ProviderCancelledError` after already
    yielded events — never silent truncation. Tool-call lifecycle events are
    staged until the stream ends because a parallel-group identifier is only
    known once every tool call has been seen, and the Agent IR requires one
    group identity per tool-call lifecycle.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        env_key: str = DEFAULT_ENV_KEY,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        model_catalog: Sequence[str] = DEFAULT_MODEL_CATALOG,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.BaseTransport | None = None,
        call_id_factory: _IdFactory = new_call_id,
        output_id_factory: _IdFactory = new_output_id,
        parallel_group_factory: _IdFactory = new_parallel_group_id,
    ) -> None:
        if type(base_url) is not str or not base_url.startswith(("http://", "https://")):
            raise ValueError("Provider base URL is invalid")
        if type(env_key) is not str or not env_key or any(
            character.isspace() for character in env_key
        ):
            raise ValueError("Provider env key name is invalid")
        if type(anthropic_version) is not str or not anthropic_version or any(
            character.isspace() for character in anthropic_version
        ):
            raise ValueError("Provider anthropic-version header is invalid")
        if not isinstance(model_catalog, Sequence) or isinstance(
            model_catalog, (str, bytes, bytearray)
        ):
            raise ValueError("Provider model catalog is invalid")
        catalog = tuple(model_catalog)
        if any(type(item) is not str or not item for item in catalog):
            raise ValueError("Provider model catalog is invalid")
        if len(set(catalog)) != len(catalog):
            raise ValueError("Provider model catalog contains duplicates")
        if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("Provider timeout is invalid")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("Provider response size limit is invalid")
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            raise ValueError("Provider transport is invalid")
        for factory in (call_id_factory, output_id_factory, parallel_group_factory):
            if not callable(factory):
                raise ValueError("Provider identifier factory is invalid")
        self._base_url = base_url.rstrip("/")
        self._env_key = env_key
        self._anthropic_version = anthropic_version
        self._model_catalog = catalog
        self._timeout_seconds = float(timeout_seconds)
        self._max_response_bytes = max_response_bytes
        self._transport = transport
        self._call_id_factory = call_id_factory
        self._output_id_factory = output_id_factory
        self._parallel_group_factory = parallel_group_factory

    @property
    def provider_id(self) -> str:
        return "anthropic"

    def _resolve_api_key(self) -> str:
        key = os.environ.get(self._env_key)
        if type(key) is not str or not key:
            raise ProviderError(provider_id=self.provider_id, code="credential_missing")
        return key

    def _headers(self) -> dict[str, str]:
        # The resolved key lives only in this short-lived header mapping. It
        # is never logged, embedded in exceptions, or exposed through reprs.
        return {
            "x-api-key": self._resolve_api_key(),
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }

    def capabilities(self, model_id: str) -> ProviderCapabilities:
        if type(model_id) is not str or not model_id or any(
            character.isspace() for character in model_id
        ):
            raise ProviderError(provider_id=self.provider_id, code="invalid_model_id")
        return ProviderCapabilities(
            text=True,
            images=False,
            tools=True,
            parallel_tool_calls=True,
            reasoning=False,
            streaming=ProviderStreamingMode.NATIVE,
            context_window=None,
            data_path=ProviderDataPath.REMOTE,
            credential_types=(ProviderCredentialType.API_KEY,),
            tool_support_status=ProviderToolSupportStatus.NOT_EVALUATED,
        )

    def list_models(self) -> tuple[ProviderModel, ...]:
        models = [
            ProviderModel(
                provider_id=self.provider_id,
                model_id=model_id,
                capabilities=self.capabilities(model_id),
            )
            for model_id in self._model_catalog
        ]
        return tuple(sorted(models, key=lambda model: model.model_id))

    def health(self) -> ProviderHealth:
        try:
            self._resolve_api_key()
        except ProviderError:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderHealthStatus.UNAVAILABLE,
                code="credential_missing",
                model_count=len(self._model_catalog),
            )
        return ProviderHealth(
            provider_id=self.provider_id,
            status=ProviderHealthStatus.HEALTHY,
            code="ready",
            model_count=len(self._model_catalog),
        )

    # ------------------------------------------------------------------
    # Capability gating
    # ------------------------------------------------------------------

    def _reject_unsupported(self, invocation: ProviderInvocation) -> None:
        # capabilities() reports reasoning=False for every model, so an
        # extended-thinking opt-in must fail before any request leaves the
        # process.
        if invocation.enable_reasoning:
            raise ProviderError(provider_id=self.provider_id, code="reasoning_unsupported")

    # ------------------------------------------------------------------
    # Request encoding (Agent IR -> Anthropic Messages JSON)
    # ------------------------------------------------------------------

    def _encode_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    def _encode_message(self, message: Any) -> dict[str, Any]:
        if message.role == "tool":
            content: list[dict[str, Any]] = []
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise ProviderError(provider_id=self.provider_id, code="unsupported_content")
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.call_id,
                        "content": _text_of(block.content),
                    }
                )
            # The Messages API carries tool results in user-turn content.
            return {"role": "user", "content": content}
        if message.role not in ("user", "assistant"):
            raise ProviderError(provider_id=self.provider_id, code="unsupported_content")
        encoded_content: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                encoded_content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCallBlock):
                encoded_content.append(
                    {
                        "type": "tool_use",
                        "id": block.call_id,
                        "name": block.name,
                        "input": _arguments_input(block.arguments_raw),
                    }
                )
            elif isinstance(block, ToolResultBlock) and message.role == "user":
                # The Anthropic adapter decodes harness tool results as
                # user-turn ToolResultBlocks (role stays "user"), unlike the
                # OpenAI wire format which uses a dedicated "tool" role.
                encoded_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.call_id,
                        "content": _text_of(block.content),
                    }
                )
            else:
                # Images, documents, reasoning, refusals, and extension blocks
                # have no negotiated wire form here; dropping them silently
                # would lose request content.
                raise ProviderError(provider_id=self.provider_id, code="unsupported_content")
        return {"role": message.role, "content": encoded_content}

    def _encode_request(self, invocation: ProviderInvocation) -> dict[str, Any]:
        envelope = invocation.request
        request = envelope.request
        system_text = "\n".join(
            _text_of(instruction.content)
            for instruction in sorted(request.instructions, key=lambda item: item.order)
        ).strip()
        messages = [self._encode_message(message) for message in request.messages]
        payload: dict[str, Any] = {
            "model": invocation.model_id,
            "max_tokens": invocation.max_tokens_cap,
            "messages": messages,
        }
        if system_text:
            payload["system"] = system_text
        generation = request.generation
        if generation is not None:
            if generation.temperature is not None:
                payload["temperature"] = generation.temperature
            if generation.top_p is not None:
                payload["top_p"] = generation.top_p
            if generation.max_output_tokens is not None:
                payload["max_tokens"] = min(
                    generation.max_output_tokens, invocation.max_tokens_cap
                )
            if generation.stop:
                payload["stop_sequences"] = list(generation.stop)
            if generation.reasoning_effort is not None:
                raise ProviderError(provider_id=self.provider_id, code="reasoning_unsupported")
            # Anthropic has no seed parameter; the field is intentionally
            # dropped rather than sent as unknown API input.
        if request.tools:
            payload["tools"] = [self._encode_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            choice: Any = request.tool_choice
            if isinstance(choice, NamedToolChoice):
                choice = {"type": "tool", "name": choice.name}
            elif choice == "auto":
                choice = {"type": "auto"}
            else:
                # The Messages API has no "none"/"required" choice mapping
                # that stays safe across tool sets; reject instead of guessing.
                raise ProviderError(provider_id=self.provider_id, code="unsupported_tool_choice")
            payload["tool_choice"] = choice
        return payload

    # ------------------------------------------------------------------
    # HTTP execution with typed safe errors
    # ------------------------------------------------------------------

    def _build_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    def _post_buffered(self, invocation: ProviderInvocation) -> httpx.Response:
        payload = self._encode_request(invocation)
        try:
            with self._build_client() as client:
                return client.post("/messages", json=payload, headers=self._headers())
        except httpx.TimeoutException:
            raise ProviderError(provider_id=self.provider_id, code="timeout") from None
        except httpx.TransportError:
            raise ProviderError(provider_id=self.provider_id, code="network_error") from None
        except ProviderError:
            raise
        except Exception:
            raise ProviderError(
                provider_id=self.provider_id,
                code="provider_invoke_failed",
            ) from None

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        code = _STATUS_ERROR_CODES.get(response.status_code, "request_rejected")
        raise ProviderError(provider_id=self.provider_id, code=code)

    @staticmethod
    def _decode_usage(raw: Any) -> Usage | None:
        if not isinstance(raw, Mapping):
            return None

        def _count(key: str) -> int | None:
            value = raw.get(key)
            if type(value) is int and value >= 0:
                return value
            return None

        input_tokens = _count("input_tokens")
        output_tokens = _count("output_tokens")
        total_tokens = _count("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        # Agent IR rejects explicitly-null fields, so None counts are omitted
        # from the constructor instead of passed as null.
        usage_fields: dict[str, Any] = {"source": UsageSource.PROVIDER}
        if input_tokens is not None:
            usage_fields["input_tokens"] = input_tokens
        if output_tokens is not None:
            usage_fields["output_tokens"] = output_tokens
        if total_tokens is not None:
            usage_fields["total_tokens"] = total_tokens
        return Usage(**usage_fields)

    @staticmethod
    def _finish_reason(stop_reason: Any, *, has_calls: bool) -> str:
        if type(stop_reason) is str and stop_reason in _KNOWN_STOP_REASONS:
            return stop_reason
        return "tool_use" if has_calls else "end_turn"

    # ------------------------------------------------------------------
    # Buffered completion parsing (Agent IR events from one JSON document)
    # ------------------------------------------------------------------

    def _parse_completion_events(
        self, invocation: ProviderInvocation, body: bytes
    ) -> tuple[tuple[AgentEvent, ...], tuple[ProviderCallReference, ...], dict[str, str]]:
        if len(body) > self._max_response_bytes:
            raise ProviderError(provider_id=self.provider_id, code="response_too_large")
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response") from None
        if not isinstance(document, Mapping) or not isinstance(document.get("content"), list):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")

        request_id = invocation.request.request_id
        output_id = invocation.output_id or self._output_id_factory()
        sequence = invocation.sequence_start
        events: list[AgentEvent] = []
        calls: list[ProviderCallReference] = []
        source_call_ids: dict[str, str] = {}

        content_index = 0
        staged_calls: list[tuple[str, str, str]] = []
        for block in document["content"]:
            if not isinstance(block, Mapping):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ProviderError(provider_id=self.provider_id, code="invalid_response")
                if not text:
                    continue
                events.append(
                    ContentStartedEvent(
                        type="content.started",
                        request_id=request_id,
                        sequence=sequence,
                        choice_index=0,
                        output_id=output_id,
                        content_index=content_index,
                        content_type="text",
                    )
                )
                sequence += 1
                events.append(
                    ContentDeltaEvent(
                        type="content.delta",
                        request_id=request_id,
                        sequence=sequence,
                        choice_index=0,
                        output_id=output_id,
                        content_index=content_index,
                        delta=text,
                    )
                )
                sequence += 1
                events.append(
                    ContentCompletedEvent(
                        type="content.completed",
                        request_id=request_id,
                        sequence=sequence,
                        choice_index=0,
                        output_id=output_id,
                        content_index=content_index,
                        content=TextBlock(type="text", text=text),
                    )
                )
                sequence += 1
                content_index += 1
            elif block_type == "tool_use":
                name = block.get("name")
                source_call_id = block.get("id")
                arguments_input = block.get("input")
                if (
                    type(name) is not str
                    or not name
                    or not _safe_source_call_id(source_call_id)
                    or not isinstance(arguments_input, Mapping)
                ):
                    raise ProviderError(provider_id=self.provider_id, code="invalid_response")
                arguments_raw = json.dumps(
                    dict(arguments_input), separators=(",", ":"), sort_keys=True
                )
                local_call_id = self._call_id_factory()
                source_call_ids[local_call_id] = source_call_id
                staged_calls.append((local_call_id, name, arguments_raw))
            else:
                # thinking/redoable_thinking blocks were never enabled on this
                # request; anything but text/tool_use is malformed here.
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")

        group_id = self._parallel_group_factory() if len(staged_calls) > 1 else None
        for position, (local_call_id, name, arguments_raw) in enumerate(staged_calls):
            tool_common: dict[str, Any] = {
                "request_id": request_id,
                "choice_index": 0,
                "output_id": output_id,
                "tool_call_index": position,
                "call_id": local_call_id,
            }
            if group_id is not None:
                tool_common["parallel_group_id"] = group_id
            events.append(
                ToolCallStartedEvent(
                    type="tool_call.started",
                    sequence=sequence,
                    name=name,
                    **tool_common,
                )
            )
            sequence += 1
            events.append(
                ToolCallArgumentsDeltaEvent(
                    type="tool_call.arguments.delta",
                    sequence=sequence,
                    delta=arguments_raw,
                    **tool_common,
                )
            )
            sequence += 1
            completed_fields: dict[str, Any] = {}
            try:
                arguments_json = json.loads(arguments_raw)
            except json.JSONDecodeError:
                arguments_json = None
            if arguments_json is not None:
                completed_fields["arguments_json"] = arguments_json
            events.append(
                ToolCallCompletedEvent(
                    type="tool_call.completed",
                    sequence=sequence,
                    name=name,
                    arguments_raw=arguments_raw,
                    **completed_fields,
                    **tool_common,
                )
            )
            sequence += 1
            calls.append(
                ProviderCallReference(
                    call_id=local_call_id,
                    name=name,
                    choice_index=0,
                    output_id=output_id,
                    tool_call_index=position,
                    parallel_group_id=group_id,
                )
            )

        usage = self._decode_usage(document.get("usage"))
        # Agent IR rejects explicitly-null fields, so usage is only passed
        # when the provider supplied a parsable document.
        usage_fields: dict[str, Any] = {"usage": usage} if usage is not None else {}
        events.append(
            ResponseCompletedEvent(
                type="response.completed",
                request_id=request_id,
                sequence=sequence,
                choice_index=0,
                output_id=output_id,
                finish_reason=self._finish_reason(
                    document.get("stop_reason"), has_calls=bool(calls)
                ),
                **usage_fields,
            )
        )
        return tuple(events), tuple(calls), source_call_ids

    def invoke(self, invocation: ProviderInvocation) -> ProviderResult:
        if not isinstance(invocation, ProviderInvocation):
            raise ProviderError(provider_id=self.provider_id, code="invalid_invocation")
        self._reject_unsupported(invocation)
        handle = invocation.cancel_handle
        if handle is not None and handle.cancelled:
            return ProviderResult(
                provider_id=self.provider_id,
                model_id=invocation.model_id,
                events=(),
                streaming=ProviderStreamingMode.BUFFERED,
                cancelled=True,
            )
        response = self._post_buffered(invocation)
        self._raise_for_status(response)
        if handle is not None and handle.cancelled:
            return ProviderResult(
                provider_id=self.provider_id,
                model_id=invocation.model_id,
                events=(),
                streaming=ProviderStreamingMode.BUFFERED,
                cancelled=True,
            )
        try:
            events, calls, source_call_ids = self._parse_completion_events(
                invocation, response.content
            )
        except ProviderError:
            raise
        except Exception:
            raise ProviderError(
                provider_id=self.provider_id,
                code="provider_invoke_failed",
            ) from None
        return ProviderResult(
            provider_id=self.provider_id,
            model_id=invocation.model_id,
            events=events,
            calls=calls,
            source_call_ids=source_call_ids,
            streaming=ProviderStreamingMode.BUFFERED,
            cancelled=False,
        )

    # ------------------------------------------------------------------
    # Native SSE streaming
    # ------------------------------------------------------------------

    def stream(self, invocation: ProviderInvocation) -> Iterator[AgentEvent]:
        if not isinstance(invocation, ProviderInvocation):
            raise ProviderError(provider_id=self.provider_id, code="invalid_invocation")
        self._reject_unsupported(invocation)
        handle = invocation.cancel_handle
        if handle is not None and handle.cancelled:
            raise ProviderCancelledError(provider_id=self.provider_id)
        return self._iter_stream_events(invocation, handle)

    def _handle_stream_frame(self, chunk: Mapping[str, Any], state: dict[str, Any]) -> None:
        """Fold one SSE data frame into mutable stream ``state``."""
        event_type = chunk.get("type")
        if event_type == "message_start":
            message = chunk.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("usage"), Mapping):
                merged = dict(state["usage_document"] or {})
                merged.update(message["usage"])
                state["usage_document"] = merged
        elif event_type == "content_block_start":
            index = chunk.get("index")
            block = chunk.get("content_block")
            if type(index) is not int or index < 0 or not isinstance(block, Mapping):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            block_type = block.get("type")
            if block_type == "text":
                return
            if block_type != "tool_use" or index in state["tool_states"]:
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            name = block.get("name")
            source_call_id = block.get("id")
            if (
                type(name) is not str
                or not name
                or not _safe_source_call_id(source_call_id)
            ):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            state["tool_states"][index] = {
                "source_call_id": source_call_id,
                "local_call_id": self._call_id_factory(),
                "name": name,
                "fragments": [],
            }
        elif event_type == "content_block_delta":
            index = chunk.get("index")
            delta = chunk.get("delta")
            if type(index) is not int or index < 0 or not isinstance(delta, Mapping):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text_delta = delta.get("text")
                if not isinstance(text_delta, str):
                    raise ProviderError(provider_id=self.provider_id, code="invalid_response")
                if text_delta:
                    state["pending_text"].append(text_delta)
            elif delta_type == "input_json_delta":
                fragment_state = state["tool_states"].get(index)
                if fragment_state is None:
                    raise ProviderError(provider_id=self.provider_id, code="invalid_response")
                fragment = delta.get("partial_json")
                if not isinstance(fragment, str):
                    raise ProviderError(provider_id=self.provider_id, code="invalid_response")
                if fragment:
                    fragment_state["fragments"].append(fragment)
            else:
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        elif event_type == "message_delta":
            delta = chunk.get("delta")
            if delta is not None and not isinstance(delta, Mapping):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            if isinstance(delta, Mapping):
                candidate = delta.get("stop_reason")
                if candidate is not None:
                    state["stop_reason"] = candidate
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, Mapping):
                merged = dict(state["usage_document"] or {})
                merged.update(chunk_usage)
                state["usage_document"] = merged
        elif event_type == "error":
            # Provider-reported stream failure; the body itself is never
            # embedded in the typed error.
            raise ProviderError(provider_id=self.provider_id, code="provider_stream_failed")
        elif event_type not in ("content_block_stop", "message_stop", "ping"):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        if event_type == "message_stop":
            state["saw_message_stop"] = True

    def _iter_stream_events(
        self,
        invocation: ProviderInvocation,
        handle: ProviderCancellationHandle | None,
    ) -> Iterator[AgentEvent]:
        payload = self._encode_request(invocation)
        payload["stream"] = True
        request_id = invocation.request.request_id
        output_id = invocation.output_id or self._output_id_factory()

        state_sequence = invocation.sequence_start

        def _next_sequence() -> int:
            nonlocal state_sequence
            value = state_sequence
            state_sequence += 1
            return value

        def _cancelled() -> bool:
            return handle is not None and handle.cancelled

        # Tool-call lifecycles are staged here and flushed after the stream
        # ends: the parallel group identity is only known once every tool call
        # has been seen, and the Agent IR forbids changing it mid-lifecycle.
        state: dict[str, Any] = {
            "tool_states": {},
            "stop_reason": None,
            "usage_document": None,
            "saw_message_stop": False,
            "content_started": False,
            "content_chunks": [],
            "pending_text": [],
        }

        client: httpx.Client | None = None
        response: httpx.Response | None = None
        try:
            client = self._build_client()
            request = client.build_request(
                "POST",
                "/messages",
                json=payload,
                headers=self._headers(),
            )
            response = client.send(request, stream=True)
            self._raise_for_status(response)
            saw_message_stop = False
            total_bytes = 0
            for line in response.iter_lines():
                total_bytes += len(line.encode("utf-8")) + 1
                if total_bytes > self._max_response_bytes:
                    raise ProviderError(
                        provider_id=self.provider_id, code="response_too_large"
                    )
                if _cancelled():
                    # Typed cancellation at a chunk boundary: events already
                    # yielded stay valid; nothing is silently truncated.
                    raise ProviderCancelledError(provider_id=self.provider_id)
                if not line.startswith("data:"):
                    # event:/ping/keep-alive lines carry their meaning in the
                    # data frames below; comments carry no payload.
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    raise ProviderError(
                        provider_id=self.provider_id, code="invalid_response"
                    ) from None
                if not isinstance(chunk, Mapping) or type(chunk.get("type")) is not str:
                    raise ProviderError(
                        provider_id=self.provider_id, code="invalid_response"
                    )
                self._handle_stream_frame(chunk, state)
                # Flush text deltas as they arrive so consumers observe the
                # native SSE order for content (tools stay staged).
                while state["pending_text"]:
                    text_delta = state["pending_text"].pop(0)
                    if not state["content_started"]:
                        yield ContentStartedEvent(
                            type="content.started",
                            request_id=request_id,
                            sequence=_next_sequence(),
                            choice_index=0,
                            output_id=output_id,
                            content_index=0,
                            content_type="text",
                        )
                        state["content_started"] = True
                    state["content_chunks"].append(text_delta)
                    yield ContentDeltaEvent(
                        type="content.delta",
                        request_id=request_id,
                        sequence=_next_sequence(),
                        choice_index=0,
                        output_id=output_id,
                        content_index=0,
                        delta=text_delta,
                    )
                saw_message_stop = saw_message_stop or state["saw_message_stop"]
            del state["saw_message_stop"]
            if not saw_message_stop:
                # A 200 response whose stream ended without message_stop is
                # not an empty completion; it is a malformed response.
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            if _cancelled():
                raise ProviderCancelledError(provider_id=self.provider_id)
            if state["content_started"]:
                yield ContentCompletedEvent(
                    type="content.completed",
                    request_id=request_id,
                    sequence=_next_sequence(),
                    choice_index=0,
                    output_id=output_id,
                    content_index=0,
                    content=TextBlock(type="text", text="".join(state["content_chunks"])),
                )
            tool_states: dict[int, dict[str, Any]] = state["tool_states"]
            group_id = (
                self._parallel_group_factory() if len(tool_states) > 1 else None
            )
            for index in sorted(tool_states):
                fragment_state = tool_states[index]
                tool_common: dict[str, Any] = {
                    "request_id": request_id,
                    "choice_index": 0,
                    "output_id": output_id,
                    "tool_call_index": index,
                    "call_id": fragment_state["local_call_id"],
                }
                if group_id is not None:
                    tool_common["parallel_group_id"] = group_id
                yield ToolCallStartedEvent(
                    type="tool_call.started",
                    sequence=_next_sequence(),
                    name=fragment_state["name"],
                    **tool_common,
                )
                for fragment in fragment_state["fragments"]:
                    yield ToolCallArgumentsDeltaEvent(
                        type="tool_call.arguments.delta",
                        sequence=_next_sequence(),
                        delta=fragment,
                        **tool_common,
                    )
                arguments_raw = "".join(fragment_state["fragments"])
                completed_fields: dict[str, Any] = {}
                try:
                    arguments_json = json.loads(arguments_raw)
                except json.JSONDecodeError:
                    arguments_json = None
                if arguments_json is not None:
                    completed_fields["arguments_json"] = arguments_json
                yield ToolCallCompletedEvent(
                    type="tool_call.completed",
                    sequence=_next_sequence(),
                    name=fragment_state["name"],
                    arguments_raw=arguments_raw,
                    **completed_fields,
                    **tool_common,
                )
            usage = self._decode_usage(state["usage_document"])
            usage_fields: dict[str, Any] = (
                {"usage": usage} if usage is not None else {}
            )
            yield ResponseCompletedEvent(
                type="response.completed",
                request_id=request_id,
                sequence=_next_sequence(),
                choice_index=0,
                output_id=output_id,
                finish_reason=self._finish_reason(
                    state["stop_reason"], has_calls=bool(tool_states)
                ),
                **usage_fields,
            )
        except ProviderCancelledError:
            raise
        except ProviderError:
            raise
        except httpx.TimeoutException:
            raise ProviderError(provider_id=self.provider_id, code="timeout") from None
        except httpx.TransportError:
            raise ProviderError(provider_id=self.provider_id, code="network_error") from None
        except Exception:
            import traceback; traceback.print_exc()
            raise ProviderError(
                provider_id=self.provider_id,
                code="provider_invoke_failed",
            ) from None
        finally:
            if response is not None:
                response.close()
            if client is not None:
                client.close()


__all__ = ["AnthropicProvider"]
