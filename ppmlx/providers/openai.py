"""Remote OpenAI chat-completions implementation of the provider interface."""
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
    RefusalBlock,
    ResponseCompletedEvent,
    ResponseRefusedEvent,
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

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ENV_KEY = "OPENAI_API_KEY"
DEFAULT_MODEL_CATALOG = ("gpt-4o", "gpt-4o-mini")
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


class OpenAIProvider:
    """Adapter for the remote OpenAI chat-completions API.

    Data path is REMOTE and credentials are ``api_key`` per ADR 0004. The API
    key is resolved from the environment at request time and is never logged,
    embedded in errors, or included in reprs. Keychain-backed secret storage
    is intentionally out of scope for this adapter; only the ``env_key``
    pattern is supported here.

    Streaming iterates the SSE response live so cancellation is observed at
    chunk boundaries. Tool-call lifecycle events are staged until the stream
    ends because a parallel-group identifier is only known once every tool
    call has been seen, and the Agent IR requires one group identity per
    tool-call lifecycle.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        env_key: str = DEFAULT_ENV_KEY,
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
        self._model_catalog = catalog
        self._timeout_seconds = float(timeout_seconds)
        self._max_response_bytes = max_response_bytes
        self._transport = transport
        self._call_id_factory = call_id_factory
        self._output_id_factory = output_id_factory
        self._parallel_group_factory = parallel_group_factory

    @property
    def provider_id(self) -> str:
        return "openai"

    def _resolve_api_key(self) -> str:
        key = os.environ.get(self._env_key)
        if type(key) is not str or not key:
            raise ProviderError(provider_id=self.provider_id, code="credential_missing")
        return key

    def _headers(self) -> dict[str, str]:
        # The resolved key lives only in this short-lived header mapping. It
        # is never logged, embedded in exceptions, or exposed through reprs.
        return {
            "Authorization": f"Bearer {self._resolve_api_key()}",
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
        # capabilities() reports reasoning=False for every model, so a
        # reasoning opt-in must fail before any request leaves the process.
        if invocation.enable_reasoning:
            raise ProviderError(provider_id=self.provider_id, code="reasoning_unsupported")

    # ------------------------------------------------------------------
    # Request encoding (Agent IR -> OpenAI chat-completions JSON)
    # ------------------------------------------------------------------

    def _encode_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}

    def _encode_message(self, message: Any) -> list[dict[str, Any]]:
        if message.role == "tool":
            encoded_results: list[dict[str, Any]] = []
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise ProviderError(provider_id=self.provider_id, code="unsupported_content")
                encoded_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.call_id,
                        "content": _text_of(block.content),
                    }
                )
            return encoded_results
        text_parts: list[str] = []
        tool_calls_payload: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolCallBlock):
                tool_calls_payload.append(
                    {
                        "id": block.call_id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.arguments_raw,
                        },
                    }
                )
            else:
                # Images, documents, reasoning, refusals, and extension blocks
                # have no negotiated wire form here; dropping them silently
                # would lose request content.
                raise ProviderError(provider_id=self.provider_id, code="unsupported_content")
        encoded: dict[str, Any] = {"role": message.role, "content": "".join(text_parts)}
        if tool_calls_payload:
            encoded["tool_calls"] = tool_calls_payload
        return [encoded]

    def _encode_request(self, invocation: ProviderInvocation) -> dict[str, Any]:
        envelope = invocation.request
        request = envelope.request
        messages: list[dict[str, Any]] = []
        instruction_text = "\n".join(
            _text_of(instruction.content)
            for instruction in sorted(request.instructions, key=lambda item: item.order)
        ).strip()
        if instruction_text:
            messages.append({"role": "system", "content": instruction_text})
        for message in request.messages:
            messages.extend(self._encode_message(message))
        payload: dict[str, Any] = {
            "model": invocation.model_id,
            "messages": messages,
        }
        if request.tools:
            payload["tools"] = [self._encode_tool(tool) for tool in request.tools]
            payload["parallel_tool_calls"] = invocation.parallel_tool_calls
        if request.tool_choice is not None:
            choice: Any = request.tool_choice
            if isinstance(choice, NamedToolChoice):
                choice = {"type": "function", "function": {"name": choice.name}}
            payload["tool_choice"] = choice
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
                payload["stop"] = list(generation.stop)
            if generation.seed is not None:
                payload["seed"] = generation.seed
            if generation.reasoning_effort is not None:
                raise ProviderError(provider_id=self.provider_id, code="reasoning_unsupported")
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
        payload["stream"] = False
        try:
            with self._build_client() as client:
                return client.post(
                    "/chat/completions", json=payload, headers=self._headers()
                )
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

        usage = Usage(
            source=UsageSource.PROVIDER,
            input_tokens=_count("prompt_tokens"),
            output_tokens=_count("completion_tokens"),
            total_tokens=_count("total_tokens"),
        )
        if (
            usage.input_tokens is None
            and usage.output_tokens is None
            and usage.total_tokens is None
        ):
            return None
        return usage

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
        if (
            not isinstance(document, Mapping)
            or not isinstance(document.get("choices"), list)
            or not document["choices"]
        ):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        choice = document["choices"][0]
        if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        message = choice["message"]

        request_id = invocation.request.request_id
        output_id = invocation.output_id or self._output_id_factory()
        sequence = invocation.sequence_start
        events: list[AgentEvent] = []
        calls: list[ProviderCallReference] = []
        source_call_ids: dict[str, str] = {}
        usage = self._decode_usage(document.get("usage"))
        # Agent IR rejects explicitly-null fields, so usage is only passed
        # when the provider supplied a parsable document.
        usage_fields: dict[str, Any] = {"usage": usage} if usage is not None else {}
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            events.append(
                ResponseRefusedEvent(
                    type="response.refused",
                    request_id=request_id,
                    sequence=sequence,
                    choice_index=0,
                    output_id=output_id,
                    refusal=RefusalBlock(type="refusal", text=refusal),
                    **usage_fields,
                )
            )
            return tuple(events), (), {}

        text = message.get("content")
        if text is not None and not isinstance(text, str):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        if text:
            events.append(
                ContentStartedEvent(
                    type="content.started",
                    request_id=request_id,
                    sequence=sequence,
                    choice_index=0,
                    output_id=output_id,
                    content_index=0,
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
                    content_index=0,
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
                    content_index=0,
                    content=TextBlock(type="text", text=text),
                )
            )
            sequence += 1

        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is not None and not isinstance(raw_tool_calls, list):
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        parsed_tool_calls = list(raw_tool_calls or [])
        for item in parsed_tool_calls:
            if not isinstance(item, Mapping):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        parsed_tool_calls.sort(
            key=lambda item: item["index"] if type(item.get("index")) is int else 0
        )
        group_id = self._parallel_group_factory() if len(parsed_tool_calls) > 1 else None
        for position, item in enumerate(parsed_tool_calls):
            function = item.get("function")
            call_id = item.get("id")
            if (
                not isinstance(function, Mapping)
                or not _safe_source_call_id(call_id)
                or type(function.get("name")) is not str
                or not function["name"]
            ):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            name = function["name"]
            arguments_raw = function.get("arguments")
            if arguments_raw is None:
                arguments_raw = ""
            if not isinstance(arguments_raw, str):
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            local_call_id = self._call_id_factory()
            source_call_ids[local_call_id] = call_id
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

        finish_reason = choice.get("finish_reason")
        if finish_reason is None:
            finish_reason = "tool_calls" if calls else "stop"
        if type(finish_reason) is not str or not finish_reason:
            raise ProviderError(provider_id=self.provider_id, code="invalid_response")
        events.append(
            ResponseCompletedEvent(
                type="response.completed",
                request_id=request_id,
                sequence=sequence,
                choice_index=0,
                output_id=output_id,
                finish_reason=finish_reason,
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

        content_started = False
        content_chunks: list[str] = []
        # Tool-call lifecycles are staged here and flushed after the stream
        # ends: the parallel group identity is only known once every tool call
        # has been seen, and the Agent IR forbids changing it mid-lifecycle.
        tool_states: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_document: Mapping[str, Any] | None = None
        saw_data_frame = False
        total_bytes = 0

        client: httpx.Client | None = None
        response: httpx.Response | None = None
        try:
            client = self._build_client()
            request = client.build_request(
                "POST",
                "/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            response = client.send(request, stream=True)
            self._raise_for_status(response)
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
                    # SSE comments and keep-alive lines carry no payload.
                    continue
                saw_data_frame = True
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    raise ProviderError(
                        provider_id=self.provider_id, code="invalid_response"
                    ) from None
                if not isinstance(chunk, Mapping) or not isinstance(
                    chunk.get("choices"), list
                ):
                    raise ProviderError(
                        provider_id=self.provider_id, code="invalid_response"
                    )
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, Mapping):
                    usage_document = chunk_usage
                for choice in chunk["choices"]:
                    if not isinstance(choice, Mapping):
                        raise ProviderError(
                            provider_id=self.provider_id, code="invalid_response"
                        )
                    choice_index = choice.get("index", 0)
                    if type(choice_index) is not int or choice_index != 0:
                        # This adapter negotiates exactly one output choice.
                        raise ProviderError(
                            provider_id=self.provider_id, code="invalid_response"
                        )
                    delta = choice.get("delta")
                    finish_reason_value = choice.get("finish_reason")
                    if delta is None and finish_reason_value is None:
                        continue
                    if delta is not None and not isinstance(delta, Mapping):
                        raise ProviderError(
                            provider_id=self.provider_id, code="invalid_response"
                        )
                    if delta:
                        text_delta = delta.get("content")
                        if isinstance(text_delta, str) and text_delta:
                            if not content_started:
                                yield ContentStartedEvent(
                                    type="content.started",
                                    request_id=request_id,
                                    sequence=_next_sequence(),
                                    choice_index=0,
                                    output_id=output_id,
                                    content_index=0,
                                    content_type="text",
                                )
                                content_started = True
                            content_chunks.append(text_delta)
                            yield ContentDeltaEvent(
                                type="content.delta",
                                request_id=request_id,
                                sequence=_next_sequence(),
                                choice_index=0,
                                output_id=output_id,
                                content_index=0,
                                delta=text_delta,
                            )
                        raw_tool_deltas = delta.get("tool_calls") or []
                        if not isinstance(raw_tool_deltas, list):
                            raise ProviderError(
                                provider_id=self.provider_id, code="invalid_response"
                            )
                        for item in raw_tool_deltas:
                            if not isinstance(item, Mapping):
                                raise ProviderError(
                                    provider_id=self.provider_id,
                                    code="invalid_response",
                                )
                            index = item.get("index")
                            if type(index) is not int or index < 0:
                                raise ProviderError(
                                    provider_id=self.provider_id,
                                    code="invalid_response",
                                )
                            function_fragment = item.get("function")
                            if function_fragment is None:
                                function_fragment = {}
                            if not isinstance(function_fragment, Mapping):
                                raise ProviderError(
                                    provider_id=self.provider_id,
                                    code="invalid_response",
                                )
                            state = tool_states.get(index)
                            if state is None:
                                call_id = item.get("id")
                                name = function_fragment.get("name")
                                if not _safe_source_call_id(call_id):
                                    raise ProviderError(
                                        provider_id=self.provider_id,
                                        code="invalid_response",
                                    )
                                if type(name) is not str or not name:
                                    raise ProviderError(
                                        provider_id=self.provider_id,
                                        code="invalid_response",
                                    )
                                state = {
                                    "source_call_id": call_id,
                                    "local_call_id": self._call_id_factory(),
                                    "name": name,
                                    "fragments": [],
                                }
                                tool_states[index] = state
                            else:
                                later_id = item.get("id")
                                if later_id is not None and later_id != state["source_call_id"]:
                                    raise ProviderError(
                                        provider_id=self.provider_id,
                                        code="invalid_response",
                                    )
                                later_name = function_fragment.get("name")
                                if later_name is not None and later_name != state["name"]:
                                    raise ProviderError(
                                        provider_id=self.provider_id,
                                        code="invalid_response",
                                    )
                            argument_fragment = function_fragment.get("arguments")
                            if argument_fragment is not None:
                                if type(argument_fragment) is not str:
                                    raise ProviderError(
                                        provider_id=self.provider_id,
                                        code="invalid_response",
                                    )
                                if argument_fragment:
                                    state["fragments"].append(argument_fragment)
                    if type(finish_reason_value) is str and finish_reason_value:
                        finish_reason = finish_reason_value
            if not saw_data_frame:
                # A 200 response that never contained a data frame is not an
                # empty completion; it is a malformed response.
                raise ProviderError(provider_id=self.provider_id, code="invalid_response")
            if _cancelled():
                raise ProviderCancelledError(provider_id=self.provider_id)
            if content_started:
                yield ContentCompletedEvent(
                    type="content.completed",
                    request_id=request_id,
                    sequence=_next_sequence(),
                    choice_index=0,
                    output_id=output_id,
                    content_index=0,
                    content=TextBlock(type="text", text="".join(content_chunks)),
                )
            group_id = (
                self._parallel_group_factory() if len(tool_states) > 1 else None
            )
            for index in sorted(tool_states):
                state = tool_states[index]
                tool_common: dict[str, Any] = {
                    "request_id": request_id,
                    "choice_index": 0,
                    "output_id": output_id,
                    "tool_call_index": index,
                    "call_id": state["local_call_id"],
                }
                if group_id is not None:
                    tool_common["parallel_group_id"] = group_id
                yield ToolCallStartedEvent(
                    type="tool_call.started",
                    sequence=_next_sequence(),
                    name=state["name"],
                    **tool_common,
                )
                for fragment in state["fragments"]:
                    yield ToolCallArgumentsDeltaEvent(
                        type="tool_call.arguments.delta",
                        sequence=_next_sequence(),
                        delta=fragment,
                        **tool_common,
                    )
                arguments_raw = "".join(state["fragments"])
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
                    name=state["name"],
                    arguments_raw=arguments_raw,
                    **completed_fields,
                    **tool_common,
                )
            usage = self._decode_usage(usage_document)
            usage_fields: dict[str, Any] = (
                {"usage": usage} if usage is not None else {}
            )
            yield ResponseCompletedEvent(
                type="response.completed",
                request_id=request_id,
                sequence=_next_sequence(),
                choice_index=0,
                output_id=output_id,
                finish_reason=finish_reason
                or ("tool_calls" if tool_states else "stop"),
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
            raise ProviderError(
                provider_id=self.provider_id,
                code="provider_invoke_failed",
            ) from None
        finally:
            if response is not None:
                response.close()
            if client is not None:
                client.close()


__all__ = ["OpenAIProvider"]
