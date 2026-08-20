"""Opt-in streamed-tool coordinator for the strict local Agent IR runtime."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from ppmlx import __version__
from ppmlx.agent_ir import (
    AgentEvent,
    AgentIR,
    Protocol,
    RequestEnvelope,
    Source,
    ToolResultEvent,
    new_conversation_id,
    new_request_id,
)
from ppmlx.continuation import (
    CallRegistration,
    CallState,
    ContinuationLedger,
    ContinuationLedgerError,
    ContinuationOutcome,
    ContinuationProbe,
    ContinuationScope,
    ContinuationTicket,
    LedgerKey,
    ResultIdentity,
    RoutePin,
)
from ppmlx.protocols import (
    CallReference,
    DecodeContext,
    EncodeContext,
    NormalizationPolicy,
    ProtocolAdapter,
    ProtocolAdapterError,
    anthropic_messages_adapter,
    openai_chat_adapter,
    openai_responses_adapter,
)

from .backend import (
    LocalEngineRequest,
    LocalExecution,
    LocalGeneration,
    LocalGenerator,
    LocalRuntimeError,
    TerminalReasons,
    execute_local_request,
)
from .normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    select_normalization_profile,
)

if TYPE_CHECKING:
    from ppmlx.engine import TextEngine


SUPPORTED_PROTOCOLS = frozenset(
    {"openai-chat", "openai-responses", "anthropic-messages"}
)


class AgentRuntimeError(ValueError):
    """A safe strict-runtime error that can cross the HTTP boundary."""

    def __init__(self, code: str, *, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(f"Agent runtime error {code}")


@dataclass(frozen=True, slots=True)
class RuntimeScope:
    """The authenticated local listener scope for one request."""

    principal_id: str
    project_id: str
    harness_id: str

    def __post_init__(self) -> None:
        for value in (self.principal_id, self.project_id, self.harness_id):
            if (
                type(value) is not str
                or not 1 <= len(value) <= 128
                or any(ord(character) < 33 or ord(character) > 126 for character in value)
            ):
                raise AgentRuntimeError("invalid_runtime_scope")


@dataclass(frozen=True, slots=True)
class RuntimeResponse:
    """One complete native SSE response."""

    protocol: str
    conversation_id: str
    request_id: str
    native_response_id: str
    sse: str


@dataclass(slots=True)
class _ConversationRecord:
    source: Source
    route_pin: RoutePin
    requests: list[RequestEnvelope]
    events: list[AgentEvent]
    expires_at: float
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _RequestSemantics:
    parallel_tool_calls: bool = True
    include_usage: bool = True
    enable_thinking: bool = False


class _ContinuationJoin(Exception):
    def __init__(self, ticket: ContinuationTicket, cache_key: str) -> None:
        self.ticket = ticket
        self.cache_key = cache_key
        super().__init__("continuation join")


_ADAPTERS: Mapping[str, ProtocolAdapter] = MappingProxyType(
    {
        "openai-chat": openai_chat_adapter,
        "openai-responses": openai_responses_adapter,
        "anthropic-messages": anthropic_messages_adapter,
    }
)
_TERMINAL_REASONS = {
    "openai-chat": TerminalReasons(text="stop", tool_calls="tool_calls"),
    "openai-responses": TerminalReasons(text="completed", tool_calls="completed"),
    "anthropic-messages": TerminalReasons(text="end_turn", tool_calls="tool_use"),
}
_PROTOCOL_VERSIONS = {
    "openai-chat": "v1",
    "openai-responses": "v1",
    "anthropic-messages": "2023-06-01",
}
_RESPONSE_CACHE_SECONDS = 60.0
_CONTINUATION_JOIN_SECONDS = 120.0
_MAX_RESPONSE_CACHE_ITEMS = 64
_DEFAULT_MAX_ACTIVE_CONVERSATIONS = 128
_DEFAULT_MAX_CONVERSATION_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_TOTAL_CONVERSATION_BYTES = 64 * 1024 * 1024
_ANTHROPIC_CACHE_CONTROL = "anthropic-messages.cache_control"
_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppmlx-mlx")
_STRICT_ENGINE: TextEngine | None = None
_STRICT_ENGINE_LOCK = threading.Lock()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise AgentRuntimeError("invalid_native_request") from None


def _option_object(extensions: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = extensions.get(name)
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise AgentRuntimeError("unsupported_local_request_option")
    return value


def _request_semantics(
    envelope: RequestEnvelope,
    *,
    protocol: str,
) -> _RequestSemantics:
    request = envelope.request
    parallel_tool_calls = True
    include_usage = True
    enable_thinking = False

    if protocol == "openai-chat":
        options = _option_object(request.extensions, "openai-chat.request_options")
        if set(options) - {"n", "parallel_tool_calls", "store", "stream_options"}:
            raise AgentRuntimeError("unsupported_local_request_option")
        if "n" in options and (type(options["n"]) is not int or options["n"] != 1):
            raise AgentRuntimeError("unsupported_local_request_option")
        if "store" in options and options["store"] is not False:
            raise AgentRuntimeError("unsupported_local_storage")
        if "parallel_tool_calls" in options:
            if type(options["parallel_tool_calls"]) is not bool:
                raise AgentRuntimeError("unsupported_local_request_option")
            parallel_tool_calls = options["parallel_tool_calls"]
        if "stream_options" in options:
            stream_options = options["stream_options"]
            if not isinstance(stream_options, Mapping) or set(stream_options) - {"include_usage"}:
                raise AgentRuntimeError("unsupported_local_request_option")
            if "include_usage" in stream_options:
                if type(stream_options["include_usage"]) is not bool:
                    raise AgentRuntimeError("unsupported_local_request_option")
                include_usage = stream_options["include_usage"]
        reasoning_effort = (
            request.generation.reasoning_effort
            if request.generation is not None
            else None
        )
        if reasoning_effort not in {None, "none"}:
            raise AgentRuntimeError("local_reasoning_with_tools_unsupported")

    elif protocol == "openai-responses":
        options = _option_object(
            request.extensions,
            "openai-responses.request_options",
        )
        if set(options) - {"include", "parallel_tool_calls", "store"}:
            raise AgentRuntimeError("unsupported_local_request_option")
        if "store" in options and options["store"] is not False:
            raise AgentRuntimeError("unsupported_local_storage")
        if "parallel_tool_calls" in options:
            if type(options["parallel_tool_calls"]) is not bool:
                raise AgentRuntimeError("unsupported_local_request_option")
            parallel_tool_calls = options["parallel_tool_calls"]
        if "include" in options:
            include = options["include"]
            if not isinstance(include, list) or any(
                item != "reasoning.encrypted_content" for item in include
            ):
                raise AgentRuntimeError("unsupported_local_request_option")
        generation = request.generation
        if generation is not None:
            if generation.reasoning_effort not in {None, "none"}:
                raise AgentRuntimeError("local_reasoning_with_tools_unsupported")
            generation_options = _option_object(
                generation.extensions,
                "openai-responses.generation",
            )
            if generation_options:
                if set(generation_options) != {"reasoning"}:
                    raise AgentRuntimeError("unsupported_local_request_option")
                reasoning = generation_options["reasoning"]
                if not isinstance(reasoning, Mapping) or set(reasoning) - {"summary"}:
                    raise AgentRuntimeError("unsupported_local_request_option")
                if reasoning.get("summary") not in {None, "auto"}:
                    raise AgentRuntimeError("unsupported_local_request_option")

    elif protocol == "anthropic-messages":
        options = _option_object(
            request.extensions,
            "anthropic-messages.request_options",
        )
        if set(options) - {"disable_parallel_tool_use"}:
            raise AgentRuntimeError("unsupported_local_request_option")
        if "disable_parallel_tool_use" in options:
            disabled = options["disable_parallel_tool_use"]
            if type(disabled) is not bool:
                raise AgentRuntimeError("unsupported_local_request_option")
            parallel_tool_calls = not disabled
        generation = request.generation
        if generation is not None:
            generation_options = _option_object(
                generation.extensions,
                "anthropic-messages.generation",
            )
            if generation_options:
                if set(generation_options) - {
                    "context_management",
                    "output_config",
                    "thinking",
                }:
                    raise AgentRuntimeError("unsupported_local_request_option")
                thinking = generation_options.get("thinking")
                if thinking is not None:
                    if not isinstance(thinking, Mapping) or dict(thinking) != {
                        "type": "adaptive"
                    }:
                        raise AgentRuntimeError("unsupported_local_request_option")
                    enable_thinking = True
                context_management = generation_options.get("context_management")
                if context_management is not None and context_management != {
                    "edits": [
                        {"type": "clear_thinking_20251015", "keep": "all"}
                    ]
                }:
                    raise AgentRuntimeError("unsupported_local_request_option")
                output_config = generation_options.get("output_config")
                if output_config is not None:
                    if output_config != {"effort": "high"}:
                        raise AgentRuntimeError("unsupported_local_request_option")
                    enable_thinking = True
    else:
        raise AgentRuntimeError("unsupported_protocol")

    return _RequestSemantics(
        parallel_tool_calls=parallel_tool_calls,
        include_usage=include_usage,
        enable_thinking=enable_thinking,
    )


def _inherit_anthropic_cache_control(
    parent: RequestEnvelope,
    continuation: RequestEnvelope,
) -> RequestEnvelope:
    """Keep an ephemeral cache marker on repeated prior content."""

    parent_data = parent.model_dump(mode="json", exclude_unset=True)
    continuation_data = continuation.model_dump(mode="json", exclude_unset=True)

    def inherit(parent_value: object, continuation_value: object) -> None:
        if isinstance(parent_value, dict) and isinstance(continuation_value, dict):
            parent_extensions = parent_value.get("extensions")
            continuation_extensions = continuation_value.get("extensions")
            if (
                isinstance(parent_extensions, dict)
                and _ANTHROPIC_CACHE_CONTROL in parent_extensions
            ):
                if not isinstance(continuation_extensions, dict):
                    continuation_extensions = {}
                    continuation_value["extensions"] = continuation_extensions
                continuation_extensions.setdefault(
                    _ANTHROPIC_CACHE_CONTROL,
                    parent_extensions[_ANTHROPIC_CACHE_CONTROL],
                )
            for key in parent_value.keys() & continuation_value.keys():
                inherit(parent_value[key], continuation_value[key])
        elif isinstance(parent_value, list) and isinstance(continuation_value, list):
            for parent_item, continuation_item in zip(parent_value, continuation_value):
                inherit(parent_item, continuation_item)

    inherit(parent_data["request"], continuation_data["request"])
    try:
        return RequestEnvelope.model_validate(continuation_data)
    except ValueError:
        raise AgentRuntimeError("continuation_contract_mismatch") from None


def _continuation_request_id(protocol: str, native: Mapping[str, object]) -> str:
    digest = hashlib.sha256(protocol.encode("ascii") + b"\0" + _canonical_json(native)).hexdigest()
    return f"req_{digest[:32]}"


def _result_digest(event: ToolResultEvent) -> str:
    data = {
        "call_id": str(event.call_id),
        "content": [
            block.model_dump(mode="json", exclude_unset=True)
            for block in event.content
        ],
        "is_error": event.is_error,
    }
    return "sha256:" + hashlib.sha256(_canonical_json(data)).hexdigest()


def _native_response_ids(protocol: str) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:24]
    if protocol == "openai-chat":
        response_id = f"chatcmpl-{suffix}"
        return response_id, response_id
    if protocol == "anthropic-messages":
        response_id = f"msg_{suffix}"
        return response_id, response_id
    return f"resp_{suffix}", f"output_{suffix}"


def probe_tool_result_ids(
    protocol: str, native: Mapping[str, object]
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    """Find only explicit native tool-result identifiers before strict decode."""

    call_ids: list[str] = []
    output_ids: dict[str, str] = {}

    def add(call_id: object, output_id: object | None = None) -> None:
        if type(call_id) is not str or not call_id:
            raise AgentRuntimeError("invalid_tool_result_identity")
        if call_id in call_ids:
            raise AgentRuntimeError("duplicate_tool_result")
        call_ids.append(call_id)
        if output_id is not None:
            if type(output_id) is not str or not output_id:
                raise AgentRuntimeError("invalid_tool_result_identity")
            output_ids[call_id] = output_id

    if protocol == "openai-chat":
        messages = native.get("messages", [])
        if type(messages) is not list:
            raise AgentRuntimeError("invalid_native_request")
        for message in messages:
            if type(message) is dict and message.get("role") == "tool":
                add(message.get("tool_call_id"))
    elif protocol == "anthropic-messages":
        messages = native.get("messages", [])
        if type(messages) is not list:
            raise AgentRuntimeError("invalid_native_request")
        for message in messages:
            if type(message) is not dict:
                continue
            content = message.get("content", [])
            if type(content) is not list:
                continue
            for block in content:
                if type(block) is dict and block.get("type") == "tool_result":
                    add(block.get("tool_use_id"))
    elif protocol == "openai-responses":
        items = native.get("input", [])
        if type(items) is not list:
            raise AgentRuntimeError("invalid_native_request")
        for item in items:
            if type(item) is dict and item.get("type") == "function_call_output":
                add(item.get("call_id"), item.get("id"))
    else:
        raise AgentRuntimeError("unsupported_protocol")
    return tuple(call_ids), MappingProxyType(output_ids)


def default_local_generator(request: LocalEngineRequest) -> LocalGeneration:
    """Run the current MLX text engine through its strict tool surface."""

    global _STRICT_ENGINE
    if _STRICT_ENGINE is None:
        with _STRICT_ENGINE_LOCK:
            if _STRICT_ENGINE is None:
                from ppmlx.engine import TextEngine

                _STRICT_ENGINE = TextEngine(max_loaded=1)
    engine = _STRICT_ENGINE
    result = engine.generate(
        request.model,
        [dict(message) for message in request.messages],
        temperature=0.7 if request.temperature is None else request.temperature,
        top_p=1.0 if request.top_p is None else request.top_p,
        max_tokens=request.max_tokens,
        stop=list(request.stop) if request.stop is not None else None,
        seed=request.seed,
        strip_thinking=True,
        enable_thinking=request.enable_thinking,
        tools=[dict(tool) for tool in request.tools],
        strict_tools=True,
    )
    return LocalGeneration(
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def default_model_resolver(model: str, protocol: str) -> str:
    """Resolve a request to a model that is already present on this host."""

    if protocol not in SUPPORTED_PROTOCOLS:
        raise AgentRuntimeError("unsupported_protocol")
    try:
        from ppmlx.models import get_model_path, resolve_alias
    except Exception:
        raise AgentRuntimeError("local_model_unavailable", status_code=503) from None

    def local_model(candidate: str) -> str | None:
        try:
            resolved = resolve_alias(candidate)
            if get_model_path(resolved) is not None:
                return resolved
        except Exception:
            return None
        return None

    requested = local_model(model)
    if requested is not None:
        return requested
    try:
        from ppmlx.config import load_config

        configured = local_model(load_config().defaults.model)
    except Exception:
        configured = None
    if configured is not None:
        return configured
    raise AgentRuntimeError("local_model_unavailable", status_code=503)


class LocalAgentRuntime:
    """Coordinate strict adapters, local inference, Agent IR, and continuations."""

    def __init__(
        self,
        *,
        ledger: ContinuationLedger | None = None,
        generate: LocalGenerator = default_local_generator,
        resolve_model: Callable[[str, str], str] = default_model_resolver,
        max_tokens_cap: int = 32_768,
        conversation_ttl_seconds: int = 86_400,
        max_active_conversations: int = _DEFAULT_MAX_ACTIVE_CONVERSATIONS,
        max_conversation_bytes: int = _DEFAULT_MAX_CONVERSATION_BYTES,
        max_total_conversation_bytes: int = _DEFAULT_MAX_TOTAL_CONVERSATION_BYTES,
        max_concurrent_generations: int = 1,
    ) -> None:
        for value in (
            max_tokens_cap,
            conversation_ttl_seconds,
            max_active_conversations,
            max_conversation_bytes,
            max_total_conversation_bytes,
            max_concurrent_generations,
        ):
            if type(value) is not int or value < 1:
                raise AgentRuntimeError("invalid_runtime_limit")
        self._ledger = ledger or ContinuationLedger(
            active_ttl_seconds=conversation_ttl_seconds,
        )
        self._generate = generate
        self._resolve_model = resolve_model
        self._max_tokens_cap = max_tokens_cap
        self._conversation_ttl_seconds = conversation_ttl_seconds
        self._max_active_conversations = max_active_conversations
        self._max_conversation_bytes = max_conversation_bytes
        self._max_total_conversation_bytes = max_total_conversation_bytes
        self._response_cache_seconds = min(
            _RESPONSE_CACHE_SECONDS,
            float(conversation_ttl_seconds),
        )
        self._generation_gate = threading.BoundedSemaphore(max_concurrent_generations)
        self._lock = threading.RLock()
        self._conversations: dict[tuple[str, str, str, str], _ConversationRecord] = {}
        self._conversation_bytes = 0
        self._response_cache: dict[str, tuple[float, RuntimeResponse]] = {}
        self._pending_conversations = 0

    @property
    def ledger(self) -> ContinuationLedger:
        return self._ledger

    def execute(
        self,
        native: Mapping[str, object],
        *,
        protocol: str,
        scope: RuntimeScope,
    ) -> RuntimeResponse:
        """Run one buffered streamed-tool turn without executing a harness tool."""

        try:
            return self._execute(native, protocol=protocol, scope=scope)
        except _ContinuationJoin as pending:
            try:
                pending.ticket.result(timeout=_CONTINUATION_JOIN_SECONDS)
            except FutureTimeoutError:
                raise AgentRuntimeError("continuation_wait_timeout", status_code=504) from None
            cached = self._cached_response(pending.cache_key)
            if cached is None:
                raise AgentRuntimeError("continuation_failed", status_code=409)
            return cached
        except Exception as error:
            raise self._safe_error(error) from None

    async def execute_async(
        self,
        native: Mapping[str, object],
        *,
        protocol: str,
        scope: RuntimeScope,
    ) -> RuntimeResponse:
        """Run one turn and join duplicate continuations without blocking a worker."""

        try:
            return await asyncio.to_thread(
                self._execute,
                native,
                protocol=protocol,
                scope=scope,
            )
        except _ContinuationJoin as pending:
            try:
                await pending.ticket.wait(timeout=_CONTINUATION_JOIN_SECONDS)
            except TimeoutError:
                raise AgentRuntimeError("continuation_wait_timeout", status_code=504) from None
            cached = self._cached_response(pending.cache_key)
            if cached is None:
                raise AgentRuntimeError("continuation_failed", status_code=409)
            return cached
        except Exception as error:
            raise self._safe_error(error) from None

    @staticmethod
    def _safe_error(error: Exception) -> AgentRuntimeError:
        if isinstance(error, AgentRuntimeError):
            return error
        if isinstance(error, ProtocolAdapterError):
            return AgentRuntimeError(error.code)
        if isinstance(error, LocalRuntimeError):
            status = 503 if error.code == "generation_failed" else 400
            return AgentRuntimeError(error.code, status_code=status)
        if isinstance(error, ToolNormalizationError):
            return AgentRuntimeError(error.code)
        if isinstance(error, ContinuationLedgerError):
            code = (
                "tool_continuation_expired"
                if error.code == "tool_call_unknown"
                else error.code
            )
            status = 503 if code == "continuation_capacity_exceeded" else 400
            return AgentRuntimeError(code, status_code=status)
        if isinstance(error, (TypeError, ValueError)):
            return AgentRuntimeError("invalid_agent_runtime_state")
        return AgentRuntimeError("agent_runtime_failed", status_code=500)

    def _execute(
        self,
        native: Mapping[str, object],
        *,
        protocol: str,
        scope: RuntimeScope,
    ) -> RuntimeResponse:
        if protocol not in SUPPORTED_PROTOCOLS:
            raise AgentRuntimeError("unsupported_protocol")
        if not isinstance(native, Mapping):
            raise AgentRuntimeError("invalid_native_request")
        adapter = _ADAPTERS[protocol]
        harness = f"{protocol}:{scope.harness_id}"
        continuation_scope = ContinuationScope(
            principal_id=scope.principal_id,
            project_id=scope.project_id,
            harness=harness,
        )
        native_call_ids, native_result_output_ids = probe_tool_result_ids(protocol, native)
        call_ids = native_call_ids
        policy = NormalizationPolicy()
        ticket = None
        slot_reserved = False
        result_digests: dict[str, str] = {}
        response_cache_key: str | None = None

        if native_call_ids:
            request_id = _continuation_request_id(protocol, native)
            response_cache_key = self._cache_key(scope, protocol, request_id)
            cached = self._cached_response(response_cache_key)
            if cached is not None:
                return cached
            probe, call_ids = self._probe_pending_calls(
                continuation_scope,
                native_call_ids,
                native_result_output_ids,
            )
            conversation_id = probe.conversation_id
            decoded = adapter.decode_request(
                native,
                context=DecodeContext(
                    request_id=request_id,
                    kind="continuation",
                    parent_request_id=probe.parent_request_id,
                    prior_calls=probe.prior_calls,
                    result_output_ids=probe.result_output_ids,
                    policy=policy,
                ),
            )
            tool_results = tuple(
                event for event in decoded.tool_results if str(event.call_id) in call_ids
            )
            request_envelope = self._preflight_continuation(
                protocol=protocol,
                harness=harness,
                scope=scope,
                conversation_id=conversation_id,
                route_pin=probe.route_pin,
                request=decoded.request,
                tool_results=tool_results,
            )
            events_by_call = {str(event.call_id): event for event in tool_results}
            if set(events_by_call) != set(call_ids):
                raise AgentRuntimeError("tool_result_identity_mismatch")
            for call_id in call_ids:
                event = events_by_call[call_id]
                digest = _result_digest(event)
                result_digests[call_id] = digest
            for call_id in call_ids:
                event = events_by_call[call_id]
                digest = result_digests[call_id]
                reference = probe.prior_calls[call_id]
                key = LedgerKey(
                    principal_id=scope.principal_id,
                    project_id=scope.project_id,
                    harness=harness,
                    conversation_id=conversation_id,
                    call_id=call_id,
                )
                self._ledger.accept_result(
                    key,
                    ResultIdentity(
                        request_id=request_id,
                        parent_request_id=probe.parent_request_id,
                        choice_index=reference.choice_index,
                        tool_call_index=reference.tool_call_index,
                        result_digest=digest,
                        source_output_id=str(event.output_id),
                    ),
                )
            ticket = self._ledger.acquire_group_continuation(
                continuation_scope,
                call_ids,
                result_digests=result_digests,
            )
            if ticket.disposition != "owner":
                cached = self._cached_response(response_cache_key)
                if cached is not None:
                    return cached
                if ticket.disposition == "join":
                    raise _ContinuationJoin(ticket, response_cache_key)
                raise AgentRuntimeError("continuation_already_completed", status_code=409)
            route_pin = probe.route_pin
        else:
            request_id = new_request_id()
            conversation_id = new_conversation_id()
            decoded = adapter.decode_request(
                native,
                context=DecodeContext(
                    request_id=request_id,
                    kind="initial",
                    policy=policy,
                ),
            )
            request_envelope = decoded.request
            tool_results = ()
            requested_model = str(decoded.request.request.model)
            local_model = self._resolve_model(requested_model, protocol)
            route_pin = RoutePin(
                decision_id=f"decision_{request_id}",
                provider="local-mlx",
                model=local_model,
                candidate_id="candidate_" + hashlib.sha256(local_model.encode()).hexdigest()[:16],
            )

        semantics = _request_semantics(request_envelope, protocol=protocol)
        local_model = route_pin.model
        profile = select_normalization_profile(local_model)
        native_response_id, output_id = _native_response_ids(protocol)
        if not call_ids:
            self._reserve_conversation_slot()
            slot_reserved = True

        try:
            execution = self._execute_generation(
                request_envelope,
                model=local_model,
                profile=profile,
                terminal_reasons=_TERMINAL_REASONS[protocol],
                output_id=output_id,
                sequence_start=(
                    max(event.sequence for event in tool_results) + 1
                    if tool_results
                    else 0
                ),
                enable_thinking=semantics.enable_thinking,
                parallel_tool_calls=semantics.parallel_tool_calls,
            )
            record = self._validate_conversation(
                protocol=protocol,
                harness=harness,
                scope=scope,
                conversation_id=conversation_id,
                route_pin=route_pin,
                request=request_envelope,
                tool_results=tool_results,
                execution=execution,
            )
            sse = adapter.encode_stream(
                execution.events,
                context=EncodeContext(
                    model=str(request_envelope.request.model),
                    created_at=0,
                    response_id=native_response_id,
                    include_usage=semantics.include_usage,
                    parallel_tool_calls=semantics.parallel_tool_calls,
                ),
            )
            response = RuntimeResponse(
                protocol=protocol,
                conversation_id=conversation_id,
                request_id=request_id,
                native_response_id=native_response_id,
                sse=sse,
            )
            if execution.calls:
                self._activate_conversation(
                    scope=scope,
                    harness=harness,
                    conversation_id=conversation_id,
                    route_pin=route_pin,
                    record=record,
                    execution=execution,
                )
            elif call_ids:
                with self._lock:
                    self._drop_conversation(
                        (scope.principal_id, scope.project_id, harness, conversation_id)
                    )
            if response_cache_key is not None:
                self._cache_response(response_cache_key, response)
            if ticket is not None:
                first_key = LedgerKey(
                    principal_id=scope.principal_id,
                    project_id=scope.project_id,
                    harness=harness,
                    conversation_id=conversation_id,
                    call_id=call_ids[0],
                )
                self._ledger.complete_continuation(
                    first_key,
                    ContinuationOutcome(state=CallState.RESOLVED),
                )
            return response
        except Exception:
            if ticket is not None and ticket.disposition == "owner":
                first_key = LedgerKey(
                    principal_id=scope.principal_id,
                    project_id=scope.project_id,
                    harness=harness,
                    conversation_id=conversation_id,
                    call_id=call_ids[0],
                )
                try:
                    self._ledger.complete_continuation(
                        first_key,
                        ContinuationOutcome(
                            state=CallState.ABANDONED,
                            error_code="continuation_failed",
                        ),
                    )
                except ContinuationLedgerError:
                    pass
                with self._lock:
                    self._drop_conversation(
                        (scope.principal_id, scope.project_id, harness, conversation_id)
                    )
            raise
        finally:
            if slot_reserved:
                self._release_conversation_slot()

    def _probe_pending_calls(
        self,
        scope: ContinuationScope,
        call_ids: Sequence[str],
        result_output_ids: Mapping[str, str],
    ) -> tuple[ContinuationProbe, tuple[str, ...]]:
        conversations: set[str] = set()
        routes: set[RoutePin] = set()
        references: dict[str, CallReference] = {}
        outputs: dict[str, str] = {}
        pending: list[str] = []
        for call_id in call_ids:
            supplied_output = (
                {call_id: result_output_ids[call_id]}
                if call_id in result_output_ids
                else None
            )
            probe = self._ledger.probe_calls(
                scope,
                (call_id,),
                result_output_ids=supplied_output,
            )
            conversations.add(probe.conversation_id)
            routes.add(probe.route_pin)
            references.update(probe.prior_calls)
            outputs.update(probe.result_output_ids)
            key = LedgerKey(
                principal_id=scope.principal_id,
                project_id=scope.project_id,
                harness=scope.harness,
                conversation_id=probe.conversation_id,
                call_id=call_id,
            )
            if self._ledger.get(key).state not in {CallState.RESOLVED, CallState.ABANDONED}:
                pending.append(call_id)
        if len(conversations) != 1 or len(routes) != 1 or not pending:
            raise AgentRuntimeError("tool_continuation_expired", status_code=409)
        pending_outputs = {
            call_id: result_output_ids[call_id]
            for call_id in pending
            if call_id in result_output_ids
        }
        current = self._ledger.probe_calls(
            scope,
            tuple(pending),
            result_output_ids=pending_outputs,
        )
        return (
            ContinuationProbe(
                conversation_id=current.conversation_id,
                parent_request_id=current.parent_request_id,
                route_pin=current.route_pin,
                prior_calls=MappingProxyType(references),
                result_output_ids=MappingProxyType(outputs),
            ),
            tuple(pending),
        )

    def _preflight_continuation(
        self,
        *,
        protocol: str,
        harness: str,
        scope: RuntimeScope,
        conversation_id: str,
        route_pin: RoutePin,
        request: RequestEnvelope,
        tool_results: Sequence[ToolResultEvent],
    ) -> RequestEnvelope:
        key = (scope.principal_id, scope.project_id, harness, conversation_id)
        with self._lock:
            self._prune_conversations()
            current = self._conversations.get(key)
            if current is None:
                raise AgentRuntimeError("tool_continuation_expired", status_code=409)
            if current.route_pin != route_pin:
                raise AgentRuntimeError("route_pin_changed")
            inherited = (
                _inherit_anthropic_cache_control(current.requests[-1], request)
                if protocol == "anthropic-messages"
                else request
            )
            self._validate_agent_ir(
                conversation_id=conversation_id,
                source=current.source,
                requests=[*current.requests, inherited],
                events=[*current.events, *tool_results],
                error_code="continuation_contract_mismatch",
            )
            return inherited

    def _validate_conversation(
        self,
        *,
        protocol: str,
        harness: str,
        scope: RuntimeScope,
        conversation_id: str,
        route_pin: RoutePin,
        request: RequestEnvelope,
        tool_results: Sequence[ToolResultEvent],
        execution: LocalExecution,
    ) -> _ConversationRecord:
        key = (scope.principal_id, scope.project_id, harness, conversation_id)
        with self._lock:
            self._prune_conversations()
            current = self._conversations.get(key)
            if current is None:
                source = Source(
                    harness=scope.harness_id,
                    harness_version=__version__,
                    protocol=Protocol(protocol),
                    protocol_version=_PROTOCOL_VERSIONS[protocol],
                )
                requests = [request]
                events: list[AgentEvent] = [*tool_results, *execution.events]
            else:
                if current.route_pin != route_pin:
                    raise AgentRuntimeError("route_pin_changed")
                source = current.source
                requests = [*current.requests, request]
                events = [*current.events, *tool_results, *execution.events]
            size_bytes = self._validate_agent_ir(
                conversation_id=conversation_id,
                source=source,
                requests=requests,
                events=events,
                error_code="agent_ir_validation_failed",
            )
            record = _ConversationRecord(
                source=source,
                route_pin=route_pin,
                requests=requests,
                events=events,
                expires_at=time.monotonic() + self._conversation_ttl_seconds,
                size_bytes=size_bytes,
            )
            return record

    def _validate_agent_ir(
        self,
        *,
        conversation_id: str,
        source: Source,
        requests: Sequence[RequestEnvelope],
        events: Sequence[AgentEvent],
        error_code: str,
    ) -> int:
        payload = {
            "ir_version": "agent-ir/v1",
            "conversation_id": conversation_id,
            "source": source.model_dump(mode="json", exclude_unset=True),
            "requests": [item.model_dump(mode="json", exclude_unset=True) for item in requests],
            "events": [item.model_dump(mode="json", exclude_unset=True) for item in events],
        }
        size_bytes = len(_canonical_json(payload))
        if size_bytes > self._max_conversation_bytes:
            raise AgentRuntimeError("conversation_limit_exceeded")
        try:
            AgentIR.model_validate(payload)
        except ValueError:
            raise AgentRuntimeError(error_code) from None
        return size_bytes

    def _execute_generation(
        self,
        request: RequestEnvelope,
        *,
        model: str,
        profile: NormalizationProfile | None,
        terminal_reasons: TerminalReasons,
        output_id: str,
        sequence_start: int,
        enable_thinking: bool,
        parallel_tool_calls: bool,
    ) -> LocalExecution:
        if not self._generation_gate.acquire(blocking=False):
            raise AgentRuntimeError("agent_runtime_busy", status_code=503)
        try:
            future = _MLX_EXECUTOR.submit(
                execute_local_request,
                request,
                model=model,
                generate=self._generate,
                profile=profile,
                terminal_reasons=terminal_reasons,
                output_id=output_id,
                sequence_start=sequence_start,
                max_tokens_cap=self._max_tokens_cap,
                enable_thinking=enable_thinking,
                parallel_tool_calls=parallel_tool_calls,
            )
            return future.result()
        finally:
            self._generation_gate.release()

    def _reserve_conversation_slot(self) -> None:
        with self._lock:
            self._prune_conversations()
            if (
                len(self._conversations) + self._pending_conversations
                >= self._max_active_conversations
            ):
                raise AgentRuntimeError("conversation_capacity_exceeded", status_code=503)
            self._pending_conversations += 1

    def _release_conversation_slot(self) -> None:
        with self._lock:
            if self._pending_conversations < 1:  # pragma: no cover - internal invariant
                raise RuntimeError("A conversation reservation is missing")
            self._pending_conversations -= 1

    def _activate_conversation(
        self,
        *,
        scope: RuntimeScope,
        harness: str,
        conversation_id: str,
        route_pin: RoutePin,
        record: _ConversationRecord,
        execution: LocalExecution,
    ) -> None:
        key = (scope.principal_id, scope.project_id, harness, conversation_id)
        with self._lock:
            self._prune_conversations()
            if key not in self._conversations and len(self._conversations) >= self._max_active_conversations:
                raise AgentRuntimeError("conversation_capacity_exceeded", status_code=503)
            previous = self._conversations.get(key)
            previous_size = previous.size_bytes if previous is not None else 0
            projected_bytes = self._conversation_bytes - previous_size + record.size_bytes
            if projected_bytes > self._max_total_conversation_bytes:
                raise AgentRuntimeError("conversation_capacity_exceeded", status_code=503)
            self._register_calls(
                scope=scope,
                harness=harness,
                conversation_id=conversation_id,
                route_pin=route_pin,
                record=record,
                execution=execution,
            )
            self._conversations[key] = record
            self._conversation_bytes = projected_bytes

    def _register_calls(
        self,
        *,
        scope: RuntimeScope,
        harness: str,
        conversation_id: str,
        route_pin: RoutePin,
        record: _ConversationRecord,
        execution: LocalExecution,
    ) -> None:
        initial_request_id = str(record.requests[0].request_id)
        prior_request_ids = tuple(str(item.request_id) for item in record.requests[1:])
        registrations: list[CallRegistration] = []
        for reference in execution.calls:
            key = LedgerKey(
                principal_id=scope.principal_id,
                project_id=scope.project_id,
                harness=harness,
                conversation_id=conversation_id,
                call_id=reference.call_id,
            )
            registrations.append(
                CallRegistration(
                    key=key,
                    source_call_id=execution.source_call_ids.get(reference.call_id),
                    tool_name=reference.name,
                    initial_request_id=initial_request_id,
                    prior_continuation_request_ids=prior_request_ids,
                    choice_index=reference.choice_index,
                    output_id=reference.output_id,
                    tool_call_index=reference.tool_call_index,
                    parallel_group_id=reference.parallel_group_id,
                    route_pin=route_pin,
                )
            )
        self._ledger.register_calls(registrations)
        for registration in registrations:
            key = registration.key
            self._ledger.mark_arguments_complete(key)
            self._ledger.mark_waiting_for_result(key)

    @staticmethod
    def _cache_key(
        scope: RuntimeScope,
        protocol: str,
        request_id: str,
    ) -> str:
        value = {
            "principal": scope.principal_id,
            "project": scope.project_id,
            "harness": scope.harness_id,
            "protocol": protocol,
            "request": request_id,
        }
        return hashlib.sha256(_canonical_json(value)).hexdigest()

    def _prune_conversations(self) -> None:
        now = time.monotonic()
        for key, record in tuple(self._conversations.items()):
            if now >= record.expires_at:
                self._drop_conversation(key)
        self._ledger.cleanup()

    def _drop_conversation(self, key: tuple[str, str, str, str]) -> None:
        record = self._conversations.pop(key, None)
        if record is not None:
            self._conversation_bytes -= record.size_bytes

    def _cache_response(self, key: str, response: RuntimeResponse) -> None:
        with self._lock:
            self._prune_cache()
            if len(self._response_cache) >= _MAX_RESPONSE_CACHE_ITEMS:
                oldest = min(self._response_cache, key=lambda item: self._response_cache[item][0])
                self._response_cache.pop(oldest, None)
            self._response_cache[key] = (
                time.monotonic() + self._response_cache_seconds,
                response,
            )

    def _cached_response(self, key: str) -> RuntimeResponse | None:
        with self._lock:
            self._prune_cache()
            cached = self._response_cache.get(key)
            return cached[1] if cached is not None else None

    def _prune_cache(self) -> None:
        now = time.monotonic()
        for key, (expires_at, _) in tuple(self._response_cache.items()):
            if now >= expires_at:
                self._response_cache.pop(key, None)


_runtime: LocalAgentRuntime | None = None
_runtime_lock = threading.Lock()


def get_local_agent_runtime(
    *,
    continuation_ttl_seconds: int = 86400,
    max_tokens_cap: int = 32_768,
) -> LocalAgentRuntime:
    """Return the process-local strict runtime singleton."""

    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = LocalAgentRuntime(
                    ledger=ContinuationLedger(active_ttl_seconds=continuation_ttl_seconds),
                    conversation_ttl_seconds=continuation_ttl_seconds,
                    max_tokens_cap=max_tokens_cap,
                )
    return _runtime


def reset_local_agent_runtime() -> None:
    """Reset the process-local runtime for tests and explicit shutdown."""

    global _runtime
    with _runtime_lock:
        _runtime = None
    _MLX_EXECUTOR.submit(_reset_strict_engine).result()


def _reset_strict_engine() -> None:
    global _STRICT_ENGINE
    with _STRICT_ENGINE_LOCK:
        if _STRICT_ENGINE is not None:
            _STRICT_ENGINE.unload_all()
            _STRICT_ENGINE = None


__all__ = [
    "AgentRuntimeError",
    "LocalAgentRuntime",
    "RuntimeResponse",
    "RuntimeScope",
    "SUPPORTED_PROTOCOLS",
    "default_local_generator",
    "default_model_resolver",
    "get_local_agent_runtime",
    "probe_tool_result_ids",
    "reset_local_agent_runtime",
]
