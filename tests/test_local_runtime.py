from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Mapping

import pytest

from ppmlx import continuation as continuation_module
from ppmlx.local_runtime import runtime as runtime_module
from ppmlx.local_runtime.backend import LocalEngineRequest, LocalGeneration
from ppmlx.local_runtime.runtime import AgentRuntimeError, LocalAgentRuntime, RuntimeResponse, RuntimeScope
from ppmlx.continuation import CallState, ContinuationLedger, ContinuationTicket, LedgerKey
from ppmlx.protocols.sse import SSEFrame, parse_sse


CONTRACTS = Path(__file__).parent / "fixtures" / "contracts"
SCOPE = RuntimeScope(
    principal_id="principal_a",
    project_id="project_a",
    harness_id="harness_a",
)
FINAL_TEXT = "The tool returned fixture-ok."


@dataclass(frozen=True, slots=True)
class RuntimeCase:
    protocol: str
    fixture: Path
    tool_name: str
    arguments: Mapping[str, str]
    initial_lifecycle: tuple[str, ...]
    final_lifecycle: tuple[str, ...]


CASES = (
    RuntimeCase(
        protocol="openai-chat",
        fixture=CONTRACTS / "openai-chat" / "opencode-1.18.18",
        tool_name="bash",
        arguments={"command": "printf fixture-ok"},
        initial_lifecycle=("assistant", "tool_start", "tool_delta", "tool_calls", "[DONE]"),
        final_lifecycle=("assistant", "text_delta", "stop", "[DONE]"),
    ),
    RuntimeCase(
        protocol="anthropic-messages",
        fixture=CONTRACTS / "anthropic-messages" / "claude-code-2.1.231",
        tool_name="Bash",
        arguments={"command": "printf fixture-ok"},
        initial_lifecycle=(
            "message_start",
            "content_block_start:tool_use",
            "content_block_delta:input_json_delta",
            "content_block_stop",
            "message_delta:tool_use",
            "message_stop",
        ),
        final_lifecycle=(
            "message_start",
            "content_block_start:text",
            "content_block_delta:text_delta",
            "content_block_stop",
            "message_delta:end_turn",
            "message_stop",
        ),
    ),
    RuntimeCase(
        protocol="openai-responses",
        fixture=CONTRACTS / "openai-responses" / "codex-0.147.0",
        tool_name="exec_command",
        arguments={"cmd": "printf fixture-ok"},
        initial_lifecycle=(
            "response.created",
            "response.output_item.added:function_call",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done:function_call",
            "response.completed",
            "[DONE]",
        ),
        final_lifecycle=(
            "response.created",
            "response.output_item.added:message",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done:message",
            "response.completed",
            "[DONE]",
        ),
    ),
)


class StubGenerator:
    def __init__(self, case: RuntimeCase) -> None:
        arguments = json.dumps(case.arguments, separators=(",", ":"))
        self._outputs = (
            LocalGeneration(
                f'<tool_call>{{"name":"{case.tool_name}","arguments":{arguments}}}</tool_call>',
                64,
                12,
            ),
            LocalGeneration(FINAL_TEXT, 64, 8),
        )
        self.requests: list[LocalEngineRequest] = []

    def __call__(self, request: LocalEngineRequest) -> LocalGeneration:
        self.requests.append(request)
        return self._outputs[len(self.requests) - 1]


class QueuedGenerator:
    def __init__(self, outputs: tuple[str, ...]) -> None:
        self._outputs = outputs
        self.requests: list[LocalEngineRequest] = []

    def __call__(self, request: LocalEngineRequest) -> LocalGeneration:
        self.requests.append(request)
        return LocalGeneration(self._outputs[len(self.requests) - 1], 1, 1)


def _json(case: RuntimeCase, name: str) -> dict[str, object]:
    value = json.loads((case.fixture / name).read_text())
    assert isinstance(value, dict)
    return value


def _runtime(case: RuntimeCase) -> tuple[LocalAgentRuntime, StubGenerator]:
    generator = StubGenerator(case)
    runtime = LocalAgentRuntime(
        generate=generator,
        resolve_model=lambda _model, _protocol: "mlx-community/Qwen3",
    )
    return runtime, generator


def _frames(response: RuntimeResponse) -> tuple[SSEFrame, ...]:
    return parse_sse(response.sse, protocol=response.protocol)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _sequence(case: RuntimeCase, frames: tuple[SSEFrame, ...]) -> tuple[str, ...]:
    if case.protocol == "openai-chat":
        values: list[str] = []
        for frame in frames:
            assert frame.event is None
            if frame.data == "[DONE]":
                values.append("[DONE]")
                continue
            data = _mapping(frame.data)
            choice = _mapping(data["choices"][0])  # type: ignore[index]
            delta = _mapping(choice["delta"])
            finish_reason = choice["finish_reason"]
            if "role" in delta:
                values.append("assistant")
            elif "tool_calls" in delta:
                tool_delta = _mapping(delta["tool_calls"][0])  # type: ignore[index]
                values.append("tool_start" if "id" in tool_delta else "tool_delta")
            elif "content" in delta:
                values.append("text_delta")
            else:
                assert isinstance(finish_reason, str)
                values.append(finish_reason)
        return tuple(values)

    values = []
    for frame in frames:
        if frame.data == "[DONE]":
            values.append("[DONE]")
            continue
        data = _mapping(frame.data)
        event_type = data["type"]
        assert isinstance(event_type, str)
        assert frame.event == event_type
        suffix: object | None = None
        if event_type in {"content_block_start", "response.output_item.added", "response.output_item.done"}:
            key = "content_block" if event_type == "content_block_start" else "item"
            suffix = _mapping(data[key])["type"]
        elif event_type == "content_block_delta":
            suffix = _mapping(data["delta"])["type"]
        elif event_type == "message_delta":
            suffix = _mapping(data["delta"])["stop_reason"]
        values.append(f"{event_type}:{suffix}" if suffix is not None else event_type)
    return tuple(values)


def _first_call(case: RuntimeCase, response: RuntimeResponse) -> tuple[str, str]:
    frames = _frames(response)
    if case.protocol == "openai-chat":
        data = _mapping(frames[1].data)
        choice = _mapping(data["choices"][0])  # type: ignore[index]
        tool_call = _mapping(_mapping(choice["delta"])["tool_calls"][0])  # type: ignore[index]
        call_id = tool_call["id"]
        assert isinstance(call_id, str)
        return call_id, response.native_response_id
    if case.protocol == "anthropic-messages":
        block = _mapping(_mapping(frames[1].data)["content_block"])
        call_id = block["id"]
        assert isinstance(call_id, str)
        return call_id, response.native_response_id
    item = _mapping(_mapping(frames[1].data)["item"])
    call_id = item["call_id"]
    output_id = item["id"]
    assert isinstance(call_id, str)
    assert isinstance(output_id, str)
    return call_id, output_id


def _continuation(case: RuntimeCase, response: RuntimeResponse) -> dict[str, object]:
    native = deepcopy(_json(case, "tool-result-request.json"))
    call_id, output_id = _first_call(case, response)
    if case.protocol == "openai-chat":
        messages = native["messages"]
        assert isinstance(messages, list)
        assistant = _mapping(messages[-2])
        tool_call = _mapping(assistant["tool_calls"][0])  # type: ignore[index]
        tool_call["id"] = call_id
        _mapping(messages[-1])["tool_call_id"] = call_id
    elif case.protocol == "anthropic-messages":
        messages = native["messages"]
        assert isinstance(messages, list)
        assistant_block = _mapping(_mapping(messages[-2])["content"][0])  # type: ignore[index]
        result_block = _mapping(_mapping(messages[-1])["content"][0])  # type: ignore[index]
        assistant_block["id"] = call_id
        result_block["tool_use_id"] = call_id
    else:
        items = native["input"]
        assert isinstance(items, list)
        call = _mapping(items[-2])
        result = _mapping(items[-1])
        call["id"] = output_id
        call["call_id"] = call_id
        result["call_id"] = call_id
    return native


def _assert_call_identity(case: RuntimeCase, response: RuntimeResponse) -> None:
    frames = _frames(response)
    call_id, output_id = _first_call(case, response)
    expected_arguments = json.dumps(case.arguments, separators=(",", ":"))
    assert call_id
    if case.protocol == "openai-chat":
        ids = {
            _mapping(frame.data)["id"]
            for frame in frames[:-1]
        }
        assert ids == {response.native_response_id}
        delta = _mapping(_mapping(_mapping(frames[2].data)["choices"][0])["delta"])  # type: ignore[index]
        function = _mapping(_mapping(delta["tool_calls"][0])["function"])  # type: ignore[index]
        assert function["arguments"] == expected_arguments
    elif case.protocol == "anthropic-messages":
        message = _mapping(_mapping(frames[0].data)["message"])
        assert message["id"] == response.native_response_id
        delta = _mapping(_mapping(frames[2].data)["delta"])
        assert delta["partial_json"] == expected_arguments
    else:
        delta = _mapping(frames[2].data)
        done = _mapping(frames[3].data)
        item_done = _mapping(_mapping(frames[4].data)["item"])
        completed_output = _mapping(_mapping(frames[5].data)["response"])["output"]
        assert isinstance(completed_output, list)
        assert delta["item_id"] == done["item_id"] == output_id
        assert delta["delta"] == expected_arguments
        assert item_done["id"] == output_id
        assert item_done["call_id"] == call_id
        assert _mapping(completed_output[0])["call_id"] == call_id


def _read_request(messages: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model": "capture-model",
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read one file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "stream": True,
    }


def _chat_public_call_id(response: RuntimeResponse) -> str:
    frames = _frames(response)
    choice = _mapping(_mapping(frames[1].data)["choices"][0])  # type: ignore[index]
    tool_call = _mapping(_mapping(choice["delta"])["tool_calls"][0])  # type: ignore[index]
    call_id = tool_call["id"]
    assert isinstance(call_id, str)
    return call_id


def _read_call(call_id: str, path: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read", "arguments": json.dumps({"path": path})},
            }
        ],
    }


def _read_result(call_id: str, content: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_three_turn_kimi_reused_source_call_id_gets_distinct_public_ids() -> None:
    source_call_id = "functions.read:0"
    generator = QueuedGenerator(
        (
            "<|tool_calls_section_begin|><|tool_call_begin|>"
            f'{source_call_id}<|tool_call_argument_begin|>{{"path":"a"}}'
            "<|tool_call_end|><|tool_calls_section_end|>",
            "<|tool_calls_section_begin|><|tool_call_begin|>"
            f'{source_call_id}<|tool_call_argument_begin|>{{"path":"b"}}'
            "<|tool_call_end|><|tool_calls_section_end|>",
            "Complete.",
        )
    )
    runtime = LocalAgentRuntime(
        generate=generator,
        resolve_model=lambda _model, _protocol: "moonshot/Kimi-K2",
    )
    system: dict[str, object] = {"role": "system", "content": "Use read."}
    user: dict[str, object] = {"role": "user", "content": "Read two files."}

    first = runtime.execute(_read_request([system, user]), protocol="openai-chat", scope=SCOPE)
    first_call_id = _chat_public_call_id(first)
    first_call = _read_call(first_call_id, "a")
    first_result = _read_result(first_call_id, "A")
    first_continuation = _read_request([system, user, first_call, first_result])
    second = runtime.execute(
        first_continuation,
        protocol="openai-chat",
        scope=SCOPE,
    )
    second_call_id = _chat_public_call_id(second)
    replay = runtime.execute(
        deepcopy(first_continuation),
        protocol="openai-chat",
        scope=SCOPE,
    )
    assert replay == second
    assert len(generator.requests) == 2

    modified_continuation = deepcopy(first_continuation)
    modified_messages = modified_continuation["messages"]
    assert isinstance(modified_messages, list)
    _mapping(modified_messages[0])["content"] = "Changed frozen instruction."
    with pytest.raises(AgentRuntimeError, match="tool_continuation_expired") as caught:
        runtime.execute(
            modified_continuation,
            protocol="openai-chat",
            scope=SCOPE,
        )
    assert caught.value.status_code == 409
    assert len(generator.requests) == 2

    second_call = _read_call(second_call_id, "b")
    second_result = _read_result(second_call_id, "B")
    third = runtime.execute(
        _read_request(
            [system, user, first_call, first_result, second_call, second_result]
        ),
        protocol="openai-chat",
        scope=SCOPE,
    )

    assert first_call_id != second_call_id
    assert first.conversation_id == second.conversation_id == third.conversation_id
    assert _sequence(CASES[0], _frames(third)) == CASES[0].final_lifecycle
    assert len(generator.requests) == 3
    snapshots = [
        runtime.ledger.get(
            LedgerKey(
                principal_id=SCOPE.principal_id,
                project_id=SCOPE.project_id,
                harness=f"openai-chat:{SCOPE.harness_id}",
                conversation_id=first.conversation_id,
                call_id=call_id,
            )
        )
        for call_id in (first_call_id, second_call_id)
    ]
    assert [snapshot.source_call_id for snapshot in snapshots] == [source_call_id, source_call_id]
    assert all(snapshot.state is CallState.RESOLVED for snapshot in snapshots)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_modified_frozen_prompt_does_not_replay_cached_sse(case: RuntimeCase) -> None:
    runtime, generator = _runtime(case)
    first = runtime.execute(
        _json(case, "initial-request.json"),
        protocol=case.protocol,
        scope=SCOPE,
    )
    continuation = _continuation(case, first)
    runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)
    modified = deepcopy(continuation)
    if case.protocol == "openai-chat":
        messages = modified["messages"]
        assert isinstance(messages, list)
        first_message = _mapping(messages[0])
        first_message["content"] = f'{first_message["content"]} changed'
    elif case.protocol == "anthropic-messages":
        instructions = modified["system"]
        assert isinstance(instructions, list)
        first_instruction = _mapping(instructions[0])
        first_instruction["text"] = f'{first_instruction["text"]} changed'
    else:
        modified["instructions"] = f'{modified["instructions"]} changed'

    with pytest.raises(AgentRuntimeError, match="tool_continuation_expired") as caught:
        runtime.execute(modified, protocol=case.protocol, scope=SCOPE)

    assert caught.value.status_code == 409
    assert len(generator.requests) == 2


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_two_turn_streamed_tool_runtime_and_exact_retry(case: RuntimeCase) -> None:
    runtime, generator = _runtime(case)

    first = runtime.execute(_json(case, "initial-request.json"), protocol=case.protocol, scope=SCOPE)
    _assert_call_identity(case, first)
    assert _sequence(case, _frames(first)) == case.initial_lifecycle

    continuation = _continuation(case, first)
    second = runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)

    assert second.conversation_id == first.conversation_id
    assert _sequence(case, _frames(second)) == case.final_lifecycle
    assert len(generator.requests) == 2
    second_messages = generator.requests[1].messages
    assistant_call = _mapping(second_messages[-2])["tool_calls"]
    assert isinstance(assistant_call, list)
    stable_id, _ = _first_call(case, first)
    assert _mapping(assistant_call[0])["id"] == stable_id
    assert _mapping(second_messages[-1])["tool_call_id"] == stable_id
    assert "fixture-ok" in str(_mapping(second_messages[-1])["content"])

    retry = runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)

    assert retry == second
    assert len(generator.requests) == 2


def test_exact_retry_cache_does_not_outlive_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CASES[0]
    now = [100.0]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(continuation_module.time, "time", lambda: now[0])
    ledger = ContinuationLedger(active_ttl_seconds=1, clock=lambda: now[0])
    generator = StubGenerator(case)
    runtime = LocalAgentRuntime(
        ledger=ledger,
        generate=generator,
        resolve_model=lambda _model, _protocol: "mlx-community/Qwen3",
        conversation_ttl_seconds=1,
    )

    first = runtime.execute(
        _json(case, "initial-request.json"),
        protocol=case.protocol,
        scope=SCOPE,
    )
    continuation = _continuation(case, first)
    second = runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)
    assert runtime.execute(continuation, protocol=case.protocol, scope=SCOPE) == second

    now[0] += 1.01
    with pytest.raises(AgentRuntimeError, match="tool_continuation_expired") as caught:
        runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)

    assert caught.value.status_code == 409
    assert len(generator.requests) == 2


def test_exact_concurrent_retry_joins_one_continuation() -> None:
    case = CASES[0]
    second_started = threading.Event()
    join_started = threading.Event()
    release_second = threading.Event()

    class BlockingGenerator(StubGenerator):
        def __call__(self, request: LocalEngineRequest) -> LocalGeneration:
            if self.requests:
                second_started.set()
                assert release_second.wait(timeout=5)
            return super().__call__(request)

    generator = BlockingGenerator(case)
    ledger = ContinuationLedger()
    acquire_group = ledger.acquire_group_continuation

    def observed_acquire(*args: object, **kwargs: object) -> ContinuationTicket:
        ticket = acquire_group(*args, **kwargs)  # type: ignore[arg-type]
        if ticket.disposition == "join":
            join_started.set()
        return ticket

    ledger.acquire_group_continuation = observed_acquire  # type: ignore[method-assign]
    runtime = LocalAgentRuntime(
        ledger=ledger,
        generate=generator,
        resolve_model=lambda _model, _protocol: "mlx-community/Qwen3",
    )
    first = runtime.execute(
        _json(case, "initial-request.json"),
        protocol=case.protocol,
        scope=SCOPE,
    )
    continuation = _continuation(case, first)

    async def run() -> tuple[RuntimeResponse, RuntimeResponse]:
        owner = asyncio.create_task(
            runtime.execute_async(
                continuation,
                protocol=case.protocol,
                scope=SCOPE,
            )
        )
        assert await asyncio.to_thread(second_started.wait, 5)
        joined = asyncio.create_task(
            runtime.execute_async(
                deepcopy(continuation),
                protocol=case.protocol,
                scope=SCOPE,
            )
        )
        assert await asyncio.to_thread(join_started.wait, 5)
        release_second.set()
        return await owner, await joined

    owner_response, joined_response = asyncio.run(run())

    assert joined_response == owner_response
    assert len(generator.requests) == 2


def test_responses_echoes_disabled_parallel_tool_calls() -> None:
    case = CASES[2]
    runtime, _ = _runtime(case)

    response = runtime.execute(
        _json(case, "initial-request.json"),
        protocol=case.protocol,
        scope=SCOPE,
    )

    created = _mapping(_frames(response)[0].data)
    native_response = _mapping(created["response"])
    assert native_response["parallel_tool_calls"] is False


def test_chat_rejects_parallel_calls_when_disabled() -> None:
    case = CASES[0]
    output = (
        '<tool_call>{"name":"bash","arguments":{"command":"a"}}</tool_call>'
        '<tool_call>{"name":"bash","arguments":{"command":"b"}}</tool_call>'
    )
    generator = QueuedGenerator((output,))
    runtime = LocalAgentRuntime(
        generate=generator,
        resolve_model=lambda _model, _protocol: "mlx-community/Qwen3",
    )
    native = _json(case, "initial-request.json")
    native["parallel_tool_calls"] = False

    with pytest.raises(AgentRuntimeError, match="parallel_tool_calls_disabled"):
        runtime.execute(native, protocol=case.protocol, scope=SCOPE)

    assert len(generator.requests) == 1


def test_chat_applies_stream_usage_option() -> None:
    case = CASES[0]
    runtime, _ = _runtime(case)
    native = _json(case, "initial-request.json")
    native["stream_options"] = {"include_usage": False}

    response = runtime.execute(native, protocol=case.protocol, scope=SCOPE)

    terminal = _mapping(_frames(response)[-2].data)
    assert "usage" not in terminal


@pytest.mark.parametrize("case", (CASES[0], CASES[2]), ids=lambda case: case.protocol)
def test_local_runtime_rejects_store_true_before_generation(case: RuntimeCase) -> None:
    runtime, generator = _runtime(case)
    native = _json(case, "initial-request.json")
    native["store"] = True

    with pytest.raises(AgentRuntimeError, match="unsupported_local_storage"):
        runtime.execute(native, protocol=case.protocol, scope=SCOPE)

    assert generator.requests == []


def test_claude_adaptive_high_effort_enables_hidden_local_thinking() -> None:
    case = CASES[1]
    runtime, generator = _runtime(case)

    runtime.execute(
        _json(case, "initial-request.json"),
        protocol=case.protocol,
        scope=SCOPE,
    )

    assert generator.requests[0].enable_thinking is True


def test_grok_profile_completes_a_tool_continuation_with_text() -> None:
    case = CASES[0]
    tool_output = json.dumps(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "model_call_1",
                    "type": "function",
                    "function": {
                        "name": case.tool_name,
                        "arguments": json.dumps(case.arguments, separators=(",", ":")),
                    },
                }
            ],
        },
        separators=(",", ":"),
    )
    generator = QueuedGenerator((tool_output, FINAL_TEXT))
    runtime = LocalAgentRuntime(
        generate=generator,
        resolve_model=lambda _model, _protocol: "xai/grok-4",
    )

    first = runtime.execute(
        _json(case, "initial-request.json"),
        protocol=case.protocol,
        scope=SCOPE,
    )
    second = runtime.execute(
        _continuation(case, first),
        protocol=case.protocol,
        scope=SCOPE,
    )

    assert _sequence(case, _frames(second)) == case.final_lifecycle
    assert len(generator.requests) == 2


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
@pytest.mark.parametrize(
    "isolated_scope",
    (
        RuntimeScope(
            principal_id="principal_b",
            project_id="project_a",
            harness_id="harness_a",
        ),
        RuntimeScope(
            principal_id="principal_a",
            project_id="project_b",
            harness_id="harness_a",
        ),
        RuntimeScope(
            principal_id="principal_a",
            project_id="project_a",
            harness_id="harness_b",
        ),
    ),
    ids=("principal", "project", "harness"),
)
def test_tool_continuation_is_isolated_by_scope(
    case: RuntimeCase, isolated_scope: RuntimeScope
) -> None:
    runtime, generator = _runtime(case)
    first = runtime.execute(_json(case, "initial-request.json"), protocol=case.protocol, scope=SCOPE)

    with pytest.raises(AgentRuntimeError, match="tool_continuation_expired") as caught:
        runtime.execute(_continuation(case, first), protocol=case.protocol, scope=isolated_scope)

    assert caught.value.code == "tool_continuation_expired"
    assert len(generator.requests) == 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_malformed_tool_result_identity_fails_before_generation(case: RuntimeCase) -> None:
    runtime, generator = _runtime(case)
    first = runtime.execute(_json(case, "initial-request.json"), protocol=case.protocol, scope=SCOPE)
    continuation = _continuation(case, first)
    if case.protocol == "openai-chat":
        messages = continuation["messages"]
        assert isinstance(messages, list)
        _mapping(messages[-1])["tool_call_id"] = None
    elif case.protocol == "anthropic-messages":
        messages = continuation["messages"]
        assert isinstance(messages, list)
        _mapping(_mapping(messages[-1])["content"][0])["tool_use_id"] = None  # type: ignore[index]
    else:
        items = continuation["input"]
        assert isinstance(items, list)
        _mapping(items[-1])["call_id"] = None

    with pytest.raises(AgentRuntimeError, match="invalid_tool_result_identity") as caught:
        runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)

    assert caught.value.code == "invalid_tool_result_identity"
    assert len(generator.requests) == 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.protocol)
def test_unknown_tool_result_fails_before_generation(case: RuntimeCase) -> None:
    runtime, generator = _runtime(case)
    first = runtime.execute(_json(case, "initial-request.json"), protocol=case.protocol, scope=SCOPE)
    continuation = _continuation(case, first)
    unknown = "call_unknown"
    if case.protocol == "openai-chat":
        messages = continuation["messages"]
        assert isinstance(messages, list)
        _mapping(_mapping(messages[-2])["tool_calls"][0])["id"] = unknown  # type: ignore[index]
        _mapping(messages[-1])["tool_call_id"] = unknown
    elif case.protocol == "anthropic-messages":
        messages = continuation["messages"]
        assert isinstance(messages, list)
        _mapping(_mapping(messages[-2])["content"][0])["id"] = unknown  # type: ignore[index]
        _mapping(_mapping(messages[-1])["content"][0])["tool_use_id"] = unknown  # type: ignore[index]
    else:
        items = continuation["input"]
        assert isinstance(items, list)
        _mapping(items[-2])["call_id"] = unknown
        _mapping(items[-1])["call_id"] = unknown

    with pytest.raises(AgentRuntimeError, match="tool_continuation_expired") as caught:
        runtime.execute(continuation, protocol=case.protocol, scope=SCOPE)

    assert caught.value.code == "tool_continuation_expired"
    assert len(generator.requests) == 1
