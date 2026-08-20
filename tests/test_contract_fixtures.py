from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "tests" / "fixtures" / "contracts"
MANIFEST_PATH = CONTRACT_ROOT / "manifest.json"
ADR_ROOT = ROOT / "docs" / "architecture" / "adr"
AGENT_IR_SCHEMA_PATH = ROOT / "docs" / "architecture" / "schema" / "agent-ir-v1.schema.json"

FIXTURE_FILES = {
    "README.md",
    "agent-ir.json",
    "initial-request.json",
    "tool-call-stream.sse",
    "tool-result-request.json",
    "final-response.sse",
}

EXPECTED_HARNESSES = {
    "claude-code": ("2.1.231", "anthropic-messages", "/v1/messages"),
    "codex": ("0.147.0", "openai-responses", "/v1/responses"),
    "opencode": ("1.18.18", "openai-chat", "/v1/chat/completions"),
    "pi": ("0.84.2", "openai-chat", "/v1/chat/completions"),
}

ADR_FILES = [
    "0001-product-boundary.md",
    "0002-agent-ir.md",
    "0003-tool-execution.md",
    "0004-provider-authentication.md",
    "0005-routing-and-fallback.md",
    "0006-memory-capture-and-read.md",
    "0007-retention-and-redaction.md",
    "0008-compatibility.md",
    "0009-bounded-tool-argument-repair.md",
]


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), f"{path} must contain a JSON object"
    return value


def parse_sse(path: Path) -> list[tuple[str | None, Any]]:
    text = path.read_text()
    assert text.endswith("\n\n"), f"{path} must end with a complete SSE frame"
    frames: list[tuple[str | None, Any]] = []
    for block in re.split(r"\r?\n\r?\n", text):
        if not block:
            continue
        event: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value
            elif field == "data":
                data_lines.append(value)
        assert data_lines, f"{path} contains an SSE frame without data"
        data = "\n".join(data_lines)
        payload = data if data == "[DONE]" else json.loads(data)
        frames.append((event, payload))
    assert frames, f"{path} must contain SSE frames"
    return frames


def walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_json(item)


def assert_order(values: list[str], expected: list[str]) -> None:
    position = -1
    for item in expected:
        position = values.index(item, position + 1)


def protocol_event_types(frames: list[tuple[str | None, Any]]) -> list[str]:
    values = []
    for event, payload in frames:
        if event:
            values.append(event)
        elif isinstance(payload, dict) and isinstance(payload.get("type"), str):
            values.append(payload["type"])
    return values


def tool_call_ids(protocol: str, frames: list[tuple[str | None, Any]]) -> list[str]:
    call_ids: list[str] = []
    for _, payload in frames:
        if not isinstance(payload, dict):
            continue
        if protocol == "anthropic-messages":
            block = payload.get("content_block", {})
            if block.get("type") == "tool_use" and block.get("id"):
                call_ids.append(block["id"])
        elif protocol == "openai-responses":
            item = payload.get("item", {})
            if item.get("type") == "function_call" and item.get("call_id"):
                call_ids.append(item["call_id"])
        else:
            for choice in payload.get("choices", []):
                for tool_call in choice.get("delta", {}).get("tool_calls", []):
                    if tool_call.get("id"):
                        call_ids.append(tool_call["id"])
    return call_ids


def tool_result_ids(protocol: str, request: dict[str, Any]) -> list[str]:
    if protocol == "anthropic-messages":
        return [
            item["tool_use_id"]
            for item in walk_json(request)
            if item.get("type") == "tool_result" and item.get("tool_use_id")
        ]
    if protocol == "openai-responses":
        return [
            item["call_id"]
            for item in walk_json(request)
            if item.get("type") == "function_call_output" and item.get("call_id")
        ]
    return [
        item["tool_call_id"]
        for item in walk_json(request)
        if item.get("role") == "tool" and item.get("tool_call_id")
    ]


def native_tool_call(protocol: str, frames: list[tuple[str | None, Any]]) -> tuple[str, str]:
    name = ""
    argument_fragments: list[str] = []
    for _, payload in frames:
        if not isinstance(payload, dict):
            continue
        if protocol == "anthropic-messages":
            block = payload.get("content_block", {})
            if block.get("type") == "tool_use":
                name = block.get("name", name)
            delta = payload.get("delta", {})
            if delta.get("type") == "input_json_delta":
                argument_fragments.append(delta.get("partial_json", ""))
        elif protocol == "openai-responses":
            item = payload.get("item", {})
            if item.get("type") == "function_call":
                name = item.get("name", name)
            if payload.get("type") == "response.function_call_arguments.delta":
                argument_fragments.append(payload.get("delta", ""))
        else:
            for choice in payload.get("choices", []):
                for tool_call in choice.get("delta", {}).get("tool_calls", []):
                    function = tool_call.get("function", {})
                    name = function.get("name") or name
                    argument_fragments.append(function.get("arguments", ""))
    assert name
    assert argument_fragments
    return name, "".join(argument_fragments)


def sse_extension(frames: list[tuple[str | None, Any]]) -> list[dict[str, Any]]:
    return [{"event": event, "data": payload} for event, payload in frames]


def native_tools(protocol: str, request: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in request["tools"]:
        if protocol == "anthropic-messages":
            native = tool
            input_schema = native["input_schema"]
        elif protocol == "openai-responses":
            native = tool
            input_schema = native["parameters"]
        else:
            native = tool["function"]
            input_schema = native["parameters"]
        item = {
            "name": native["name"],
            "description": native["description"],
            "input_schema": input_schema,
        }
        if "strict" in native:
            item["strict"] = native["strict"]
        normalized.append(item)
    return normalized


def ir_tools(request: dict[str, Any]) -> list[dict[str, Any]]:
    fields = {"name", "description", "input_schema", "strict"}
    return [{key: value for key, value in tool.items() if key in fields} for tool in request["tools"]]


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "".join(item.get("text", "") for item in value if isinstance(item, dict))


def native_instructions(protocol: str, request: dict[str, Any]) -> list[dict[str, Any]]:
    instructions: list[dict[str, Any]] = []
    if protocol == "anthropic-messages":
        system = request["system"]
        blocks = [{"type": "text", "text": system}] if isinstance(system, str) else system
        for index, block in enumerate(blocks):
            instruction = {
                "source_role": "system",
                "source_location": f"/system/{index}",
                "order": index,
                "text": block["text"],
            }
            if "cache_control" in block:
                instruction["cache_control"] = block["cache_control"]
            instructions.append(instruction)
    elif protocol == "openai-responses":
        instructions.append(
            {
                "source_role": "developer",
                "source_location": "/instructions",
                "order": 0,
                "text": request["instructions"],
            }
        )
    else:
        for index, message in enumerate(request["messages"]):
            if message["role"] in {"system", "developer"}:
                instructions.append(
                    {
                        "source_role": message["role"],
                        "source_location": f"/messages/{index}/content",
                        "order": len(instructions),
                        "text": content_text(message["content"]),
                    }
                )
    return instructions


def ir_instructions(request: dict[str, Any]) -> list[dict[str, Any]]:
    instructions: list[dict[str, Any]] = []
    for instruction in request["instructions"]:
        item = {
            "source_role": instruction["source_role"],
            "source_location": instruction["source_location"],
            "order": instruction["order"],
            "text": content_text(instruction["content"]),
        }
        native_block = instruction["content"][0].get("extensions", {}).get(
            "anthropic-messages.native_block", {}
        )
        if "cache_control" in native_block:
            item["cache_control"] = native_block["cache_control"]
        instructions.append(item)
    return instructions


def native_messages(protocol: str, request: dict[str, Any]) -> list[dict[str, Any]]:
    if protocol == "anthropic-messages":
        messages = request["messages"]
    elif protocol == "openai-responses":
        messages = request["input"]
    else:
        messages = request["messages"]

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if protocol == "openai-responses" and message.get("type") == "function_call":
            normalized.append(
                {
                    "id": message["id"],
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "call_id": message["call_id"],
                            "name": message["name"],
                            "arguments_raw": message["arguments"],
                        }
                    ],
                }
            )
            continue
        if protocol == "openai-responses" and message.get("type") == "function_call_output":
            normalized.append(
                {
                    "id": message["id"],
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "call_id": message["call_id"],
                            "text": message["output"],
                            "is_error": False,
                        }
                    ],
                }
            )
            continue

        item: dict[str, Any] = {"role": message["role"], "content": []}
        if message.get("id"):
            item["id"] = message["id"]
        content = message.get("content")
        if protocol == "openai-chat" and message["role"] == "tool":
            item["content"].append(
                {
                    "type": "tool_result",
                    "call_id": message["tool_call_id"],
                    "text": content_text(content),
                    "is_error": False,
                }
            )
        else:
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
            for block in blocks:
                block_type = block.get("type")
                if block_type in {"text", "input_text", "output_text"}:
                    normalized_block = {"type": "text", "text": block["text"]}
                    if protocol == "anthropic-messages" and "cache_control" in block:
                        normalized_block["cache_control"] = block["cache_control"]
                    item["content"].append(normalized_block)
                elif block_type == "tool_use":
                    normalized_block = {
                        "type": "tool_call",
                        "call_id": block["id"],
                        "name": block["name"],
                        "arguments_raw": json.dumps(block["input"], separators=(",", ":")),
                    }
                    if "cache_control" in block:
                        normalized_block["cache_control"] = block["cache_control"]
                    item["content"].append(normalized_block)
                elif block_type == "tool_result":
                    normalized_block = {
                        "type": "tool_result",
                        "call_id": block["tool_use_id"],
                        "text": content_text(block["content"]),
                        "is_error": block.get("is_error", False),
                    }
                    if "cache_control" in block:
                        normalized_block["cache_control"] = block["cache_control"]
                    item["content"].append(normalized_block)
        for tool_call in message.get("tool_calls", []):
            function = tool_call["function"]
            item["content"].append(
                {
                    "type": "tool_call",
                    "call_id": tool_call["id"],
                    "name": function["name"],
                    "arguments_raw": function["arguments"],
                }
            )
        normalized.append(item)
    return normalized


def ir_messages(protocol: str, request: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in request["messages"]:
        item: dict[str, Any] = {"role": message["role"], "content": []}
        if message.get("id"):
            item["id"] = message["id"]
        for block in message["content"]:
            if block["type"] == "text":
                normalized_block = {"type": "text", "text": block["text"]}
            elif block["type"] == "tool_call":
                normalized_block = {
                    "type": "tool_call",
                    "call_id": block["call_id"],
                    "name": block["name"],
                    "arguments_raw": block["arguments_raw"],
                }
            elif block["type"] == "tool_result":
                normalized_block = {
                    "type": "tool_result",
                    "call_id": block["call_id"],
                    "text": content_text(block["content"]),
                    "is_error": block["is_error"],
                }
            else:
                continue
            if protocol == "anthropic-messages":
                native_block = block.get("extensions", {}).get("anthropic-messages.native_block", {})
                if "cache_control" in native_block:
                    normalized_block["cache_control"] = native_block["cache_control"]
            item["content"].append(normalized_block)
        normalized.append(item)
    return normalized


def native_generation(protocol: str, request: dict[str, Any]) -> dict[str, Any]:
    generation: dict[str, Any] = {}
    max_tokens = request.get("max_output_tokens")
    if max_tokens is None:
        max_tokens = request.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = request.get("max_tokens")
    if max_tokens is not None:
        generation["max_output_tokens"] = max_tokens
    if protocol == "anthropic-messages":
        extension = {
            key: request[key]
            for key in ("thinking", "context_management", "output_config")
            if key in request
        }
        if extension:
            generation["extensions"] = {"anthropic-messages.generation": extension}
    elif protocol == "openai-responses" and "reasoning" in request:
        generation["extensions"] = {"openai-responses.generation": {"reasoning": request["reasoning"]}}
    return generation


def native_metadata(protocol: str, request: dict[str, Any]) -> dict[str, Any] | None:
    if protocol == "anthropic-messages":
        return request.get("metadata")
    if protocol == "openai-responses":
        metadata = {
            key: request[key]
            for key in ("client_metadata", "prompt_cache_key")
            if key in request
        }
        return metadata or None
    return request.get("metadata")


def native_request_options(protocol: str, request: dict[str, Any]) -> dict[str, Any] | None:
    if protocol == "openai-responses":
        keys = ("include", "parallel_tool_calls", "store")
    elif protocol == "openai-chat":
        keys = ("store", "stream_options")
    else:
        keys = ()
    options = {key: request[key] for key in keys if key in request}
    return options or None


def native_tool_result(protocol: str, request: dict[str, Any]) -> tuple[str, str, bool]:
    if protocol == "anthropic-messages":
        item = next(
            item
            for item in walk_json(request)
            if item.get("type") == "tool_result" and item.get("tool_use_id")
        )
        return item["tool_use_id"], item["content"], item.get("is_error", False)
    if protocol == "openai-responses":
        item = next(
            item
            for item in walk_json(request)
            if item.get("type") == "function_call_output" and item.get("call_id")
        )
        return item["call_id"], item["output"], False
    item = next(item for item in walk_json(request) if item.get("role") == "tool" and item.get("tool_call_id"))
    return item["tool_call_id"], item["content"], False


def native_final_text(protocol: str, frames: list[tuple[str | None, Any]]) -> str:
    fragments: list[str] = []
    for _, payload in frames:
        if not isinstance(payload, dict):
            continue
        if protocol == "anthropic-messages":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                fragments.append(delta.get("text", ""))
        elif protocol == "openai-responses":
            if payload.get("type") == "response.output_text.delta":
                fragments.append(payload.get("delta", ""))
        else:
            for choice in payload.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    fragments.append(content)
    assert fragments
    return "".join(fragments)


def native_finish_reason(protocol: str, frames: list[tuple[str | None, Any]]) -> str:
    values: list[str] = []
    for _, payload in frames:
        if not isinstance(payload, dict):
            continue
        if protocol == "anthropic-messages":
            value = payload.get("delta", {}).get("stop_reason")
            if value:
                values.append(value)
        elif protocol == "openai-responses":
            if payload.get("type") == "response.completed":
                values.append(payload["response"]["status"])
        else:
            values.extend(
                choice["finish_reason"]
                for choice in payload.get("choices", [])
                if choice.get("finish_reason")
            )
    assert values
    return values[-1]


def native_usage(protocol: str, frames: list[tuple[str | None, Any]]) -> dict[str, Any]:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    for _, payload in frames:
        if not isinstance(payload, dict):
            continue
        if protocol == "anthropic-messages":
            if payload.get("type") == "message_start":
                input_tokens = payload["message"]["usage"]["input_tokens"]
            elif payload.get("type") == "message_delta":
                output_tokens = payload["usage"]["output_tokens"]
        elif protocol == "openai-responses" and payload.get("type") == "response.completed":
            usage = payload["response"]["usage"]
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            total_tokens = usage["total_tokens"]
        elif protocol == "openai-chat" and payload.get("usage"):
            usage = payload["usage"]
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
    assert input_tokens is not None
    assert output_tokens is not None
    return {
        "source": "provider",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
    }


def native_output_id(protocol: str, frames: list[tuple[str | None, Any]]) -> str:
    for _, payload in frames:
        if not isinstance(payload, dict):
            continue
        if protocol == "anthropic-messages" and payload.get("type") == "message_start":
            return payload["message"]["id"]
        if protocol == "openai-responses":
            item = payload.get("item", {})
            if item.get("id"):
                return item["id"]
        if protocol == "openai-chat" and payload.get("id"):
            return payload["id"]
    raise AssertionError("the native stream must identify its output")


def assert_initial_request(protocol: str, request: dict[str, Any]) -> None:
    assert request.get("stream") is True
    assert request.get("tools"), "the initial request must declare at least one tool"
    if protocol == "anthropic-messages":
        assert request.get("system")
        messages = request.get("messages", [])
    elif protocol == "openai-responses":
        messages = request.get("input", [])
        assert request.get("instructions") or any(
            item.get("role") in {"developer", "system"} for item in messages if isinstance(item, dict)
        )
    else:
        messages = request.get("messages", [])
        assert any(item.get("role") in {"developer", "system"} for item in messages if isinstance(item, dict))
    assert any(item.get("role") == "user" for item in messages if isinstance(item, dict))


def assert_tool_stream(protocol: str, frames: list[tuple[str | None, Any]]) -> None:
    if protocol == "anthropic-messages":
        assert_order(
            protocol_event_types(frames),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
    elif protocol == "openai-responses":
        assert_order(
            protocol_event_types(frames),
            [
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
    else:
        payloads = [payload for _, payload in frames if isinstance(payload, dict)]
        finish_reasons = [choice.get("finish_reason") for payload in payloads for choice in payload.get("choices", [])]
        assert "tool_calls" in finish_reasons
        assert frames[-1][1] == "[DONE]"


def assert_final_stream(protocol: str, frames: list[tuple[str | None, Any]]) -> None:
    if protocol == "anthropic-messages":
        event_types = protocol_event_types(frames)
        assert_order(event_types, ["message_start", "content_block_start", "content_block_delta", "message_stop"])
        blocks = [item for _, payload in frames if isinstance(payload, dict) for item in walk_json(payload)]
        assert any(item.get("type") == "text" for item in blocks)
    elif protocol == "openai-responses":
        assert_order(protocol_event_types(frames), ["response.created", "response.output_text.delta", "response.completed"])
    else:
        payloads = [payload for _, payload in frames if isinstance(payload, dict)]
        assert any(choice.get("delta", {}).get("content") for payload in payloads for choice in payload.get("choices", []))
        assert any(choice.get("finish_reason") == "stop" for payload in payloads for choice in payload.get("choices", []))
        assert frames[-1][1] == "[DONE]"
    assert not tool_call_ids(protocol, frames), "the final answer must not start another tool call"


def test_contract_manifest_covers_exact_harness_versions_and_protocols():
    manifest = load_manifest()
    assert manifest["schema_version"] == 1
    assert re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", manifest["capture_date"])
    fixtures = manifest["fixtures"]
    assert len(fixtures) == 4
    assert {fixture["harness"] for fixture in fixtures} == set(EXPECTED_HARNESSES)
    assert {fixture["protocol"] for fixture in fixtures} == {
        "anthropic-messages",
        "openai-chat",
        "openai-responses",
    }
    assert len({fixture["id"] for fixture in fixtures}) == len(fixtures)

    for fixture in fixtures:
        version, protocol, endpoint = EXPECTED_HARNESSES[fixture["harness"]]
        assert (fixture["version"], fixture["protocol"], fixture["endpoint"]) == (version, protocol, endpoint)
        assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", fixture["version"])
        assert fixture["protocol_version"]
        directory = CONTRACT_ROOT / fixture["directory"]
        assert directory.resolve().is_relative_to(CONTRACT_ROOT.resolve())
        assert directory.name.endswith(fixture["version"])
        assert FIXTURE_FILES <= {path.name for path in directory.iterdir() if path.is_file()}


def test_agent_ir_schema_is_valid_and_accepts_all_fixtures():
    schema = load_json(AGENT_IR_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for fixture in load_manifest()["fixtures"]:
        agent_ir_path = CONTRACT_ROOT / fixture["directory"] / "agent-ir.json"
        errors = sorted(validator.iter_errors(load_json(agent_ir_path)), key=lambda item: list(item.path))
        assert not errors, f"{agent_ir_path} failed Agent IR validation: {errors}"


@pytest.mark.parametrize("fixture", load_manifest()["fixtures"], ids=lambda item: item["id"])
def test_fixture_has_a_complete_streamed_tool_round_trip(fixture: dict[str, Any]):
    directory = CONTRACT_ROOT / fixture["directory"]
    protocol = fixture["protocol"]
    initial_request = load_json(directory / "initial-request.json")
    tool_frames = parse_sse(directory / "tool-call-stream.sse")
    tool_result_request = load_json(directory / "tool-result-request.json")
    final_frames = parse_sse(directory / "final-response.sse")

    assert_initial_request(protocol, initial_request)
    assert_tool_stream(protocol, tool_frames)
    assert_final_stream(protocol, final_frames)

    agent_ir = load_json(directory / "agent-ir.json")
    assert agent_ir["ir_version"] == "agent-ir/v1"
    assert re.fullmatch(r"conv_[A-Za-z0-9_-]+", agent_ir["conversation_id"])
    assert agent_ir["source"] == {
        "harness": fixture["harness"],
        "harness_version": fixture["version"],
        "protocol": protocol,
        "protocol_version": fixture["protocol_version"],
    }

    requests = agent_ir["requests"]
    assert len(requests) == 2
    initial_envelope, continuation_envelope = requests
    assert initial_envelope["kind"] == "initial"
    assert "parent_request_id" not in initial_envelope
    assert continuation_envelope["kind"] == "continuation"
    assert continuation_envelope["parent_request_id"] == initial_envelope["request_id"]
    assert re.fullmatch(r"req_[A-Za-z0-9_-]+", initial_envelope["request_id"])
    assert re.fullmatch(r"req_[A-Za-z0-9_-]+", continuation_envelope["request_id"])

    native_request_key = f"{protocol}.native_request"
    initial_ir_request = initial_envelope["request"]
    continuation_ir_request = continuation_envelope["request"]
    assert initial_ir_request["extensions"][native_request_key] == initial_request
    assert continuation_ir_request["extensions"][native_request_key] == tool_result_request
    assert ir_tools(initial_ir_request) == native_tools(protocol, initial_request)
    assert ir_tools(continuation_ir_request) == native_tools(protocol, tool_result_request)
    for ir_request, native_request in (
        (initial_ir_request, initial_request),
        (continuation_ir_request, tool_result_request),
    ):
        assert ir_request["model"] == native_request["model"]
        assert ir_request["stream"] is native_request["stream"]
        assert ("tool_choice" in ir_request) == ("tool_choice" in native_request)
        if "tool_choice" in native_request:
            assert ir_request["tool_choice"] == native_request["tool_choice"]
        assert ir_request["generation"] == native_generation(protocol, native_request)
        assert ir_request.get("metadata") == native_metadata(protocol, native_request)
        assert ir_instructions(ir_request) == native_instructions(protocol, native_request)
        assert ir_messages(protocol, ir_request) == native_messages(protocol, native_request)
        options_key = f"{protocol}.request_options"
        assert ir_request["extensions"].get(options_key) == native_request_options(protocol, native_request)
        for message in ir_request["messages"]:
            for block in message["content"]:
                assert block["type"] in {
                    "text",
                    "image",
                    "document",
                    "reasoning",
                    "tool_call",
                    "tool_result",
                    "refusal",
                    "extension",
                }
    for field in ("model", "instructions", "tools", "tool_choice", "generation", "stream"):
        assert initial_ir_request.get(field) == continuation_ir_request.get(field)

    events = agent_ir["events"]
    sequences = [event["sequence"] for event in events]
    assert sequences == list(range(len(events)))
    request_ids = {initial_envelope["request_id"], continuation_envelope["request_id"]}
    assert {event["request_id"] for event in events} == request_ids
    initial_events = [event for event in events if event["request_id"] == initial_envelope["request_id"]]
    continuation_events = [
        event for event in events if event["request_id"] == continuation_envelope["request_id"]
    ]
    initial_types = [event["type"] for event in initial_events]
    continuation_types = [event["type"] for event in continuation_events]
    assert initial_types[0] == "tool_call.started"
    assert initial_types[-2:] == ["tool_call.completed", "response.completed"]
    assert initial_types[1:-2]
    assert set(initial_types[1:-2]) == {"tool_call.arguments.delta"}
    assert continuation_types[0:2] == ["tool_result", "content.started"]
    assert continuation_types[-2:] == ["content.completed", "response.completed"]
    assert continuation_types[2:-2]
    assert set(continuation_types[2:-2]) == {"content.delta"}

    call_ids = tool_call_ids(protocol, tool_frames)
    result_ids = tool_result_ids(protocol, tool_result_request)
    assert call_ids, "the streamed response must contain a tool-call ID"
    assert result_ids, "the follow-up request must contain a tool-result ID"
    assert len(set(call_ids)) == 1, "the tool-call ID must stay stable across stream events"
    assert set(result_ids) == set(call_ids), "the tool result must reference the streamed tool call"
    ir_tool_events = [event for event in events if "call_id" in event]
    assert {event["call_id"] for event in ir_tool_events} == set(call_ids)
    assert {event["choice_index"] for event in ir_tool_events} == {0}
    assert {event["tool_call_index"] for event in ir_tool_events} == {0}
    assert len({event["parallel_group_id"] for event in ir_tool_events}) == 1
    call_events = [event for event in initial_events if event["type"].startswith("tool_call.")]
    assert {event["output_id"] for event in call_events} == {native_output_id(protocol, tool_frames)}

    completed_call = next(event for event in events if event["type"] == "tool_call.completed")
    native_name, native_arguments = native_tool_call(protocol, tool_frames)
    assert completed_call["name"] == native_name
    assert completed_call["arguments_raw"] == native_arguments
    assert "".join(event["delta"] for event in events if event["type"] == "tool_call.arguments.delta") == native_arguments
    assert json.loads(completed_call["arguments_raw"]) == completed_call["arguments_json"]
    assert completed_call["sensitivity"] == "public"
    assert completed_call["provenance"] == {"origin": "provider", "trust": "untrusted"}

    result = next(event for event in events if event["type"] == "tool_result")
    result_call_id, native_result_text, native_is_error = native_tool_result(protocol, tool_result_request)
    assert result["call_id"] == result_call_id
    assert "".join(block["text"] for block in result["content"] if block["type"] == "text") == native_result_text
    assert result["is_error"] is native_is_error
    assert result["sensitivity"] == "public"
    assert result["provenance"] == {"origin": "tool", "trust": "untrusted"}

    final_text = native_final_text(protocol, final_frames)
    content_events = [event for event in continuation_events if event["type"].startswith("content.")]
    assert {event["output_id"] for event in content_events} == {native_output_id(protocol, final_frames)}
    assert content_events[0]["content_type"] == "text"
    assert "".join(event["delta"] for event in content_events if event["type"] == "content.delta") == final_text
    completed_content = content_events[-1]["content"]
    assert completed_content["type"] == "text"
    assert completed_content["text"] == final_text
    assert completed_content["sensitivity"] == "public"
    assert completed_content["provenance"] == {"origin": "provider", "trust": "untrusted"}
    assert not tool_call_ids(protocol, final_frames), "the final answer must not start another tool call"

    terminal_events = [event for event in events if event["type"] == "response.completed"]
    assert len(terminal_events) == 2
    stream_key = f"{protocol}.native_stream"
    assert terminal_events[0]["extensions"][stream_key] == sse_extension(tool_frames)
    assert terminal_events[1]["extensions"][stream_key] == sse_extension(final_frames)
    assert terminal_events[0]["output_id"] == native_output_id(protocol, tool_frames)
    assert terminal_events[1]["output_id"] == native_output_id(protocol, final_frames)
    assert terminal_events[0]["finish_reason"] == native_finish_reason(protocol, tool_frames)
    assert terminal_events[1]["finish_reason"] == native_finish_reason(protocol, final_frames)
    assert terminal_events[0]["usage"] == native_usage(protocol, tool_frames)
    assert terminal_events[1]["usage"] == native_usage(protocol, final_frames)


def test_contract_fixtures_are_sanitized_and_have_provenance():
    forbidden = [
        re.compile(r"/Users/", re.IGNORECASE),
        re.compile(r"/home/[A-Za-z0-9._-]+/"),
        re.compile(r"[A-Z]:\\\\Users\\\\", re.IGNORECASE),
        re.compile(r"\brafalw?\b", re.IGNORECASE),
        re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE),
        re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}", re.IGNORECASE),
        re.compile(r"(?:<redacted>|CHANGEME|\$\{[^}]+\}|\{\{[^}]+\}\})", re.IGNORECASE),
    ]
    for fixture in load_manifest()["fixtures"]:
        directory = CONTRACT_ROOT / fixture["directory"]
        readme = (directory / "README.md").read_text()
        assert fixture["version"] in readme
        assert "Capture" in readme
        assert "Sanitization" in readme
        for path in directory.iterdir():
            if path.is_file():
                text = path.read_text()
                for pattern in forbidden:
                    assert not pattern.search(text), f"{path} contains forbidden fixture data: {pattern.pattern}"


def test_accepted_architecture_decision_set_is_complete():
    assert sorted(path.name for path in ADR_ROOT.glob("*.md")) == ADR_FILES
    required_sections = [
        "Context",
        "Decision",
        "Consequences",
        "Rejected alternatives",
        "Security and privacy",
        "Compatibility effects",
    ]
    for name in ADR_FILES:
        text = (ADR_ROOT / name).read_text()
        assert re.search(r"Status[^\n]*Accepted", text, re.IGNORECASE)
        for section in required_sections:
            assert re.search(rf"^##+\s+{re.escape(section)}\s*$", text, re.IGNORECASE | re.MULTILINE), (
                f"{name} must contain a {section!r} section"
            )

    index = (ROOT / "docs" / "architecture" / "README.md").read_text()
    for name in ADR_FILES:
        assert f"adr/{name}" in index
