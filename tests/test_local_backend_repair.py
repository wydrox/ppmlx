"""Backend tests for profile-gated bounded argument repair."""
from __future__ import annotations

import pytest

from ppmlx.agent_ir import (
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
)
import ppmlx.local_runtime.backend as backend_module
from ppmlx.local_runtime.backend import (
    LocalGeneration,
    LocalRuntimeError,
    TerminalReasons,
    execute_local_request,
)
from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    ToolNormalizationError,
)
from ppmlx.local_runtime.tool_argument_repair import ToolArgumentRepairPolicy
from ppmlx.local_runtime.tool_profiles import (
    ToolCapabilityLevel,
    ToolProfileContract,
)
from ppmlx.protocols import DecodeContext, openai_chat_adapter


def _request(*, model: str = "qwen-local"):
    return openai_chat_adapter.decode_request(
        {
            "model": model,
            "messages": [{"role": "user", "content": "List files."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": True,
        },
        context=DecodeContext(request_id="req_repair", kind="initial"),
    ).request


def _enable_repair(monkeypatch: pytest.MonkeyPatch, profile: NormalizationProfile) -> None:
    contract = ToolProfileContract(
        normalization_profile=profile,
        capability_level=ToolCapabilityLevel.TEMPLATE_STRUCTURED,
        repair_policy=ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
    )
    monkeypatch.setattr(
        backend_module,
        "get_tool_profile_contract",
        lambda selected: contract if selected is profile else None,
    )


def test_backend_emits_only_repaired_arguments_and_sanitized_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_repair(monkeypatch, NormalizationProfile.QWEN_JSON_V1)
    malformed = '{"cmd":"printf  a  b",}'
    repaired = '{"cmd":"printf  a  b"}'
    output = f'<tool_call>{{"name":"bash","arguments":{malformed}}}</tool_call>'

    result = execute_local_request(
        _request(),
        model="mlx-community/Qwen",
        generate=lambda request: LocalGeneration(output, 11, 5),
        profile=NormalizationProfile.QWEN_JSON_V1,
        terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        output_id="chatcmpl_repair",
        call_id_factory=lambda: "call_public",
    )

    delta = result.events[1]
    completed = result.events[2]
    assert isinstance(delta, ToolCallArgumentsDeltaEvent)
    assert isinstance(completed, ToolCallCompletedEvent)
    assert delta.delta == repaired
    assert malformed not in delta.delta
    assert completed.arguments_raw == repaired
    assert completed.arguments_json == {"cmd": "printf  a  b"}
    assert completed.extensions == {
        "ppmlx.tool_argument_repair": {
            "policy": "bounded-json-v1",
            "kind": "trailing_comma",
            "profile": "qwen-json-v1",
        }
    }
    assert malformed not in repr(completed.extensions)
    assert result.calls[0].call_id == "call_public"
    assert result.calls[0].output_id == "chatcmpl_repair"


def test_repaired_arguments_still_must_match_the_selected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_repair(monkeypatch, NormalizationProfile.QWEN_JSON_V1)
    output = '<tool_call>{"name":"bash","arguments":{"cmd":7,}}</tool_call>'

    with pytest.raises(LocalRuntimeError) as captured:
        execute_local_request(
            _request(),
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(output, 1, 1),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert captured.value.code == "tool_arguments_schema_mismatch"
    assert output not in str(captured.value)


def test_kimi_repair_keeps_source_and_public_call_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_repair(monkeypatch, NormalizationProfile.KIMI_K2_V1)
    output = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.bash:7"
        "<|tool_call_argument_begin|>{\"cmd\":\"ls\",}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = execute_local_request(
        _request(model="kimi-local"),
        model="mlx-community/Kimi-K2",
        generate=lambda request: LocalGeneration(output, 1, 1),
        profile=NormalizationProfile.KIMI_K2_V1,
        terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        call_id_factory=lambda: "call_public",
    )

    assert result.source_call_ids == {"call_public": "functions.bash:7"}
    completed = result.events[2]
    assert isinstance(completed, ToolCallCompletedEvent)
    assert str(completed.call_id) == "call_public"
    assert completed.arguments_raw == '{"cmd":"ls"}'
    assert completed.extensions["ppmlx.tool_argument_repair"]["profile"] == "kimi-k2-v1"


def test_backend_remains_strict_when_profile_contract_has_no_policy() -> None:
    output = '<tool_call>{"name":"bash","arguments":{"cmd":"ls",}}</tool_call>'

    with pytest.raises(ToolNormalizationError) as captured:
        execute_local_request(
            _request(),
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(output, 1, 1),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert captured.value.code == "malformed_arguments"


def test_parallel_backend_output_has_one_shared_repair_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_repair(monkeypatch, NormalizationProfile.QWEN_JSON_V1)
    output = (
        '<tool_call>{"name":"bash","arguments":{"cmd":"one",}}</tool_call>'
        '<tool_call>{"name":"bash","arguments":{"cmd":"two",}}</tool_call>'
    )

    with pytest.raises(ToolNormalizationError) as captured:
        execute_local_request(
            _request(),
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(output, 1, 1),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert captured.value.code == "repair_exhausted"
