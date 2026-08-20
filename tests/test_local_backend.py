from __future__ import annotations

import pytest

from ppmlx.agent_ir import (
    ContentCompletedEvent,
    ContentDeltaEvent,
    ContentStartedEvent,
    ResponseCompletedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from ppmlx.local_runtime.backend import (
    LocalGeneration,
    LocalRuntimeError,
    TerminalReasons,
    execute_local_request,
    prepare_local_request,
)
from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    select_normalization_profile,
)
from ppmlx.protocols import DecodeContext, openai_chat_adapter


def _request(*, model: str = "qwen-local", tool_choice: object = "auto"):
    native: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Keep the result short."},
            {"role": "user", "content": "List files."},
        ],
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
        "tool_choice": tool_choice,
        "temperature": 0.2,
        "max_tokens": 128,
        "stream": True,
    }
    return openai_chat_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_local", kind="initial"),
    ).request


def test_prepare_local_request_keeps_message_tool_and_generation_order() -> None:
    prepared = prepare_local_request(_request(), model="mlx-community/Qwen")

    assert prepared.model == "mlx-community/Qwen"
    assert [message["role"] for message in prepared.messages] == ["system", "user"]
    assert prepared.messages[0]["content"] == "Keep the result short."
    assert prepared.tools[0]["function"]["name"] == "bash"  # type: ignore[index]
    assert prepared.temperature == 0.2
    assert prepared.max_tokens == 128
    assert prepared.enable_thinking is False


def test_local_text_generation_emits_complete_agent_ir_lifecycle() -> None:
    envelope = _request(tool_choice="auto")

    result = execute_local_request(
        envelope,
        model="mlx-community/Qwen",
        generate=lambda request: LocalGeneration("Visible answer", 7, 3),
        profile=NormalizationProfile.QWEN_JSON_V1,
        terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        output_id="chatcmpl_local",
    )

    assert [type(event) for event in result.events] == [
        ContentStartedEvent,
        ContentDeltaEvent,
        ContentCompletedEvent,
        ResponseCompletedEvent,
    ]
    assert result.events[1].delta == "Visible answer"  # type: ignore[union-attr]
    assert result.events[-1].usage.total_tokens == 10  # type: ignore[union-attr]
    assert result.calls == ()


def test_qwen_tool_generation_preserves_raw_arguments_and_stable_ids() -> None:
    raw_arguments = '{"cmd":"printf  a  b"}'
    output = f'<tool_call>{{"name":"bash","arguments":{raw_arguments}}}</tool_call>'

    result = execute_local_request(
        _request(),
        model="mlx-community/Qwen",
        generate=lambda request: LocalGeneration(output, 11, 5),
        profile=NormalizationProfile.QWEN_JSON_V1,
        terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        output_id="chatcmpl_local",
        call_id_factory=lambda: "call_local",
    )

    assert [type(event) for event in result.events] == [
        ToolCallStartedEvent,
        ToolCallArgumentsDeltaEvent,
        ToolCallCompletedEvent,
        ResponseCompletedEvent,
    ]
    delta = result.events[1]
    completed = result.events[2]
    assert isinstance(delta, ToolCallArgumentsDeltaEvent)
    assert isinstance(completed, ToolCallCompletedEvent)
    assert delta.delta == raw_arguments
    assert completed.arguments_raw == raw_arguments
    assert completed.arguments_json == {"cmd": "printf  a  b"}
    assert result.calls[0].call_id == "call_local"
    assert result.calls[0].output_id == "chatcmpl_local"


def test_local_tool_arguments_must_match_the_selected_schema() -> None:
    output = '<tool_call>{"name":"bash","arguments":{"cmd":7}}</tool_call>'

    with pytest.raises(LocalRuntimeError) as caught:
        execute_local_request(
            _request(),
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(output, 1, 1),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert caught.value.code == "tool_arguments_schema_mismatch"
    assert output not in str(caught.value)


def test_local_tool_schema_does_not_resolve_an_external_reference() -> None:
    envelope = _request()
    tool = envelope.request.tools[0].model_copy(
        update={"input_schema": {"$ref": "https://example.invalid/tool-schema.json"}}
    )
    envelope = envelope.model_copy(
        update={"request": envelope.request.model_copy(update={"tools": [tool]})}
    )

    with pytest.raises(LocalRuntimeError) as caught:
        execute_local_request(
            envelope,
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(
                '<tool_call>{"name":"bash","arguments":{}}</tool_call>', 1, 1
            ),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert caught.value.code == "complex_tool_schema_unsupported"


def test_local_request_rejects_a_generation_limit_above_the_server_cap() -> None:
    envelope = _request()
    generation = envelope.request.generation.model_copy(update={"max_output_tokens": 32_769})
    envelope = envelope.model_copy(
        update={"request": envelope.request.model_copy(update={"generation": generation})}
    )

    with pytest.raises(LocalRuntimeError) as caught:
        prepare_local_request(envelope, model="mlx-community/Qwen", max_tokens_cap=32_768)

    assert caught.value.code == "max_tokens_exceeded"


def test_local_tool_schema_uses_a_bounded_pattern_validator() -> None:
    envelope = _request()
    tool = envelope.request.tools[0].model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "pattern": "^(a+)+$"}},
                "required": ["cmd"],
            }
        }
    )
    envelope = envelope.model_copy(
        update={"request": envelope.request.model_copy(update={"tools": [tool]})}
    )
    hostile = "a" * 20_000 + "!"

    with pytest.raises(LocalRuntimeError) as caught:
        execute_local_request(
            envelope,
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(
                '<tool_call>{"name":"bash","arguments":{"cmd":"'
                + hostile
                + '"}}</tool_call>',
                1,
                1,
            ),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert caught.value.code == "tool_arguments_schema_mismatch"


@pytest.mark.parametrize(
    ("tool_choice", "output", "code"),
    [
        (
            "none",
            '<tool_call>{"name":"bash","arguments":{"cmd":"ls"}}</tool_call>',
            "tool_call_forbidden",
        ),
        ("required", "plain text", "required_tool_missing"),
        (
            "auto",
            '<tool_call>{"name":"missing","arguments":{"cmd":"ls"}}</tool_call>',
            "unknown_tool",
        ),
        (
            "auto",
            'preface<tool_call>{"name":"bash","arguments":{"cmd":"ls"}}</tool_call>',
            "mixed_text_and_tool_calls",
        ),
    ],
)
def test_local_tool_contract_failures_are_typed_and_safe(
    tool_choice: object, output: str, code: str
) -> None:
    with pytest.raises(LocalRuntimeError) as caught:
        execute_local_request(
            _request(tool_choice=tool_choice),
            model="mlx-community/Qwen",
            generate=lambda request: LocalGeneration(output, 1, 1),
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert caught.value.code == code
    assert output not in str(caught.value)


def test_generation_exception_is_sanitized() -> None:
    def fail(_request):
        raise RuntimeError("secret model text")

    with pytest.raises(LocalRuntimeError) as caught:
        execute_local_request(
            _request(),
            model="mlx-community/Qwen",
            generate=fail,
            profile=NormalizationProfile.QWEN_JSON_V1,
            terminal_reasons=TerminalReasons(text="stop", tool_calls="tool_calls"),
        )

    assert caught.value.code == "generation_failed"
    assert "secret model text" not in str(caught.value)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("xai/grok-4", NormalizationProfile.GROK_OPENAI_CHAT_V1),
        ("moonshot/Kimi-K2", NormalizationProfile.KIMI_K2_V1),
        ("deepseek-ai/DeepSeek-V3", NormalizationProfile.DEEPSEEK_V3_V1),
        ("mlx-community/Qwen3", NormalizationProfile.QWEN_JSON_V1),
        ("meta-llama/Llama", None),
    ],
)
def test_profile_selection_is_explicit(model: str, expected: NormalizationProfile | None) -> None:
    assert select_normalization_profile(model) is expected
