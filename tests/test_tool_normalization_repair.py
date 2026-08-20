"""Integration tests for profile-gated local tool-argument repair."""
from __future__ import annotations

import json

import pytest

from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    normalize_tool_output,
)
from ppmlx.local_runtime.tool_argument_repair import (
    ToolArgumentRepairKind,
    ToolArgumentRepairPolicy,
)


POLICY = ToolArgumentRepairPolicy.BOUNDED_JSON_V1


def test_qwen_repairs_one_trailing_comma_inside_arguments() -> None:
    output = normalize_tool_output(
        '<tool_call>{"name":"read","arguments":{"path":"żółć.py",}}</tool_call>',
        profile=NormalizationProfile.QWEN_JSON_V1,
        repair_policy=POLICY,
    )

    call = output.tool_calls[0]
    assert call.arguments_raw == '{"path":"żółć.py"}'
    assert call.arguments_json == {"path": "żółć.py"}
    assert call.repair is not None
    assert call.repair.kind is ToolArgumentRepairKind.TRAILING_COMMA
    assert call.repair.profile == "qwen-json-v1"


def test_qwen_rejects_ambiguous_missing_inner_or_outer_delimiter() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            '<tool_call>{"name":"read","arguments":{"path":"main.py"}</tool_call>',
            profile="qwen-json-v1",
            repair_policy="bounded-json-v1",
        )

    assert captured.value.code == "repair_ambiguous"


def test_qwen_repairs_one_double_encoded_object() -> None:
    encoded = json.dumps('{"path":"main.py"}')
    output = normalize_tool_output(
        f'<tool_call>{{"name":"read","arguments":{encoded}}}</tool_call>',
        profile="qwen-json-v1",
        repair_policy=POLICY,
    )

    call = output.tool_calls[0]
    assert call.arguments_raw == '{"path":"main.py"}'
    assert call.repair is not None
    assert call.repair.kind is ToolArgumentRepairKind.DOUBLE_ENCODED_OBJECT


def test_kimi_repairs_arguments_without_changing_source_call_id() -> None:
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.read:7"
        "<|tool_call_argument_begin|>{\"path\":\"main.py\",}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    output = normalize_tool_output(
        text,
        profile=NormalizationProfile.KIMI_K2_V1,
        repair_policy=POLICY,
    )

    call = output.tool_calls[0]
    assert call.call_id == "functions.read:7"
    assert call.arguments_raw == '{"path":"main.py"}'
    assert call.repair is not None
    assert call.repair.kind is ToolArgumentRepairKind.TRAILING_COMMA


def test_deepseek_repairs_one_missing_final_delimiter() -> None:
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n"
        '{"query":"safe"'
        "\n```<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )
    output = normalize_tool_output(
        text,
        profile=NormalizationProfile.DEEPSEEK_V3_V1,
        repair_policy=POLICY,
    )

    call = output.tool_calls[0]
    assert call.arguments_raw == '{"query":"safe"}'
    assert call.repair is not None
    assert call.repair.kind is ToolArgumentRepairKind.MISSING_FINAL_DELIMITER


def test_grok_repairs_arguments_without_changing_source_call_id() -> None:
    text = json.dumps(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_safe_01",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps('{"path":"main.py"}'),
                    },
                }
            ],
        }
    )
    output = normalize_tool_output(
        text,
        profile=NormalizationProfile.GROK_OPENAI_CHAT_V1,
        repair_policy=POLICY,
    )

    call = output.tool_calls[0]
    assert call.call_id == "call_safe_01"
    assert call.arguments_raw == '{"path":"main.py"}'
    assert call.repair is not None
    assert call.repair.kind is ToolArgumentRepairKind.DOUBLE_ENCODED_OBJECT


def test_one_repair_budget_is_shared_by_parallel_calls() -> None:
    text = (
        '<tool_call>{"name":"first","arguments":{"n":1,}}</tool_call>'
        '<tool_call>{"name":"second","arguments":{"n":2}}</tool_call>'
    )
    output = normalize_tool_output(
        text,
        profile="qwen-json-v1",
        repair_policy=POLICY,
    )

    assert output.tool_calls[0].repair is not None
    assert output.tool_calls[1].repair is None
    assert [call.arguments_json for call in output.tool_calls] == [{"n": 1}, {"n": 2}]


def test_second_parallel_repair_is_rejected() -> None:
    text = (
        '<tool_call>{"name":"first","arguments":{"n":1,}}</tool_call>'
        '<tool_call>{"name":"second","arguments":{"n":2,}}</tool_call>'
    )

    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            text,
            profile="qwen-json-v1",
            repair_policy=POLICY,
        )

    assert captured.value.code == "repair_exhausted"


def test_policy_is_disabled_by_default() -> None:
    text = '<tool_call>{"name":"read","arguments":{"path":"main.py",}}</tool_call>'

    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(text, profile="qwen-json-v1")

    assert captured.value.code == "malformed_arguments"


def test_valid_arguments_remain_byte_exact_with_policy_enabled() -> None:
    raw = '{ "path" : "main.py", "line" :  7 }'
    output = normalize_tool_output(
        f'<tool_call>{{"name":"read","arguments":{raw}}}</tool_call>',
        profile="qwen-json-v1",
        repair_policy=POLICY,
    )

    call = output.tool_calls[0]
    assert call.arguments_raw == raw
    assert call.repair is None


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (
            '<tool_call>{"arguments":{"path":"main.py",},"name":"read"}</tool_call>',
            "invalid_tool_call_shape",
        ),
        (
            '<tool_call>{"name":"read","arguments":{"path":"main.py"},}</tool_call>',
            "repair_ineligible",
        ),
        (
            '<tool_call>{"name":"read","arguments":{"path":"a","path":"b"}}</tool_call>',
            "duplicate_json_key",
        ),
        (
            '<tool_call>{"name":"read","arguments":{"first":[1,],"second":[2,]}}</tool_call>',
            "repair_ambiguous",
        ),
    ],
)
def test_qwen_repair_does_not_relax_the_outer_envelope_or_json_rules(
    text: str,
    code: str,
) -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            text,
            profile="qwen-json-v1",
            repair_policy=POLICY,
        )

    assert captured.value.code == code


def test_repair_error_does_not_retain_private_arguments() -> None:
    secret = "credential-test-THIS_IS_SECRET_123456"
    text = (
        '<tool_call>{"name":"read","arguments":{"secret":"'
        + secret
        + '","first":[1,],"second":[2,]}}</tool_call>'
    )

    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            text,
            profile="qwen-json-v1",
            repair_policy=POLICY,
        )

    assert secret not in str(captured.value)
    assert secret not in captured.value.profile
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
