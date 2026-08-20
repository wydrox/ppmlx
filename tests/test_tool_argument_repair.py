"""Tests for bounded local tool-argument repair."""
from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ppmlx.local_runtime.tool_argument_repair import (
    ToolArgumentRepairBudget,
    ToolArgumentRepairError,
    ToolArgumentRepairKind,
    ToolArgumentRepairPolicy,
    repair_json_object,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source", "expected", "kind"),
    [
        (
            json.dumps('{"path":"src/żółć.py","line":7}'),
            '{"path":"src/żółć.py","line":7}',
            ToolArgumentRepairKind.DOUBLE_ENCODED_OBJECT,
        ),
        (
            '{"path":"src/żółć.py","line":7,}',
            '{"path":"src/żółć.py","line":7}',
            ToolArgumentRepairKind.TRAILING_COMMA,
        ),
        (
            '{"path":"src/żółć.py","line":7',
            '{"path":"src/żółć.py","line":7}',
            ToolArgumentRepairKind.MISSING_FINAL_DELIMITER,
        ),
    ],
)
def test_bounded_json_repairs_one_allowlisted_defect(
    source: str,
    expected: str,
    kind: ToolArgumentRepairKind,
) -> None:
    result = repair_json_object(source, profile="qwen-json-v1")

    assert result.arguments_raw == expected
    assert json.loads(result.arguments_raw) == {"path": "src/żółć.py", "line": 7}
    assert result.metadata.policy is ToolArgumentRepairPolicy.BOUNDED_JSON_V1
    assert result.metadata.kind is kind
    assert result.metadata.profile == "qwen-json-v1"


def test_trailing_comma_repair_can_target_one_nested_array() -> None:
    result = repair_json_object(
        '{"items":[{"name":"alpha"},],"mode":"safe"}',
        profile="qwen-json-v1",
    )

    assert result.arguments_raw == '{"items":[{"name":"alpha"}],"mode":"safe"}'


def test_missing_final_delimiter_preserves_existing_whitespace() -> None:
    result = repair_json_object(
        '{"path":"src/main.py"\n',
        profile="qwen-json-v1",
    )

    assert result.arguments_raw == '{"path":"src/main.py"\n}'


def test_valid_object_does_not_consume_the_output_budget() -> None:
    budget = ToolArgumentRepairBudget()

    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object(
            '{"path":"src/main.py"}',
            profile="qwen-json-v1",
            budget=budget,
        )

    assert captured.value.code == "repair_not_required"
    assert budget.used is False

    result = repair_json_object(
        '{"path":"src/main.py",}',
        profile="qwen-json-v1",
        budget=budget,
    )
    assert result.metadata.kind is ToolArgumentRepairKind.TRAILING_COMMA
    assert budget.used is True


def test_one_budget_is_shared_by_the_complete_model_output() -> None:
    budget = ToolArgumentRepairBudget()

    repair_json_object(
        '{"first":1,}',
        profile="qwen-json-v1",
        budget=budget,
    )

    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object(
            '{"second":2',
            profile="qwen-json-v1",
            budget=budget,
        )

    assert captured.value.code == "repair_exhausted"


@pytest.mark.parametrize(
    "source",
    [
        '{"first":[1,],"second":{"value":2,}}',
        '{"outer":{"inner":[1,],},"mode":"safe"}',
    ],
)
def test_multiple_possible_edit_locations_are_ambiguous(source: str) -> None:
    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object(source, profile="qwen-json-v1")

    assert captured.value.code == "repair_ambiguous"


@pytest.mark.parametrize(
    "source",
    [
        json.dumps('{"value":1,}'),
        '{"outer":[1',
        '{"value":1,',
        '{"value":"unterminated}',
        "{'value':1}",
        '{value:1}',
        '{"value":NaN}',
        '{"value":1,"value":2}',
        '[1,2',
        '"plain text"',
    ],
)
def test_repair_does_not_chain_invent_or_relax_json(source: str) -> None:
    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object(source, profile="qwen-json-v1")

    assert captured.value.code == "repair_ineligible"


def test_candidate_must_fit_the_argument_limit_after_repair() -> None:
    source = '{"value":1'

    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object(
            source,
            profile="qwen-json-v1",
            max_bytes=len(source.encode("utf-8")),
        )

    assert captured.value.code == "arguments_limit_exceeded"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"profile": "INVALID PROFILE"}, "invalid_repair_profile"),
        ({"profile": "qwen-json-v1", "policy": "unknown"}, "repair_unavailable"),
        ({"profile": "qwen-json-v1", "max_bytes": -1}, "invalid_repair_limit"),
        ({"profile": "qwen-json-v1", "budget": object()}, "invalid_repair_budget"),
    ],
)
def test_invalid_repair_configuration_fails_closed(
    kwargs: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object('{"value":1,}', **kwargs)  # type: ignore[arg-type]

    assert captured.value.code == code


def test_errors_and_metadata_do_not_retain_argument_content() -> None:
    secret = "credential-test-THIS_IS_SECRET_123456"
    result = repair_json_object(
        f'{{"token":"{secret}",}}',
        profile="qwen-json-v1",
    )

    assert secret not in repr(result)
    assert secret not in repr(result.metadata)
    assert secret not in json.dumps(asdict(result.metadata), default=str)

    with pytest.raises(ToolArgumentRepairError) as captured:
        repair_json_object(
            f'{{"token":"{secret}","nested":[',
            profile="qwen-json-v1",
        )

    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_repair_module_has_only_pure_standard_library_imports() -> None:
    source = (
        ROOT / "ppmlx" / "local_runtime" / "tool_argument_repair.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )

    assert imports <= {"dataclasses", "enum", "json", "re", "typing"}
