"""Tests for exact local model tool-profile evaluation evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.local_runtime.profile_evaluation import (
    AttemptEvaluation,
    CaseEvaluation,
    ProfileEvaluationError,
    RunEvaluation,
    SupportStatus,
    build_report,
    classify_support_status,
    evaluate_generated_output,
    load_case_set,
)
from ppmlx.local_runtime.tool_argument_repair import ToolArgumentRepairPolicy


ROOT = Path(__file__).resolve().parents[1]
CASE_SET_PATH = ROOT / "tests" / "fixtures" / "tool_profile_eval" / "cases-v1.json"


def _qwen_output(name: str, arguments: str) -> str:
    return f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>'


def _run(
    run_index: int,
    *,
    expected: int = 100,
    strict_valid: int = 100,
    effective_valid: int = 100,
    correlated: int = 100,
    repaired_valid: int = 0,
) -> RunEvaluation:
    strict = AttemptEvaluation(
        expected_calls=expected,
        valid_calls=strict_valid,
        correlated_calls=min(strict_valid, correlated),
        repair_attempts=(),
        repaired_valid_calls=0,
        error_code=None if strict_valid == expected else "call_mismatch",
    )
    effective = AttemptEvaluation(
        expected_calls=expected,
        valid_calls=effective_valid,
        correlated_calls=correlated,
        repair_attempts=("trailing_comma",) if repaired_valid else (),
        repaired_valid_calls=repaired_valid,
        error_code=None if effective_valid == expected and correlated == expected else "call_mismatch",
    )
    return RunEvaluation(
        run_index=run_index,
        seed=1000 + run_index,
        cases=(CaseEvaluation(case_id="synthetic", strict=strict, effective=effective),),
    )


def test_fixed_case_set_covers_required_argument_shapes() -> None:
    case_set = load_case_set(CASE_SET_PATH)

    assert case_set.schema_version == "tool-profile-cases/v1"
    assert case_set.case_set_version == "2026-08-20.v1"
    assert len(case_set.cases) == 9
    assert sum(len(case.expected_calls) for case in case_set.cases) == 10
    assert {case.case_id for case in case_set.cases} == {
        "weather-one-call",
        "unicode-file-path",
        "nested-search-options",
        "ordered-parallel-files",
        "array-labels",
        "nullable-assignee",
        "empty-object-status",
        "numeric-threshold",
        "escaped-command",
    }


def test_strict_valid_output_scores_without_repair() -> None:
    case = load_case_set(CASE_SET_PATH).cases[0]
    output = _qwen_output(
        "get_weather",
        '{"city":"Warsaw","unit":"celsius"}',
    )

    result = evaluate_generated_output(
        output,
        case=case,
        profile=NormalizationProfile.QWEN_JSON_V1,
        repair_policy=ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
    )

    assert result.strict.valid_calls == 1
    assert result.effective.valid_calls == 1
    assert result.effective.correlated_calls == 1
    assert result.effective.repaired_valid_calls == 0
    assert result.effective.repair_attempts == ()


def test_repaired_valid_output_reports_strict_and_effective_rates_separately() -> None:
    case = load_case_set(CASE_SET_PATH).cases[0]
    output = _qwen_output(
        "get_weather",
        '{"city":"Warsaw","unit":"celsius",}',
    )

    result = evaluate_generated_output(
        output,
        case=case,
        profile=NormalizationProfile.QWEN_JSON_V1,
        repair_policy=ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
    )

    assert result.strict.valid_calls == 0
    assert result.strict.error_code == "malformed_arguments"
    assert result.effective.valid_calls == 1
    assert result.effective.repaired_valid_calls == 1
    assert result.effective.repair_attempts == ("trailing_comma",)


def test_wrong_call_or_extra_text_does_not_score_as_valid() -> None:
    case = load_case_set(CASE_SET_PATH).cases[0]
    output = "preface" + _qwen_output(
        "get_weather",
        '{"city":"Krakow","unit":"celsius"}',
    )

    result = evaluate_generated_output(
        output,
        case=case,
        profile=NormalizationProfile.QWEN_JSON_V1,
        repair_policy=None,
    )

    assert result.effective.valid_calls == 0
    assert result.effective.correlated_calls == 0
    assert result.effective.error_code == "call_sequence_mismatch"


def test_generated_output_is_not_retained_in_evaluation_result() -> None:
    case = load_case_set(CASE_SET_PATH).cases[0]
    secret = "credential-test-THIS_IS_SECRET_123456"
    output = _qwen_output(
        "get_weather",
        '{"city":"Warsaw","unit":"celsius","secret":"'
        + secret
        + '"}',
    )

    result = evaluate_generated_output(
        output,
        case=case,
        profile=NormalizationProfile.QWEN_JSON_V1,
        repair_policy=None,
    )

    assert secret not in repr(result)
    assert secret not in json.dumps(result.effective.__dict__ if hasattr(result.effective, "__dict__") else {})


@pytest.mark.parametrize(
    ("runs", "fixtures_passed", "expected"),
    [
        ([_run(1, effective_valid=98), _run(2, effective_valid=99), _run(3, effective_valid=100)], True, SupportStatus.STABLE),
        ([_run(1, effective_valid=97), _run(2, effective_valid=96), _run(3, effective_valid=95)], True, SupportStatus.PREVIEW),
        ([_run(1, effective_valid=94), _run(2, effective_valid=100), _run(3, effective_valid=100)], True, SupportStatus.EXPERIMENTAL),
        ([_run(1, correlated=99), _run(2), _run(3)], True, SupportStatus.DISABLED),
        ([_run(1), _run(2), _run(3)], False, SupportStatus.DISABLED),
    ],
)
def test_support_status_uses_each_run_and_hard_correlation_gate(
    runs: list[RunEvaluation],
    fixtures_passed: bool,
    expected: SupportStatus,
) -> None:
    assert (
        classify_support_status(
            runs,
            deterministic_fixtures_passed=fixtures_passed,
        )
        is expected
    )


def test_publication_requires_exactly_three_runs() -> None:
    with pytest.raises(ProfileEvaluationError) as captured:
        classify_support_status(
            [_run(1), _run(2)],
            deterministic_fixtures_passed=True,
        )

    assert captured.value.code == "three_runs_required"


def test_report_contains_exact_evidence_and_no_case_content() -> None:
    case_set = load_case_set(CASE_SET_PATH)
    report = build_report(
        ppmlx_version="0.10.0",
        ppmlx_commit="a" * 40,
        model_repository="mlx-community/Example-4bit",
        model_revision="b" * 40,
        tokenizer_revision="c" * 40,
        quantization="4bit",
        normalization_profile=NormalizationProfile.QWEN_JSON_V1,
        capability_level="template_structured",
        repair_policy=ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
        apple_chip="Apple M4 Pro",
        memory_gb=48,
        macos_version="15.6",
        generation_settings={"temperature": 0, "seed_base": 1000},
        case_set=case_set,
        runs=(_run(1, repaired_valid=2), _run(2, repaired_valid=1), _run(3)),
    )

    serialized = json.dumps(report, sort_keys=True)
    assert report["schema_version"] == "tool-profile-report/v1"
    assert report["aggregate"]["support_status"] == "stable"  # type: ignore[index]
    assert report["model"]["revision"] == "b" * 40  # type: ignore[index]
    assert report["profile"]["repair_policy"] == "bounded-json-v1"  # type: ignore[index]
    assert "messages" not in serialized
    assert "expected_calls" not in serialized
    assert "Get the current temperature" not in serialized
    assert "generated_output" not in serialized


def test_case_set_rejects_duplicate_keys_and_unknown_expected_tools(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"tool-profile-cases/v1",'
        '"case_set_version":"v1","case_set_version":"v2","cases":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ProfileEvaluationError) as duplicate_error:
        load_case_set(duplicate)
    assert duplicate_error.value.code == "duplicate_case_key"

    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps(
            {
                "schema_version": "tool-profile-cases/v1",
                "case_set_version": "v1",
                "cases": [
                    {
                        "id": "unknown",
                        "messages": [{"role": "user", "content": "Do it."}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "known",
                                    "description": "Known tool.",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                        "expected_calls": [{"name": "missing", "arguments": {}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileEvaluationError) as unknown_error:
        load_case_set(unknown)
    assert unknown_error.value.code == "unknown_expected_tool"
