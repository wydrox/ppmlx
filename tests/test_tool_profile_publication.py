"""Tests for reviewed local tool-profile evidence and matrix rendering."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.local_runtime.profile_evaluation import (
    AttemptEvaluation,
    CaseEvaluation,
    RunEvaluation,
    build_report,
    load_case_set,
)
from ppmlx.local_runtime.profile_publication import (
    ProfilePublicationError,
    PublishedSupportStatus,
    classify_report,
    finalize_report,
    render_capability_matrix,
    validate_report,
)
from ppmlx.local_runtime.tool_argument_repair import ToolArgumentRepairPolicy


ROOT = Path(__file__).resolve().parents[1]
CASE_SET_PATH = ROOT / "tests" / "fixtures" / "tool_profile_eval" / "cases-v1.json"


def _run(
    index: int,
    *,
    expected: int = 100,
    strict_valid: int = 100,
    repaired_valid: int = 0,
    effective_valid: int = 100,
    correlated: int = 100,
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
        error_code=(
            None
            if effective_valid == expected and correlated == expected
            else "call_mismatch"
        ),
    )
    return RunEvaluation(
        run_index=index,
        seed=(17, 29, 43)[index - 1],
        cases=(
            CaseEvaluation(
                case_id=f"synthetic-{index}",
                strict=strict,
                effective=effective,
            ),
        ),
    )


def _report(
    *,
    runs: tuple[RunEvaluation, RunEvaluation, RunEvaluation] | None = None,
    fixtures_passed: bool = True,
) -> dict[str, object]:
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
        generation_settings={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seeds": [17, 29, 43],
        },
        case_set=case_set,
        runs=runs or (_run(1), _run(2), _run(3)),
        deterministic_fixtures_passed=fixtures_passed,
    )
    return finalize_report(
        report,
        architecture="arm64",
        case_set_sha256="d" * 64,
    )


def test_valid_report_requires_exact_immutable_content_free_evidence() -> None:
    report = _report()

    assert validate_report(report) is PublishedSupportStatus.STABLE
    serialized = json.dumps(report, sort_keys=True)
    assert '"architecture": "arm64"' in serialized
    assert '"sha256": "' + ("d" * 64) + '"' in serialized
    assert "messages" not in serialized
    assert "arguments_raw" not in serialized
    assert "model_output" not in serialized


def test_stable_status_is_blocked_when_repair_rate_exceeds_two_percent() -> None:
    report = _report(
        runs=(
            _run(1, strict_valid=97, repaired_valid=3),
            _run(2, strict_valid=98, repaired_valid=2),
            _run(3, strict_valid=100),
        )
    )

    assert classify_report(report) is PublishedSupportStatus.PREVIEW
    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["support_status"] == "preview"
    assert aggregate["maximum_run_repaired_valid_call_rate"] == 0.03


def test_fixture_or_correlation_failure_disables_publication() -> None:
    fixture_failure = _report(fixtures_passed=False)
    correlation_failure = _report(
        runs=(_run(1, correlated=99), _run(2), _run(3))
    )

    assert classify_report(fixture_failure) is PublishedSupportStatus.DISABLED
    assert classify_report(correlation_failure) is PublishedSupportStatus.DISABLED


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("model", "revision"), "main", "mutable_model_revision"),
        (("model", "tokenizer_revision"), "latest", "mutable_model_revision"),
        (("ppmlx", "commit"), "HEAD", "mutable_ppmlx_revision"),
        (("environment", "architecture"), "x86_64", "apple_silicon_required"),
    ],
)
def test_publication_rejects_mutable_or_non_apple_evidence(
    path: tuple[str, str],
    value: str,
    code: str,
) -> None:
    report = _report()
    section = report[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value

    with pytest.raises(ProfilePublicationError) as captured:
        validate_report(report)

    assert captured.value.code == code


def test_publication_rejects_content_fields_even_when_nested() -> None:
    report = _report()
    runs = report["runs"]
    assert isinstance(runs, list)
    first = runs[0]
    assert isinstance(first, dict)
    first["prompt"] = "credential-test-THIS_IS_SECRET"

    with pytest.raises(ProfilePublicationError) as captured:
        validate_report(report)

    assert captured.value.code in {"content_in_report", "invalid_run"}
    assert "credential-test" not in str(captured.value)


def test_matrix_is_generated_from_evidence_and_marks_other_profiles_not_evaluated() -> None:
    rendered = render_capability_matrix((_report(),))

    assert "mlx-community/Example-4bit" in rendered
    assert "`bbbbbbbbbbbb`" in rendered
    assert "**stable**" in rendered
    assert "`qwen-json-v1`" in rendered
    assert "`grok-openai-chat-v1`" in rendered
    assert "`kimi-k2-v1`" in rendered
    assert "`deepseek-v3-v1`" in rendered
    assert rendered.count("**not_evaluated**") == 3
    assert "family-name matching" in rendered.lower()


def test_report_rates_and_support_status_cannot_be_forged() -> None:
    report = _report()
    forged_rate = copy.deepcopy(report)
    aggregate = forged_rate["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["effective_valid_call_rate"] = 0.5

    with pytest.raises(ProfilePublicationError) as rate_error:
        validate_report(forged_rate)
    assert rate_error.value.code == "inconsistent_aggregate"

    forged_status = copy.deepcopy(report)
    status = forged_status["aggregate"]
    assert isinstance(status, dict)
    status["support_status"] = "experimental"

    with pytest.raises(ProfilePublicationError) as status_error:
        validate_report(forged_status)
    assert status_error.value.code == "support_status_mismatch"
