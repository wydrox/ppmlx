"""Reproducible evaluation of exact local model tool profiles."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    normalize_tool_output,
)
from .tool_argument_repair import ToolArgumentRepairPolicy


class ProfileEvaluationError(ValueError):
    """A safe evaluation error that does not contain model output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"tool profile evaluation error {code}")


class SupportStatus(str, Enum):
    """Internal status derived from three fixed evaluation runs."""

    STABLE = "stable"
    PREVIEW = "preview"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ExpectedToolCall:
    """One expected call in a fixed evaluation case."""

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolEvaluationCase:
    """One fixed prompt, tool set, and expected call sequence."""

    case_id: str
    messages: tuple[dict[str, object], ...]
    tools: tuple[dict[str, object], ...]
    expected_calls: tuple[ExpectedToolCall, ...]


@dataclass(frozen=True, slots=True)
class ToolEvaluationCaseSet:
    """Versioned fixed cases used for every model run."""

    schema_version: str
    case_set_version: str
    cases: tuple[ToolEvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class AttemptEvaluation:
    """The safe result of one strict or repair-enabled normalization attempt."""

    expected_calls: int
    valid_calls: int
    correlated_calls: int
    repair_attempts: tuple[str, ...]
    repaired_valid_calls: int
    error_code: str | None


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Strict and effective results for one generated output."""

    case_id: str
    strict: AttemptEvaluation
    effective: AttemptEvaluation


@dataclass(frozen=True, slots=True)
class RunEvaluation:
    """Aggregated safe metrics for one fixed run."""

    run_index: int
    seed: int
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        expected = sum(case.effective.expected_calls for case in self.cases)
        strict_valid = sum(case.strict.valid_calls for case in self.cases)
        effective_valid = sum(case.effective.valid_calls for case in self.cases)
        repaired_valid = sum(
            case.effective.repaired_valid_calls for case in self.cases
        )
        correlated = sum(case.effective.correlated_calls for case in self.cases)
        attempts: dict[str, int] = {}
        for case in self.cases:
            for kind in case.effective.repair_attempts:
                attempts[kind] = attempts.get(kind, 0) + 1
        return {
            "run_index": self.run_index,
            "seed": self.seed,
            "case_count": len(self.cases),
            "expected_call_count": expected,
            "strict_valid_call_count": strict_valid,
            "strict_valid_call_rate": _rate(strict_valid, expected),
            "repaired_valid_call_count": repaired_valid,
            "repaired_valid_call_rate": _rate(repaired_valid, expected),
            "effective_valid_call_count": effective_valid,
            "effective_valid_call_rate": _rate(effective_valid, expected),
            "correlated_call_count": correlated,
            "correlation_rate": _rate(correlated, expected),
            "repair_attempts_by_kind": attempts,
            "cases": [_case_result_dict(case) for case in self.cases],
        }


def _rate(numerator: int, denominator: int) -> float:
    if denominator < 1:
        raise ProfileEvaluationError("empty_evaluation")
    return round(numerator / denominator, 6)


def _case_result_dict(case: CaseEvaluation) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "expected_call_count": case.effective.expected_calls,
        "strict_valid_call_count": case.strict.valid_calls,
        "effective_valid_call_count": case.effective.valid_calls,
        "repaired_valid_call_count": case.effective.repaired_valid_calls,
        "correlated_call_count": case.effective.correlated_calls,
        "repair_attempts": list(case.effective.repair_attempts),
        "error_code": case.effective.error_code,
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileEvaluationError("duplicate_case_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ProfileEvaluationError("invalid_case_number")


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ProfileEvaluationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ProfileEvaluationError("invalid_case_set") from None


def _plain_dict(value: object, *, code: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProfileEvaluationError(code)
    return {str(key): item for key, item in value.items()}


def _plain_dict_tuple(
    value: object,
    *,
    code: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProfileEvaluationError(code)
    return tuple(_plain_dict(item, code=code) for item in value)


def _tool_definitions(
    tools: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    definitions: dict[str, dict[str, object]] = {}
    for tool in tools:
        if set(tool) != {"type", "function"} or tool.get("type") != "function":
            raise ProfileEvaluationError("invalid_case_tool")
        function = _plain_dict(tool.get("function"), code="invalid_case_tool")
        if set(function) != {"name", "description", "parameters"}:
            raise ProfileEvaluationError("invalid_case_tool")
        name = function.get("name")
        description = function.get("description")
        if type(name) is not str or not name or type(description) is not str:
            raise ProfileEvaluationError("invalid_case_tool")
        schema = _plain_dict(
            function.get("parameters"),
            code="invalid_case_schema",
        )
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            raise ProfileEvaluationError("invalid_case_schema") from None
        if name in definitions:
            raise ProfileEvaluationError("duplicate_case_tool")
        definitions[name] = schema
    return definitions


def load_case_set(path: Path) -> ToolEvaluationCaseSet:
    """Load and validate one fixed, non-secret evaluation case set."""

    root = _plain_dict(_load_json(path), code="invalid_case_set")
    if set(root) != {"schema_version", "case_set_version", "cases"}:
        raise ProfileEvaluationError("invalid_case_set")
    schema_version = root.get("schema_version")
    case_set_version = root.get("case_set_version")
    raw_cases = root.get("cases")
    if schema_version != "tool-profile-cases/v1":
        raise ProfileEvaluationError("unsupported_case_schema")
    if type(case_set_version) is not str or not case_set_version:
        raise ProfileEvaluationError("invalid_case_set_version")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProfileEvaluationError("empty_case_set")

    identifiers: set[str] = set()
    cases: list[ToolEvaluationCase] = []
    for raw_case in raw_cases:
        case = _plain_dict(raw_case, code="invalid_case")
        if set(case) != {"id", "messages", "tools", "expected_calls"}:
            raise ProfileEvaluationError("invalid_case")
        case_id = case.get("id")
        if type(case_id) is not str or not case_id or case_id in identifiers:
            raise ProfileEvaluationError("invalid_case_id")
        identifiers.add(case_id)
        messages = _plain_dict_tuple(
            case.get("messages"),
            code="invalid_case_messages",
        )
        tools = _plain_dict_tuple(case.get("tools"), code="invalid_case_tool")
        definitions = _tool_definitions(tools)
        raw_expected = case.get("expected_calls")
        if not isinstance(raw_expected, list) or not raw_expected:
            raise ProfileEvaluationError("invalid_expected_calls")
        expected_calls: list[ExpectedToolCall] = []
        for raw_call in raw_expected:
            expected = _plain_dict(raw_call, code="invalid_expected_call")
            if set(expected) != {"name", "arguments"}:
                raise ProfileEvaluationError("invalid_expected_call")
            name = expected.get("name")
            arguments = _plain_dict(
                expected.get("arguments"),
                code="invalid_expected_arguments",
            )
            if type(name) is not str or name not in definitions:
                raise ProfileEvaluationError("unknown_expected_tool")
            try:
                Draft202012Validator(definitions[name]).validate(arguments)
            except ValidationError:
                raise ProfileEvaluationError("invalid_expected_arguments") from None
            expected_calls.append(ExpectedToolCall(name=name, arguments=arguments))
        cases.append(
            ToolEvaluationCase(
                case_id=case_id,
                messages=messages,
                tools=tools,
                expected_calls=tuple(expected_calls),
            )
        )
    return ToolEvaluationCaseSet(
        schema_version=schema_version,
        case_set_version=case_set_version,
        cases=tuple(cases),
    )


def _attempt(
    output: str,
    *,
    case: ToolEvaluationCase,
    profile: NormalizationProfile,
    repair_policy: ToolArgumentRepairPolicy | None,
) -> AttemptEvaluation:
    expected_count = len(case.expected_calls)
    try:
        normalized = normalize_tool_output(
            output,
            profile=profile,
            repair_policy=repair_policy,
        )
    except ToolNormalizationError as error:
        return AttemptEvaluation(
            expected_calls=expected_count,
            valid_calls=0,
            correlated_calls=0,
            repair_attempts=(),
            repaired_valid_calls=0,
            error_code=error.code,
        )

    attempts = tuple(
        call.repair.kind.value
        for call in normalized.tool_calls
        if call.repair is not None
    )
    if len(normalized.tool_calls) != expected_count or normalized.remaining_text.strip():
        return AttemptEvaluation(
            expected_calls=expected_count,
            valid_calls=0,
            correlated_calls=0,
            repair_attempts=attempts,
            repaired_valid_calls=0,
            error_code="call_sequence_mismatch",
        )

    definitions = _tool_definitions(case.tools)
    valid_calls = 0
    repaired_valid = 0
    identifiers: list[str] = []
    for index, (actual, expected) in enumerate(
        zip(normalized.tool_calls, case.expected_calls, strict=True)
    ):
        valid = actual.name == expected.name and actual.arguments_json == expected.arguments
        schema = definitions.get(actual.name)
        if schema is None:
            valid = False
        else:
            try:
                Draft202012Validator(schema).validate(actual.arguments_json)
            except ValidationError:
                valid = False
        if valid:
            valid_calls += 1
            if actual.repair is not None:
                repaired_valid += 1
        identifiers.append(actual.call_id or f"generated:{index}")
    unique_identifiers = len(identifiers) == len(set(identifiers))
    correlated = valid_calls if unique_identifiers else 0
    return AttemptEvaluation(
        expected_calls=expected_count,
        valid_calls=valid_calls,
        correlated_calls=correlated,
        repair_attempts=attempts,
        repaired_valid_calls=repaired_valid,
        error_code=(
            None
            if valid_calls == expected_count and unique_identifiers
            else "call_mismatch"
        ),
    )


def evaluate_generated_output(
    output: str,
    *,
    case: ToolEvaluationCase,
    profile: NormalizationProfile,
    repair_policy: ToolArgumentRepairPolicy | None,
) -> CaseEvaluation:
    """Evaluate one generated output without retaining that output."""

    if type(output) is not str:
        raise ProfileEvaluationError("invalid_generated_output")
    return CaseEvaluation(
        case_id=case.case_id,
        strict=_attempt(
            output,
            case=case,
            profile=profile,
            repair_policy=None,
        ),
        effective=_attempt(
            output,
            case=case,
            profile=profile,
            repair_policy=repair_policy,
        ),
    )


def classify_support_status(
    runs: Sequence[RunEvaluation],
    *,
    deterministic_fixtures_passed: bool,
) -> SupportStatus:
    """Apply the accepted three-run thresholds and hard safety gates."""

    if len(runs) != 3:
        raise ProfileEvaluationError("three_runs_required")
    data = [run.to_dict() for run in runs]
    if not deterministic_fixtures_passed or any(
        run["correlation_rate"] != 1.0 for run in data
    ):
        return SupportStatus.DISABLED
    minimum = min(float(run["effective_valid_call_rate"]) for run in data)
    maximum_repair = max(float(run["repaired_valid_call_rate"]) for run in data)
    if minimum >= 0.98 and maximum_repair <= 0.02:
        return SupportStatus.STABLE
    if minimum >= 0.95:
        return SupportStatus.PREVIEW
    return SupportStatus.EXPERIMENTAL


def build_report(
    *,
    ppmlx_version: str,
    ppmlx_commit: str,
    model_repository: str,
    model_revision: str,
    tokenizer_revision: str,
    quantization: str,
    normalization_profile: NormalizationProfile,
    capability_level: str,
    repair_policy: ToolArgumentRepairPolicy | None,
    apple_chip: str,
    memory_gb: int,
    macos_version: str,
    generation_settings: Mapping[str, object],
    case_set: ToolEvaluationCaseSet,
    runs: Sequence[RunEvaluation],
    deterministic_fixtures_passed: bool = True,
) -> dict[str, object]:
    """Build one content-free machine-readable profile report."""

    if len(runs) != 3:
        raise ProfileEvaluationError("three_runs_required")
    metadata = (
        ppmlx_version,
        ppmlx_commit,
        model_repository,
        model_revision,
        tokenizer_revision,
        quantization,
        capability_level,
        apple_chip,
        macos_version,
    )
    if any(type(value) is not str or not value for value in metadata):
        raise ProfileEvaluationError("incomplete_report_metadata")
    if type(memory_gb) is not int or memory_gb < 1:
        raise ProfileEvaluationError("invalid_environment_metadata")

    run_data = [run.to_dict() for run in runs]
    expected = sum(int(run["expected_call_count"]) for run in run_data)
    strict_valid = sum(int(run["strict_valid_call_count"]) for run in run_data)
    repaired_valid = sum(int(run["repaired_valid_call_count"]) for run in run_data)
    effective_valid = sum(int(run["effective_valid_call_count"]) for run in run_data)
    correlated = sum(int(run["correlated_call_count"]) for run in run_data)
    attempts: dict[str, int] = {}
    for run in run_data:
        values = run["repair_attempts_by_kind"]
        if not isinstance(values, Mapping):
            raise ProfileEvaluationError("invalid_run")
        for kind, count in values.items():
            attempts[str(kind)] = attempts.get(str(kind), 0) + int(count)
    status = classify_support_status(
        runs,
        deterministic_fixtures_passed=deterministic_fixtures_passed,
    )
    return {
        "schema_version": "tool-profile-report/v1",
        "ppmlx": {"version": ppmlx_version, "commit": ppmlx_commit},
        "model": {
            "repository": model_repository,
            "revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "quantization": quantization,
        },
        "profile": {
            "normalization_profile": normalization_profile.value,
            "capability_level": capability_level,
            "repair_policy": repair_policy.value if repair_policy is not None else None,
        },
        "environment": {
            "apple_chip": apple_chip,
            "memory_gb": memory_gb,
            "macos_version": macos_version,
        },
        "generation_settings": dict(generation_settings),
        "case_set": {
            "schema_version": case_set.schema_version,
            "version": case_set.case_set_version,
            "case_count": len(case_set.cases),
        },
        "deterministic_fixtures_passed": deterministic_fixtures_passed,
        "runs": run_data,
        "aggregate": {
            "expected_call_count": expected,
            "strict_valid_call_count": strict_valid,
            "strict_valid_call_rate": _rate(strict_valid, expected),
            "repaired_valid_call_count": repaired_valid,
            "repaired_valid_call_rate": _rate(repaired_valid, expected),
            "effective_valid_call_count": effective_valid,
            "effective_valid_call_rate": _rate(effective_valid, expected),
            "correlated_call_count": correlated,
            "correlation_rate": _rate(correlated, expected),
            "repair_attempts_by_kind": attempts,
            "minimum_run_effective_valid_call_rate": min(
                float(run["effective_valid_call_rate"]) for run in run_data
            ),
            "support_status": status.value,
        },
    }


__all__ = [
    "AttemptEvaluation",
    "CaseEvaluation",
    "ExpectedToolCall",
    "ProfileEvaluationError",
    "RunEvaluation",
    "SupportStatus",
    "ToolEvaluationCase",
    "ToolEvaluationCaseSet",
    "build_report",
    "classify_support_status",
    "evaluate_generated_output",
    "load_case_set",
]
