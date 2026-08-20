"""Reproducible evaluation of exact local model tool profiles."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, cast

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
    """Published support state for one exact model profile."""

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
    """One fixed model prompt, tool set, and expected call sequence."""

    case_id: str
    messages: tuple[dict[str, object], ...]
    tools: tuple[dict[str, object], ...]
    expected_calls: tuple[ExpectedToolCall, ...]


@dataclass(frozen=True, slots=True)
class ToolEvaluationCaseSet:
    """Versioned fixed cases used for every model run."""

    schema_version: str
    case_set_version: str
    sha256: str
    cases: tuple[ToolEvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class AttemptEvaluation:
    """The safe result of one normalization attempt."""

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
class _RunMetrics:
    expected_calls: int
    strict_valid_calls: int
    repaired_valid_calls: int
    effective_valid_calls: int
    correlated_calls: int
    repair_attempts: dict[str, int]

    @property
    def strict_rate(self) -> float:
        return _rate(self.strict_valid_calls, self.expected_calls)

    @property
    def repaired_rate(self) -> float:
        return _rate(self.repaired_valid_calls, self.expected_calls)

    @property
    def effective_rate(self) -> float:
        return _rate(self.effective_valid_calls, self.expected_calls)

    @property
    def correlation_rate(self) -> float:
        return _rate(self.correlated_calls, self.expected_calls)


@dataclass(frozen=True, slots=True)
class RunEvaluation:
    """Aggregated safe metrics for one fixed run."""

    run_index: int
    seed: int
    cases: tuple[CaseEvaluation, ...]

    def _metrics(self) -> _RunMetrics:
        attempts: dict[str, int] = {}
        for case in self.cases:
            for kind in case.effective.repair_attempts:
                attempts[kind] = attempts.get(kind, 0) + 1
        return _RunMetrics(
            expected_calls=sum(
                case.effective.expected_calls for case in self.cases
            ),
            strict_valid_calls=sum(
                case.strict.valid_calls for case in self.cases
            ),
            repaired_valid_calls=sum(
                case.effective.repaired_valid_calls for case in self.cases
            ),
            effective_valid_calls=sum(
                case.effective.valid_calls for case in self.cases
            ),
            correlated_calls=sum(
                case.effective.correlated_calls for case in self.cases
            ),
            repair_attempts=attempts,
        )

    def to_dict(self) -> dict[str, object]:
        metrics = self._metrics()
        return {
            "run_index": self.run_index,
            "seed": self.seed,
            "case_count": len(self.cases),
            "expected_call_count": metrics.expected_calls,
            "strict_valid_call_count": metrics.strict_valid_calls,
            "strict_valid_call_rate": metrics.strict_rate,
            "repaired_valid_call_count": metrics.repaired_valid_calls,
            "repaired_valid_call_rate": metrics.repaired_rate,
            "effective_valid_call_count": metrics.effective_valid_calls,
            "effective_valid_call_rate": metrics.effective_rate,
            "correlated_call_count": metrics.correlated_calls,
            "correlation_rate": metrics.correlation_rate,
            "repair_attempts_by_kind": metrics.repair_attempts,
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


def _reject_constant(_: str) -> NoReturn:
    raise ProfileEvaluationError("invalid_case_number")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileEvaluationError("duplicate_case_key")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ProfileEvaluationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ProfileEvaluationError("invalid_case_set") from None
    return value, hashlib.sha256(raw).hexdigest()


def _plain_dict(value: Any, *, code: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProfileEvaluationError(code)
    if any(type(key) is not str for key in value):
        raise ProfileEvaluationError(code)
    return cast(dict[str, object], dict(value))


def _plain_dict_tuple(
    value: Any,
    *,
    code: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise ProfileEvaluationError(code)
    return tuple(_plain_dict(item, code=code) for item in value)


def _contains_external_ref(value: object) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "$ref" in item:
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _tool_definitions(
    tools: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    definitions: dict[str, dict[str, object]] = {}
    for tool in tools:
        if set(tool) != {"type", "function"}:
            raise ProfileEvaluationError("invalid_case_tool")
        if tool.get("type") != "function":
            raise ProfileEvaluationError("invalid_case_tool")
        function = _plain_dict(
            tool.get("function"),
            code="invalid_case_tool",
        )
        if set(function) != {"name", "description", "parameters"}:
            raise ProfileEvaluationError("invalid_case_tool")
        name_value = function.get("name")
        description = function.get("description")
        if type(name_value) is not str or not name_value:
            raise ProfileEvaluationError("invalid_case_tool")
        if type(description) is not str:
            raise ProfileEvaluationError("invalid_case_tool")
        name = cast(str, name_value)
        schema = _plain_dict(
            function.get("parameters"),
            code="invalid_case_schema",
        )
        if _contains_external_ref(schema):
            raise ProfileEvaluationError("external_case_schema_ref")
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

    loaded, digest = _load_json(path)
    root = _plain_dict(loaded, code="invalid_case_set")
    if set(root) != {"schema_version", "case_set_version", "cases"}:
        raise ProfileEvaluationError("invalid_case_set")
    schema_value = root.get("schema_version")
    version_value = root.get("case_set_version")
    raw_cases = root.get("cases")
    if schema_value != "tool-profile-cases/v1":
        raise ProfileEvaluationError("unsupported_case_schema")
    if type(version_value) is not str or not version_value:
        raise ProfileEvaluationError("invalid_case_set_version")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProfileEvaluationError("empty_case_set")

    cases: list[ToolEvaluationCase] = []
    identifiers: set[str] = set()
    for raw_case in raw_cases:
        case = _plain_dict(raw_case, code="invalid_case")
        if set(case) != {"id", "messages", "tools", "expected_calls"}:
            raise ProfileEvaluationError("invalid_case")
        case_id_value = case.get("id")
        if type(case_id_value) is not str or not case_id_value:
            raise ProfileEvaluationError("invalid_case_id")
        case_id = cast(str, case_id_value)
        if case_id in identifiers:
            raise ProfileEvaluationError("invalid_case_id")
        identifiers.add(case_id)

        messages = _plain_dict_tuple(
            case.get("messages"),
            code="invalid_case_messages",
        )
        tools = _plain_dict_tuple(
            case.get("tools"),
            code="invalid_case_tool",
        )
        definitions = _tool_definitions(tools)
        raw_expected = case.get("expected_calls")
        if not isinstance(raw_expected, list) or not raw_expected:
            raise ProfileEvaluationError("invalid_expected_calls")

        expected_calls: list[ExpectedToolCall] = []
        for raw_call in raw_expected:
            expected = _plain_dict(
                raw_call,
                code="invalid_expected_call",
            )
            if set(expected) != {"name", "arguments"}:
                raise ProfileEvaluationError("invalid_expected_call")
            name_value = expected.get("name")
            if type(name_value) is not str:
                raise ProfileEvaluationError("unknown_expected_tool")
            name = cast(str, name_value)
            if name not in definitions:
                raise ProfileEvaluationError("unknown_expected_tool")
            arguments = _plain_dict(
                expected.get("arguments"),
                code="invalid_expected_arguments",
            )
            try:
                Draft202012Validator(definitions[name]).validate(arguments)
            except ValidationError:
                raise ProfileEvaluationError(
                    "invalid_expected_arguments"
                ) from None
            expected_calls.append(
                ExpectedToolCall(name=name, arguments=arguments)
            )
        cases.append(
            ToolEvaluationCase(
                case_id=case_id,
                messages=messages,
                tools=tools,
                expected_calls=tuple(expected_calls),
            )
        )

    return ToolEvaluationCaseSet(
        schema_version=cast(str, schema_value),
        case_set_version=cast(str, version_value),
        sha256=digest,
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

    repair_attempts = tuple(
        call.repair.kind.value
        for call in normalized.tool_calls
        if call.repair is not None
    )
    if len(normalized.tool_calls) != expected_count:
        return AttemptEvaluation(
            expected_calls=expected_count,
            valid_calls=0,
            correlated_calls=0,
            repair_attempts=repair_attempts,
            repaired_valid_calls=0,
            error_code="call_sequence_mismatch",
        )
    if normalized.remaining_text.strip():
        return AttemptEvaluation(
            expected_calls=expected_count,
            valid_calls=0,
            correlated_calls=0,
            repair_attempts=repair_attempts,
            repaired_valid_calls=0,
            error_code="call_sequence_mismatch",
        )

    definitions = _tool_definitions(case.tools)
    valid_calls = 0
    repaired_valid_calls = 0
    identifiers: list[str] = []
    for index, (actual, expected) in enumerate(
        zip(normalized.tool_calls, case.expected_calls, strict=True)
    ):
        valid = actual.name == expected.name
        valid = valid and actual.arguments_json == expected.arguments
        schema = definitions.get(actual.name)
        if schema is None:
            valid = False
        else:
            try:
                Draft202012Validator(schema).validate(
                    actual.arguments_json
                )
            except ValidationError:
                valid = False
        if valid:
            valid_calls += 1
            if actual.repair is not None:
                repaired_valid_calls += 1
        identifiers.append(actual.call_id or f"generated:{index}")

    unique_identifiers = len(identifiers) == len(set(identifiers))
    correlated_calls = valid_calls if unique_identifiers else 0
    success = valid_calls == expected_count and unique_identifiers
    return AttemptEvaluation(
        expected_calls=expected_count,
        valid_calls=valid_calls,
        correlated_calls=correlated_calls,
        repair_attempts=repair_attempts,
        repaired_valid_calls=repaired_valid_calls,
        error_code=None if success else "call_mismatch",
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
    """Apply the accepted three-run publication thresholds."""

    if len(runs) != 3:
        raise ProfileEvaluationError("three_runs_required")
    metrics = [run._metrics() for run in runs]
    if not deterministic_fixtures_passed:
        return SupportStatus.DISABLED
    if any(metric.correlation_rate != 1.0 for metric in metrics):
        return SupportStatus.DISABLED
    minimum_rate = min(metric.effective_rate for metric in metrics)
    if minimum_rate >= 0.98:
        return SupportStatus.STABLE
    if minimum_rate >= 0.95:
        return SupportStatus.PREVIEW
    return SupportStatus.EXPERIMENTAL


def _validate_text(value: str, *, code: str) -> str:
    if type(value) is not str or not value:
        raise ProfileEvaluationError(code)
    return value


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
    architecture: str = "arm64",
) -> dict[str, object]:
    """Build one content-free, machine-readable profile report."""

    if len(runs) != 3:
        raise ProfileEvaluationError("three_runs_required")
    _validate_text(ppmlx_version, code="incomplete_report_metadata")
    _validate_text(ppmlx_commit, code="incomplete_report_metadata")
    _validate_text(model_repository, code="incomplete_report_metadata")
    _validate_text(model_revision, code="incomplete_report_metadata")
    _validate_text(tokenizer_revision, code="incomplete_report_metadata")
    _validate_text(quantization, code="incomplete_report_metadata")
    _validate_text(capability_level, code="incomplete_report_metadata")
    _validate_text(apple_chip, code="incomplete_report_metadata")
    _validate_text(macos_version, code="incomplete_report_metadata")
    _validate_text(architecture, code="incomplete_report_metadata")
    if type(memory_gb) is not int or memory_gb < 1:
        raise ProfileEvaluationError("invalid_environment_metadata")

    metrics = [run._metrics() for run in runs]
    expected = sum(metric.expected_calls for metric in metrics)
    strict_valid = sum(metric.strict_valid_calls for metric in metrics)
    repaired_valid = sum(
        metric.repaired_valid_calls for metric in metrics
    )
    effective_valid = sum(
        metric.effective_valid_calls for metric in metrics
    )
    correlated = sum(metric.correlated_calls for metric in metrics)
    attempts: dict[str, int] = {}
    for metric in metrics:
        for kind, count in metric.repair_attempts.items():
            attempts[kind] = attempts.get(kind, 0) + count

    status = classify_support_status(
        runs,
        deterministic_fixtures_passed=deterministic_fixtures_passed,
    )
    return {
        "schema_version": "tool-profile-report/v1",
        "ppmlx": {
            "version": ppmlx_version,
            "commit": ppmlx_commit,
        },
        "model": {
            "repository": model_repository,
            "revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "quantization": quantization,
        },
        "profile": {
            "normalization_profile": normalization_profile.value,
            "capability_level": capability_level,
            "repair_policy": (
                repair_policy.value if repair_policy is not None else None
            ),
        },
        "environment": {
            "apple_chip": apple_chip,
            "memory_gb": memory_gb,
            "macos_version": macos_version,
            "architecture": architecture,
        },
        "generation_settings": dict(generation_settings),
        "case_set": {
            "schema_version": case_set.schema_version,
            "version": case_set.case_set_version,
            "case_count": len(case_set.cases),
            "sha256": case_set.sha256,
        },
        "deterministic_fixtures_passed": (
            deterministic_fixtures_passed
        ),
        "runs": [run.to_dict() for run in runs],
        "aggregate": {
            "expected_call_count": expected,
            "strict_valid_call_count": strict_valid,
            "strict_valid_call_rate": _rate(strict_valid, expected),
            "repaired_valid_call_count": repaired_valid,
            "repaired_valid_call_rate": _rate(
                repaired_valid,
                expected,
            ),
            "effective_valid_call_count": effective_valid,
            "effective_valid_call_rate": _rate(
                effective_valid,
                expected,
            ),
            "correlated_call_count": correlated,
            "correlation_rate": _rate(correlated, expected),
            "repair_attempts_by_kind": attempts,
            "minimum_run_effective_valid_call_rate": min(
                metric.effective_rate for metric in metrics
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
