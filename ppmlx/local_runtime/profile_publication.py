"""Validation and rendering for published local tool-profile evidence."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from .normalization import NormalizationProfile
from .tool_argument_repair import ToolArgumentRepairPolicy
from .tool_profiles import ToolCapabilityLevel, list_tool_profile_contracts


_REPORT_SCHEMA = "tool-profile-report/v1"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FORBIDDEN_CONTENT_KEYS = frozenset(
    {
        "arguments",
        "arguments_raw",
        "completion",
        "content",
        "messages",
        "model_output",
        "output",
        "prompt",
        "raw_arguments",
        "reasoning",
        "response",
        "tool_result",
    }
)
_STABLE_MIN_RATE = 0.98
_PREVIEW_MIN_RATE = 0.95
_MAX_STABLE_REPAIR_RATE = 0.02


class ProfilePublicationError(ValueError):
    """A safe publication error that contains no report content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"tool profile publication error {code}")


class PublishedSupportStatus(str, Enum):
    """Support status derived from reviewed evidence."""

    NOT_EVALUATED = "not_evaluated"
    STABLE = "stable"
    PREVIEW = "preview"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


def _mapping(value: object, *, code: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProfilePublicationError(code)
    return dict(value)


def _sequence(value: object, *, code: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProfilePublicationError(code)
    return list(value)


def _number(value: object, *, code: str) -> float:
    if type(value) not in {int, float}:
        raise ProfilePublicationError(code)
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ProfilePublicationError(code)
    return number


def _count(value: object, *, code: str) -> int:
    if type(value) is not int or value < 0:
        raise ProfilePublicationError(code)
    return value


def _rate(count: int, total: int) -> float:
    if total < 1:
        raise ProfilePublicationError("empty_evaluation")
    return round(count / total, 6)


def _assert_no_content(value: object) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) is not str:
                    raise ProfilePublicationError("invalid_report")
                if key.lower() in _FORBIDDEN_CONTENT_KEYS:
                    raise ProfilePublicationError("content_in_report")
                stack.append(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            stack.extend(item)
        elif item is None or type(item) in {str, int, float, bool}:
            continue
        else:
            raise ProfilePublicationError("invalid_report_value")


def _validate_identity(report: Mapping[str, object]) -> None:
    ppmlx = _mapping(report.get("ppmlx"), code="invalid_ppmlx_identity")
    if set(ppmlx) != {"version", "commit"}:
        raise ProfilePublicationError("invalid_ppmlx_identity")
    if type(ppmlx["version"]) is not str or not ppmlx["version"]:
        raise ProfilePublicationError("invalid_ppmlx_identity")
    if type(ppmlx["commit"]) is not str or _SHA_RE.fullmatch(ppmlx["commit"]) is None:
        raise ProfilePublicationError("mutable_ppmlx_revision")

    model = _mapping(report.get("model"), code="invalid_model_identity")
    if set(model) != {"repository", "revision", "tokenizer_revision", "quantization"}:
        raise ProfilePublicationError("invalid_model_identity")
    repository = model["repository"]
    if type(repository) is not str or _MODEL_REPO_RE.fullmatch(repository) is None:
        raise ProfilePublicationError("invalid_model_repository")
    for key in ("revision", "tokenizer_revision"):
        value = model[key]
        if type(value) is not str or _SHA_RE.fullmatch(value) is None:
            raise ProfilePublicationError("mutable_model_revision")
    if type(model["quantization"]) is not str or not model["quantization"]:
        raise ProfilePublicationError("invalid_quantization")


def _validate_profile(report: Mapping[str, object]) -> None:
    profile = _mapping(report.get("profile"), code="invalid_profile")
    if set(profile) != {"normalization_profile", "capability_level", "repair_policy"}:
        raise ProfilePublicationError("invalid_profile")
    try:
        normalization_profile = NormalizationProfile(profile["normalization_profile"])
        capability_level = ToolCapabilityLevel(profile["capability_level"])
    except (TypeError, ValueError):
        raise ProfilePublicationError("invalid_profile") from None
    repair_value = profile["repair_policy"]
    if repair_value is None:
        repair_policy = None
    else:
        try:
            repair_policy = ToolArgumentRepairPolicy(repair_value)
        except (TypeError, ValueError):
            raise ProfilePublicationError("invalid_repair_policy") from None
    contract = next(
        (
            candidate
            for candidate in list_tool_profile_contracts()
            if candidate.normalization_profile is normalization_profile
        ),
        None,
    )
    if contract is None or contract.capability_level is not capability_level:
        raise ProfilePublicationError("profile_contract_mismatch")
    if repair_policy is not None and capability_level not in {
        ToolCapabilityLevel.TEMPLATE_STRUCTURED,
        ToolCapabilityLevel.PROMPT_EMULATED,
    }:
        raise ProfilePublicationError("repair_policy_ineligible")


def _validate_environment(report: Mapping[str, object]) -> None:
    environment = _mapping(report.get("environment"), code="invalid_environment")
    if set(environment) != {"apple_chip", "memory_gb", "macos_version", "architecture"}:
        raise ProfilePublicationError("invalid_environment")
    if environment["architecture"] != "arm64":
        raise ProfilePublicationError("apple_silicon_required")
    if type(environment["apple_chip"]) is not str or not environment["apple_chip"]:
        raise ProfilePublicationError("invalid_environment")
    if type(environment["macos_version"]) is not str or not environment["macos_version"]:
        raise ProfilePublicationError("invalid_environment")
    if type(environment["memory_gb"]) is not int or environment["memory_gb"] < 1:
        raise ProfilePublicationError("invalid_environment")


def _validate_case_set(report: Mapping[str, object]) -> None:
    case_set = _mapping(report.get("case_set"), code="invalid_case_set_evidence")
    if set(case_set) != {"schema_version", "version", "case_count", "sha256"}:
        raise ProfilePublicationError("invalid_case_set_evidence")
    if case_set["schema_version"] != "tool-profile-cases/v1":
        raise ProfilePublicationError("invalid_case_set_evidence")
    if type(case_set["version"]) is not str or not case_set["version"]:
        raise ProfilePublicationError("invalid_case_set_evidence")
    if type(case_set["case_count"]) is not int or case_set["case_count"] < 1:
        raise ProfilePublicationError("invalid_case_set_evidence")
    if type(case_set["sha256"]) is not str or _SHA256_RE.fullmatch(case_set["sha256"]) is None:
        raise ProfilePublicationError("invalid_case_set_evidence")


def _validate_generation_settings(report: Mapping[str, object]) -> None:
    settings = _mapping(report.get("generation_settings"), code="invalid_generation_settings")
    required = {"temperature", "top_p", "max_tokens", "seeds"}
    if set(settings) != required:
        raise ProfilePublicationError("invalid_generation_settings")
    if type(settings["temperature"]) not in {int, float}:
        raise ProfilePublicationError("invalid_generation_settings")
    if type(settings["top_p"]) not in {int, float}:
        raise ProfilePublicationError("invalid_generation_settings")
    if type(settings["max_tokens"]) is not int or settings["max_tokens"] < 1:
        raise ProfilePublicationError("invalid_generation_settings")
    seeds = _sequence(settings["seeds"], code="invalid_generation_settings")
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        raise ProfilePublicationError("invalid_generation_settings")


def _validated_runs(report: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    runs = [
        _mapping(item, code="invalid_run")
        for item in _sequence(report.get("runs"), code="three_runs_required")
    ]
    if len(runs) != 3:
        raise ProfilePublicationError("three_runs_required")
    indexes: set[int] = set()
    for run in runs:
        required = {
            "run_index",
            "seed",
            "case_count",
            "expected_call_count",
            "strict_valid_call_count",
            "strict_valid_call_rate",
            "repaired_valid_call_count",
            "repaired_valid_call_rate",
            "effective_valid_call_count",
            "effective_valid_call_rate",
            "correlated_call_count",
            "correlation_rate",
            "repair_attempts_by_kind",
            "cases",
        }
        if set(run) != required:
            raise ProfilePublicationError("invalid_run")
        index = _count(run["run_index"], code="invalid_run")
        if index not in {1, 2, 3} or index in indexes:
            raise ProfilePublicationError("invalid_run")
        indexes.add(index)
        expected = _count(run["expected_call_count"], code="invalid_run")
        if expected < 1:
            raise ProfilePublicationError("empty_evaluation")
        strict_valid = _count(run["strict_valid_call_count"], code="invalid_run")
        repaired_valid = _count(run["repaired_valid_call_count"], code="invalid_run")
        effective_valid = _count(run["effective_valid_call_count"], code="invalid_run")
        correlated = _count(run["correlated_call_count"], code="invalid_run")
        if not (
            strict_valid <= effective_valid <= expected
            and repaired_valid <= effective_valid
            and correlated <= effective_valid
        ):
            raise ProfilePublicationError("inconsistent_run_counts")
        rate_pairs = (
            (run["strict_valid_call_rate"], strict_valid),
            (run["repaired_valid_call_rate"], repaired_valid),
            (run["effective_valid_call_rate"], effective_valid),
            (run["correlation_rate"], correlated),
        )
        for actual, count in rate_pairs:
            if _number(actual, code="invalid_run_rate") != _rate(count, expected):
                raise ProfilePublicationError("inconsistent_run_rate")
        if type(run["case_count"]) is not int or run["case_count"] < 1:
            raise ProfilePublicationError("invalid_run")
        _mapping(run["repair_attempts_by_kind"], code="invalid_run")
        cases = _sequence(run["cases"], code="invalid_run")
        if len(cases) != run["case_count"]:
            raise ProfilePublicationError("inconsistent_case_count")

    aggregate = _mapping(report.get("aggregate"), code="invalid_aggregate")
    required_aggregate = {
        "expected_call_count",
        "strict_valid_call_count",
        "strict_valid_call_rate",
        "repaired_valid_call_count",
        "repaired_valid_call_rate",
        "effective_valid_call_count",
        "effective_valid_call_rate",
        "correlated_call_count",
        "correlation_rate",
        "repair_attempts_by_kind",
        "minimum_run_effective_valid_call_rate",
        "maximum_run_repaired_valid_call_rate",
        "support_status",
    }
    if set(aggregate) != required_aggregate:
        raise ProfilePublicationError("invalid_aggregate")
    expected_total = sum(int(run["expected_call_count"]) for run in runs)
    count_fields = (
        "strict_valid_call_count",
        "repaired_valid_call_count",
        "effective_valid_call_count",
        "correlated_call_count",
    )
    if aggregate["expected_call_count"] != expected_total:
        raise ProfilePublicationError("inconsistent_aggregate")
    for field in count_fields:
        if aggregate[field] != sum(int(run[field]) for run in runs):
            raise ProfilePublicationError("inconsistent_aggregate")
    aggregate_rate_fields = (
        ("strict_valid_call_rate", "strict_valid_call_count"),
        ("repaired_valid_call_rate", "repaired_valid_call_count"),
        ("effective_valid_call_rate", "effective_valid_call_count"),
        ("correlation_rate", "correlated_call_count"),
    )
    for rate_field, count_field in aggregate_rate_fields:
        if _number(aggregate[rate_field], code="invalid_aggregate") != _rate(
            int(aggregate[count_field]),
            expected_total,
        ):
            raise ProfilePublicationError("inconsistent_aggregate")
    minimum = min(float(run["effective_valid_call_rate"]) for run in runs)
    maximum_repair = max(float(run["repaired_valid_call_rate"]) for run in runs)
    if aggregate["minimum_run_effective_valid_call_rate"] != minimum:
        raise ProfilePublicationError("inconsistent_aggregate")
    if aggregate["maximum_run_repaired_valid_call_rate"] != maximum_repair:
        raise ProfilePublicationError("inconsistent_aggregate")
    return runs, aggregate


def classify_report(report: Mapping[str, object]) -> PublishedSupportStatus:
    """Derive support status from validated three-run evidence."""

    runs, _ = _validated_runs(report)
    if report.get("deterministic_fixtures_passed") is not True:
        return PublishedSupportStatus.DISABLED
    if any(float(run["correlation_rate"]) != 1.0 for run in runs):
        return PublishedSupportStatus.DISABLED
    minimum_rate = min(float(run["effective_valid_call_rate"]) for run in runs)
    maximum_repair = max(float(run["repaired_valid_call_rate"]) for run in runs)
    if minimum_rate >= _STABLE_MIN_RATE and maximum_repair <= _MAX_STABLE_REPAIR_RATE:
        return PublishedSupportStatus.STABLE
    if minimum_rate >= _PREVIEW_MIN_RATE:
        return PublishedSupportStatus.PREVIEW
    return PublishedSupportStatus.EXPERIMENTAL


def validate_report(report: Mapping[str, object]) -> PublishedSupportStatus:
    """Validate content-free evidence and return its derived support status."""

    root = _mapping(report, code="invalid_report")
    if root.get("schema_version") != _REPORT_SCHEMA:
        raise ProfilePublicationError("unsupported_report_schema")
    required = {
        "schema_version",
        "ppmlx",
        "model",
        "profile",
        "environment",
        "generation_settings",
        "case_set",
        "deterministic_fixtures_passed",
        "runs",
        "aggregate",
    }
    if set(root) != required:
        raise ProfilePublicationError("invalid_report")
    _assert_no_content(root)
    _validate_identity(root)
    _validate_profile(root)
    _validate_environment(root)
    _validate_generation_settings(root)
    _validate_case_set(root)
    _, aggregate = _validated_runs(root)
    derived = classify_report(root)
    if aggregate["support_status"] != derived.value:
        raise ProfilePublicationError("support_status_mismatch")
    return derived


def load_reports(directory: Path) -> tuple[dict[str, object], ...]:
    """Load all reviewed JSON reports from one evidence directory."""

    reports: list[dict[str, object]] = []
    if not directory.exists():
        return ()
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ProfilePublicationError("invalid_report_file") from None
        report = _mapping(value, code="invalid_report")
        validate_report(report)
        reports.append(report)
    return tuple(reports)


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def render_capability_matrix(reports: Sequence[Mapping[str, object]]) -> str:
    """Render reviewed evidence and explicit not-evaluated parser profiles."""

    reviewed: list[dict[str, object]] = []
    evaluated_profiles: set[str] = set()
    for report in reports:
        root = _mapping(report, code="invalid_report")
        validate_report(root)
        reviewed.append(root)
        profile = _mapping(root["profile"], code="invalid_profile")
        evaluated_profiles.add(str(profile["normalization_profile"]))

    lines = [
        "# Local tool capability matrix",
        "",
        "This matrix contains exact reviewed model profiles. A parser profile without reviewed evidence is not a model capability claim.",
        "",
        "| Model | Revision | Format | Level | Repair | Strict | Effective | Correlation | Status |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for report in sorted(
        reviewed,
        key=lambda item: str(_mapping(item["model"], code="invalid_model_identity")["repository"]),
    ):
        model = _mapping(report["model"], code="invalid_model_identity")
        profile = _mapping(report["profile"], code="invalid_profile")
        aggregate = _mapping(report["aggregate"], code="invalid_aggregate")
        repair = profile["repair_policy"] or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model["repository"]),
                    f"`{str(model['revision'])[:12]}`",
                    f"`{profile['normalization_profile']}`",
                    f"`{profile['capability_level']}`",
                    f"`{repair}`",
                    _percent(aggregate["strict_valid_call_rate"]),
                    _percent(aggregate["effective_valid_call_rate"]),
                    _percent(aggregate["correlation_rate"]),
                    f"**{aggregate['support_status']}**",
                ]
            )
            + " |"
        )

    for contract in list_tool_profile_contracts():
        name = contract.normalization_profile.value
        if name in evaluated_profiles:
            continue
        lines.append(
            "| — | — | "
            f"`{name}` | `{contract.capability_level.value}` | "
            "`none` | — | — | — | **not_evaluated** |"
        )

    lines.extend(
        [
            "",
            "## Publication gates",
            "",
            "- Deterministic parser and correlation fixtures must pass at 100%.",
            "- Each exact model profile must complete three fixed runs.",
            "- Stable requires at least 98% effective valid calls in every run and no more than 2% repaired valid calls in any run.",
            "- Preview requires at least 95% effective valid calls in every run.",
            "- Lower results are experimental. Any fixture or correlation failure disables the profile.",
            "- Family-name matching does not create a capability claim.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ProfilePublicationError",
    "PublishedSupportStatus",
    "classify_report",
    "load_reports",
    "render_capability_matrix",
    "validate_report",
]
