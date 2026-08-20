"""Run exact local model tool-profile evaluations without retaining model output."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ppmlx import __version__
from ppmlx.engine import TextEngine

from .normalization import NormalizationProfile
from .profile_evaluation import (
    AttemptEvaluation,
    CaseEvaluation,
    RunEvaluation,
    ToolEvaluationCase,
    ToolEvaluationCaseSet,
    build_report,
    evaluate_generated_output,
)
from .profile_publication import finalize_report
from .tool_argument_repair import ToolArgumentRepairPolicy
from .tool_profiles import ToolCapabilityLevel


class ProfileRunnerError(ValueError):
    """A safe runner error that contains no model output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"tool profile runner error {code}")


@dataclass(frozen=True, slots=True)
class AppleEvaluationEnvironment:
    """Apple Silicon environment metadata required by publication."""

    chip: str
    memory_gb: int
    macos_version: str
    architecture: str = "arm64"

    def __post_init__(self) -> None:
        if type(self.chip) is not str or not self.chip:
            raise ProfileRunnerError("invalid_environment")
        if type(self.memory_gb) is not int or self.memory_gb < 1:
            raise ProfileRunnerError("invalid_environment")
        if type(self.macos_version) is not str or not self.macos_version:
            raise ProfileRunnerError("invalid_environment")
        if self.architecture != "arm64":
            raise ProfileRunnerError("apple_silicon_required")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Deterministic settings shared by all three fixed runs."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seeds: tuple[int, int, int] = (17, 29, 43)

    def __post_init__(self) -> None:
        if type(self.temperature) not in {int, float} or self.temperature < 0:
            raise ProfileRunnerError("invalid_generation_settings")
        if type(self.top_p) not in {int, float} or not 0 < self.top_p <= 1:
            raise ProfileRunnerError("invalid_generation_settings")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ProfileRunnerError("invalid_generation_settings")
        if (
            type(self.seeds) is not tuple
            or len(self.seeds) != 3
            or any(type(seed) is not int for seed in self.seeds)
            or len(set(self.seeds)) != 3
        ):
            raise ProfileRunnerError("invalid_generation_settings")

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "max_tokens": self.max_tokens,
            "seeds": list(self.seeds),
        }


class ProfileGenerator(Protocol):
    """Generate one complete local model output for a fixed case."""

    def __call__(
        self,
        *,
        model_path: str,
        case: ToolEvaluationCase,
        seed: int,
        settings: GenerationSettings,
    ) -> str: ...


@dataclass(slots=True)
class TextEngineProfileGenerator:
    """MLX generator used by the manual Apple Silicon evaluation command."""

    engine: TextEngine

    def __call__(
        self,
        *,
        model_path: str,
        case: ToolEvaluationCase,
        seed: int,
        settings: GenerationSettings,
    ) -> str:
        result = self.engine.generate(
            model_path,
            list(case.messages),
            temperature=float(settings.temperature),
            top_p=float(settings.top_p),
            max_tokens=settings.max_tokens,
            seed=seed,
            strip_thinking=True,
            enable_thinking=False,
            tools=list(case.tools),
            strict_tools=True,
        )
        if type(result.text) is not str:
            raise ProfileRunnerError("invalid_generation")
        return result.text


def _sysctl_text(name: str) -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        raise ProfileRunnerError("environment_detection_failed") from None
    value = result.stdout.strip()
    if not value:
        raise ProfileRunnerError("environment_detection_failed")
    return value


def detect_apple_environment() -> AppleEvaluationEnvironment:
    """Detect the Apple Silicon environment and fail closed elsewhere."""

    architecture = platform.machine().lower()
    macos_version = platform.mac_ver()[0]
    if architecture != "arm64" or not macos_version:
        raise ProfileRunnerError("apple_silicon_required")
    chip = _sysctl_text("machdep.cpu.brand_string")
    try:
        memory_bytes = int(_sysctl_text("hw.memsize"))
    except ValueError:
        raise ProfileRunnerError("environment_detection_failed") from None
    return AppleEvaluationEnvironment(
        chip=chip,
        memory_gb=max(1, round(memory_bytes / (1024**3))),
        macos_version=macos_version,
        architecture=architecture,
    )


def current_git_commit(repository_root: Path) -> str:
    """Return the clean repository commit used by an evaluation."""

    try:
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise ProfileRunnerError("git_evidence_unavailable") from None
    if status.stdout.strip():
        raise ProfileRunnerError("dirty_evaluation_checkout")
    value = commit.stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ProfileRunnerError("git_evidence_unavailable")
    return value


def case_set_sha256(path: Path) -> str:
    """Return the exact case-set content digest."""

    try:
        data = path.read_bytes()
    except OSError:
        raise ProfileRunnerError("case_set_unavailable") from None
    return hashlib.sha256(data).hexdigest()


def _generation_failure(case: ToolEvaluationCase) -> CaseEvaluation:
    failed = AttemptEvaluation(
        expected_calls=len(case.expected_calls),
        valid_calls=0,
        correlated_calls=0,
        repair_attempts=(),
        repaired_valid_calls=0,
        error_code="generation_failed",
    )
    return CaseEvaluation(case_id=case.case_id, strict=failed, effective=failed)


def _run_once(
    *,
    run_index: int,
    seed: int,
    case_set: ToolEvaluationCaseSet,
    model_path: str,
    normalization_profile: NormalizationProfile,
    repair_policy: ToolArgumentRepairPolicy | None,
    settings: GenerationSettings,
    generate: ProfileGenerator,
) -> RunEvaluation:
    results: list[CaseEvaluation] = []
    for case in case_set.cases:
        try:
            output = generate(
                model_path=model_path,
                case=case,
                seed=seed,
                settings=settings,
            )
            if type(output) is not str:
                raise ProfileRunnerError("invalid_generation")
        except Exception:
            results.append(_generation_failure(case))
            continue
        results.append(
            evaluate_generated_output(
                output,
                case=case,
                profile=normalization_profile,
                repair_policy=repair_policy,
            )
        )
    return RunEvaluation(run_index=run_index, seed=seed, cases=tuple(results))


def run_profile_evaluation(
    *,
    repository_root: Path,
    case_set_path: Path,
    case_set: ToolEvaluationCaseSet,
    model_path: str,
    model_repository: str,
    model_revision: str,
    tokenizer_revision: str,
    quantization: str,
    normalization_profile: NormalizationProfile,
    capability_level: ToolCapabilityLevel,
    repair_policy: ToolArgumentRepairPolicy | None,
    environment: AppleEvaluationEnvironment,
    settings: GenerationSettings,
    generate: ProfileGenerator,
    deterministic_fixtures_passed: bool = True,
    ppmlx_commit: str | None = None,
) -> dict[str, object]:
    """Run three fixed evaluations and return one validated safe report."""

    if type(model_path) is not str or not model_path:
        raise ProfileRunnerError("model_path_required")
    commit = ppmlx_commit or current_git_commit(repository_root)
    runs = tuple(
        _run_once(
            run_index=index,
            seed=seed,
            case_set=case_set,
            model_path=model_path,
            normalization_profile=normalization_profile,
            repair_policy=repair_policy,
            settings=settings,
            generate=generate,
        )
        for index, seed in enumerate(settings.seeds, start=1)
    )
    report = build_report(
        ppmlx_version=__version__,
        ppmlx_commit=commit,
        model_repository=model_repository,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        quantization=quantization,
        normalization_profile=normalization_profile,
        capability_level=capability_level.value,
        repair_policy=repair_policy,
        apple_chip=environment.chip,
        memory_gb=environment.memory_gb,
        macos_version=environment.macos_version,
        generation_settings=settings.to_dict(),
        case_set=case_set,
        runs=runs,
        deterministic_fixtures_passed=deterministic_fixtures_passed,
    )
    return finalize_report(
        report,
        architecture=environment.architecture,
        case_set_sha256=case_set_sha256(case_set_path),
    )


__all__ = [
    "AppleEvaluationEnvironment",
    "GenerationSettings",
    "ProfileGenerator",
    "ProfileRunnerError",
    "TextEngineProfileGenerator",
    "case_set_sha256",
    "current_git_commit",
    "detect_apple_environment",
    "run_profile_evaluation",
]
