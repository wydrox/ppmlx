#!/usr/bin/env python3
"""Run the fixed local tool-profile evaluation on Apple Silicon."""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Sequence

from ppmlx import __version__
from ppmlx.engine import TextEngine
from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.local_runtime.profile_evaluation import (
    AttemptEvaluation,
    CaseEvaluation,
    ProfileEvaluationError,
    RunEvaluation,
    ToolEvaluationCase,
    build_report,
    evaluate_generated_output,
    load_case_set,
)
from ppmlx.local_runtime.tool_argument_repair import ToolArgumentRepairPolicy
from ppmlx.local_runtime.tool_profiles import (
    ToolCapabilityLevel,
    ToolProfileContract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_SET = ROOT / "tests" / "fixtures" / "tool_profile_eval" / "cases-v1.json"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _run_command(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise ProfileEvaluationError("environment_probe_failed") from None
    value = result.stdout.strip()
    if not value:
        raise ProfileEvaluationError("environment_probe_failed")
    return value


def _git_commit() -> str:
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=ROOT)
    if _SHA_RE.fullmatch(commit) is None:
        raise ProfileEvaluationError("invalid_ppmlx_commit")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise ProfileEvaluationError("environment_probe_failed") from None
    if status:
        raise ProfileEvaluationError("dirty_checkout")
    return commit


def _sysctl(name: str) -> str:
    return _run_command(["sysctl", "-n", name])


def _apple_environment() -> tuple[str, int, str]:
    if platform.system() != "Darwin" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        raise ProfileEvaluationError("apple_silicon_required")
    chip = _sysctl("machdep.cpu.brand_string")
    if not chip.lower().startswith("apple"):
        chip = _sysctl("hw.model")
    try:
        memory_gb = round(int(_sysctl("hw.memsize")) / (1024**3))
    except ValueError:
        raise ProfileEvaluationError("environment_probe_failed") from None
    macos_version = platform.mac_ver()[0]
    if memory_gb < 1 or not macos_version:
        raise ProfileEvaluationError("environment_probe_failed")
    return chip, memory_gb, macos_version


def _immutable_revision(value: str, *, code: str) -> str:
    normalized = value.strip().lower()
    if _SHA_RE.fullmatch(normalized) is None:
        raise ProfileEvaluationError(code)
    return normalized


def _repair_policy(value: str) -> ToolArgumentRepairPolicy | None:
    if value == "none":
        return None
    try:
        return ToolArgumentRepairPolicy(value)
    except ValueError:
        raise ProfileEvaluationError("invalid_repair_policy") from None


def _failed_case(case: ToolEvaluationCase, code: str) -> CaseEvaluation:
    expected = len(case.expected_calls)
    attempt = AttemptEvaluation(
        expected_calls=expected,
        valid_calls=0,
        correlated_calls=0,
        repair_attempts=(),
        repaired_valid_calls=0,
        error_code=code,
    )
    return CaseEvaluation(case_id=case.case_id, strict=attempt, effective=attempt)


def _case_seed(seed_base: int, run_index: int, case_index: int, case_count: int) -> int:
    return seed_base + ((run_index - 1) * case_count) + case_index


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """Run exactly three fixed evaluation passes and build a safe report."""

    case_set = load_case_set(args.case_set)
    try:
        profile = NormalizationProfile(args.profile)
        capability_level = ToolCapabilityLevel(args.capability_level)
    except ValueError:
        raise ProfileEvaluationError("invalid_profile_metadata") from None
    policy = _repair_policy(args.repair_policy)
    ToolProfileContract(
        normalization_profile=profile,
        capability_level=capability_level,
        repair_policy=policy,
    )
    if args.temperature != 0:
        raise ProfileEvaluationError("deterministic_temperature_required")
    if args.max_tokens < 1 or args.seed_base < 0:
        raise ProfileEvaluationError("invalid_generation_settings")

    ppmlx_commit = _git_commit()
    apple_chip, memory_gb, macos_version = _apple_environment()
    model_revision = _immutable_revision(
        args.model_revision,
        code="immutable_model_revision_required",
    )
    tokenizer_revision = _immutable_revision(
        args.tokenizer_revision,
        code="immutable_tokenizer_revision_required",
    )

    engine = TextEngine(max_loaded=1)
    runs: list[RunEvaluation] = []
    case_count = len(case_set.cases)
    for run_index in range(1, 4):
        case_results: list[CaseEvaluation] = []
        run_seed = _case_seed(args.seed_base, run_index, 0, case_count)
        for case_index, case in enumerate(case_set.cases):
            seed = _case_seed(args.seed_base, run_index, case_index, case_count)
            try:
                generated = engine.generate(
                    args.model,
                    [dict(message) for message in case.messages],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=seed,
                    strip_thinking=True,
                    enable_thinking=False,
                    tools=[dict(tool) for tool in case.tools],
                    strict_tools=True,
                )
            except Exception:
                case_results.append(_failed_case(case, "generation_failed"))
                continue
            case_results.append(
                evaluate_generated_output(
                    generated.text,
                    case=case,
                    profile=profile,
                    repair_policy=policy,
                )
            )
        runs.append(
            RunEvaluation(
                run_index=run_index,
                seed=run_seed,
                cases=tuple(case_results),
            )
        )

    return build_report(
        ppmlx_version=__version__,
        ppmlx_commit=ppmlx_commit,
        model_repository=args.model,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        quantization=args.quantization,
        normalization_profile=profile,
        capability_level=capability_level.value,
        repair_policy=policy,
        apple_chip=apple_chip,
        memory_gb=memory_gb,
        macos_version=macos_version,
        generation_settings={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed_base": args.seed_base,
            "seed_formula": "seed_base + ((run_index - 1) * case_count) + case_index",
            "enable_thinking": False,
            "strip_thinking": True,
            "strict_tools": True,
        },
        case_set=case_set,
        runs=tuple(runs),
        deterministic_fixtures_passed=True,
    )


def _write_report(path: Path, report: dict[str, object]) -> None:
    if path.suffix.lower() != ".json":
        raise ProfileEvaluationError("json_output_required")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run three fixed local tool-profile evaluation passes on Apple Silicon."
    )
    parser.add_argument("--model", required=True, help="Exact model repository or local path.")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Immutable 40-character model commit revision.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="Immutable 40-character tokenizer commit revision.",
    )
    parser.add_argument("--quantization", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=[profile.value for profile in NormalizationProfile],
    )
    parser.add_argument(
        "--capability-level",
        required=True,
        choices=[level.value for level in ToolCapabilityLevel],
    )
    parser.add_argument(
        "--repair-policy",
        default="none",
        choices=["none", *(policy.value for policy in ToolArgumentRepairPolicy)],
    )
    parser.add_argument("--case-set", type=Path, default=DEFAULT_CASE_SET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=1000)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run_evaluation(args)
        _write_report(args.output, report)
    except ProfileEvaluationError as error:
        parser.error(error.code)
    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    print(
        f"Wrote {args.output} with support_status={aggregate['support_status']} "
        f"and effective_valid_call_rate={aggregate['effective_valid_call_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
