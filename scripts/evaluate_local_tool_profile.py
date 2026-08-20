#!/usr/bin/env python3
"""Run three fixed local tool-profile evaluations on Apple Silicon."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ppmlx.engine import TextEngine
from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.local_runtime.profile_evaluation import load_case_set
from ppmlx.local_runtime.profile_runner import (
    GenerationSettings,
    ProfileRunnerError,
    TextEngineProfileGenerator,
    detect_apple_environment,
    run_profile_evaluation,
)
from ppmlx.local_runtime.tool_argument_repair import ToolArgumentRepairPolicy
from ppmlx.local_runtime.tool_profiles import ToolCapabilityLevel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_SET = ROOT / "tests" / "fixtures" / "tool_profile_eval" / "cases-v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one exact local model tool profile without storing model output."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument(
        "--normalization-profile",
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
        choices=["none", ToolArgumentRepairPolicy.BOUNDED_JSON_V1.value],
        default="none",
    )
    parser.add_argument("--case-set", type=Path, default=DEFAULT_CASE_SET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seeds", nargs=3, type=int, default=(17, 29, 43))
    parser.add_argument(
        "--deterministic-fixtures-passed",
        action="store_true",
        help="Confirm that deterministic parser and correlation fixtures passed for this commit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repair_policy = (
        None
        if args.repair_policy == "none"
        else ToolArgumentRepairPolicy(args.repair_policy)
    )
    try:
        case_set = load_case_set(args.case_set)
        settings = GenerationSettings(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seeds=tuple(args.seeds),
        )
        report = run_profile_evaluation(
            repository_root=ROOT,
            case_set_path=args.case_set,
            case_set=case_set,
            model_path=args.model_path,
            model_repository=args.model_repository,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            quantization=args.quantization,
            normalization_profile=NormalizationProfile(args.normalization_profile),
            capability_level=ToolCapabilityLevel(args.capability_level),
            repair_policy=repair_policy,
            environment=detect_apple_environment(),
            settings=settings,
            generate=TextEngineProfileGenerator(TextEngine(max_loaded=1)),
            deterministic_fixtures_passed=args.deterministic_fixtures_passed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ProfileRunnerError, ValueError) as error:
        code = getattr(error, "code", "evaluation_failed")
        print(f"Tool profile evaluation failed: {code}", file=sys.stderr)
        return 2
    print(f"Wrote validated tool-profile evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
