#!/usr/bin/env python3
"""Run the deterministic parser and repair fixtures and record the artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ppmlx.local_runtime.deterministic_fixtures import (
    DeterministicFixtureError,
    run_deterministic_fixtures,
)


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every deterministic normalization, repair, and evaluation "
            "fixture for this commit and record a content-free artifact."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the tool-profile-fixtures/v1 artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = run_deterministic_fixtures(repository_root=ROOT)
    except DeterministicFixtureError as error:
        print(f"Deterministic fixture run failed: {error.code}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote deterministic fixture evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
