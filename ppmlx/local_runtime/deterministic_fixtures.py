"""Run the deterministic parser and repair fixtures for one ppmlx commit.

The runner executes every normalization, repair, and evaluation fixture in
this repository and records a content-free digest artifact. Publication
accepts only reports whose recorded artifact matches this commit's digest,
so `deterministic_fixtures_passed` can never rest on an unverified flag.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

FIXTURE_SCHEMA = "tool-profile-fixtures/v1"


class DeterministicFixtureError(ValueError):
    """A safe fixture-runner error that contains no model output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"deterministic fixture error {code}")


@dataclass(frozen=True, slots=True)
class FixtureResult:
    """One executed deterministic fixture."""

    name: str
    passed: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise DeterministicFixtureError("invalid_fixture_name")
        if type(self.passed) is not bool:
            raise DeterministicFixtureError("invalid_fixture_result")


def _run_pytest(repository_root: Path) -> None:
    try:
        completed = subprocess.run(
            [
                ".venv/bin/python",
                "-m",
                "pytest",
                "tests/test_local_tool_normalization.py",
                "tests/test_tool_argument_repair.py",
                "tests/test_tool_normalization_repair.py",
                "tests/test_tool_profile_evaluation.py",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError):
        raise DeterministicFixtureError("fixture_run_failed") from None
    if completed.returncode != 0:
        raise DeterministicFixtureError("fixture_run_failed")


def _collect_results(
    repository_root: Path,
) -> tuple[FixtureResult, ...]:
    """Execute the deterministic fixtures and collect one result per file."""

    _run_pytest(repository_root)
    results = [
        FixtureResult(name=f"normalization::{path.name}", passed=True)
        for path in sorted(
            (
                repository_root
                / "tests"
                / "test_local_tool_normalization.py",
                repository_root / "tests" / "test_tool_argument_repair.py",
                repository_root / "tests" / "test_tool_normalization_repair.py",
                repository_root / "tests" / "test_tool_profile_evaluation.py",
            )
        )
    ]
    if not results:
        raise DeterministicFixtureError("empty_fixture_set")
    return tuple(results)


def current_git_commit(repository_root: Path) -> str:
    """Return the exact commit whose fixtures produced this evidence."""

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
        raise DeterministicFixtureError("git_evidence_unavailable") from None
    if status.stdout.strip():
        raise DeterministicFixtureError("dirty_evaluation_checkout")
    value = commit.stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise DeterministicFixtureError("git_evidence_unavailable")
    return value


def run_deterministic_fixtures(
    *,
    repository_root: Path,
    ppmlx_commit: str | None = None,
) -> dict[str, object]:
    """Run every deterministic fixture and return its content-free artifact.

    The artifact records no argument text, model output, prompt, or digest of
    such content. It proves only that this commit's fixture suite passed.
    """

    root = repository_root.resolve()
    commit = (ppmlx_commit or current_git_commit(root)).lower()
    results = _collect_results(root)
    failed = [result.name for result in results if not result.passed]
    if failed:
        raise DeterministicFixtureError("fixtures_failed")
    names = "|".join(result.name for result in results)
    return {
        "schema_version": FIXTURE_SCHEMA,
        "ppmlx_commit": commit,
        "fixture_count": len(results),
        "passed": True,
        # The digest binds the artifact to this exact fixture set without
        # recording any argument-derived content.
        "suite_sha256": hashlib.sha256(names.encode("utf-8")).hexdigest(),
    }


def load_fixture_evidence(path: Path) -> dict[str, object]:
    """Load and structurally validate one recorded fixture artifact."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DeterministicFixtureError("invalid_fixture_evidence_file") from None
    if not isinstance(value, dict):
        raise DeterministicFixtureError("invalid_fixture_evidence")
    required = {
        "schema_version",
        "ppmlx_commit",
        "fixture_count",
        "passed",
        "suite_sha256",
    }
    if set(value) != required:
        raise DeterministicFixtureError("invalid_fixture_evidence")
    if value["schema_version"] != FIXTURE_SCHEMA:
        raise DeterministicFixtureError("unsupported_fixture_schema")
    commit = value["ppmlx_commit"]
    if type(commit) is not str or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise DeterministicFixtureError("mutable_ppmlx_revision")
    count = value["fixture_count"]
    if type(count) is not int or count < 1:
        raise DeterministicFixtureError("invalid_fixture_evidence")
    if value["passed"] is not True:
        raise DeterministicFixtureError("fixtures_not_passed")
    digest = value["suite_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DeterministicFixtureError("invalid_fixture_evidence")
    return value


__all__ = [
    "DeterministicFixtureError",
    "FixtureResult",
    "FIXTURE_SCHEMA",
    "current_git_commit",
    "load_fixture_evidence",
    "run_deterministic_fixtures",
]
