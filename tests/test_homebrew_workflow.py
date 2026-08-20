"""Contract checks for the Homebrew release-update workflow."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "homebrew-update.yml"


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_homebrew_update_supports_validated_manual_recovery() -> None:
    text = workflow_text()

    assert "workflow_dispatch:" in text
    assert "Resolve released source ref" in text
    assert "refs/tags/v${MANUAL_VERSION}" in text
    assert "ref: ${{ steps.release-ref.outputs.ref }}" in text
    assert "The requested recovery version" in text
    assert "github.ref_name" not in text


def test_homebrew_update_fails_clearly_without_tap_write_access() -> None:
    text = workflow_text()

    assert "GH_TOKEN: ${{ secrets.HOMEBREW_TAP_TOKEN }}" in text
    assert "Missing Homebrew tap token" in text
    assert "gh auth status --hostname github.com" in text
    assert ".permissions.push // false" in text
    assert "Homebrew tap token lacks write access" in text


def test_homebrew_update_is_idempotent_and_resumes_automation_branch() -> None:
    text = workflow_text()

    assert "gh pr list" in text
    assert "Homebrew update already open" in text
    assert "git ls-remote --exit-code --heads origin" in text
    assert 'git checkout -B "$BRANCH" "origin/$BRANCH"' in text
    assert "Homebrew formula already current" in text


def test_homebrew_update_uses_release_artifact_metadata() -> None:
    text = workflow_text()

    assert "https://pypi.org/pypi/ppmlx/" in text
    assert "files.pythonhosted.org" in text
    assert "PyPI must contain one source archive" in text
    assert "sha256" in text
    assert "github.event.workflow_run.head_sha" in text


def test_homebrew_update_pins_external_actions() -> None:
    text = workflow_text()
    action_refs = re.findall(r"uses:\s+([^\s]+)", text)

    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in action_refs)
