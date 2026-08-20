"""Keep the public local tool capability matrix generated from reviewed evidence."""
from __future__ import annotations

from pathlib import Path

from ppmlx.local_runtime.profile_publication import (
    load_reports,
    render_capability_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "capabilities" / "tool-profile-evidence"
MATRIX = ROOT / "docs" / "capabilities" / "tool-profiles.md"


def test_checked_in_capability_matrix_matches_reviewed_evidence() -> None:
    assert MATRIX.read_text(encoding="utf-8") == render_capability_matrix(
        load_reports(EVIDENCE)
    )


def test_initial_matrix_makes_absence_of_model_evidence_explicit() -> None:
    reports = load_reports(EVIDENCE)
    matrix = MATRIX.read_text(encoding="utf-8")

    assert reports == ()
    assert matrix.count("**not_evaluated**") == 4
    assert "A parser profile without reviewed evidence is not a model capability claim" in matrix
    assert "Family-name matching does not create a capability claim" in matrix
