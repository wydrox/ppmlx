"""Keep the public local tool capability matrix generated from reviewed evidence."""
from __future__ import annotations

import json
from pathlib import Path

from ppmlx.local_runtime.deterministic_fixtures import load_fixture_evidence
from ppmlx.local_runtime.profile_publication import (
    load_reports,
    render_capability_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "capabilities" / "tool-profile-evidence"
MATRIX = ROOT / "docs" / "capabilities" / "tool-profiles.md"


def _fixtures_evidence(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "fixtures.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "tool-profile-fixtures/v1",
                # The repository ships no evaluated profile, so no report can
                # reference this artifact; only its presence is asserted.
                "ppmlx_commit": "a" * 40,
                "fixture_count": 4,
                "passed": True,
                "suite_sha256": "b" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return load_fixture_evidence(path)


def test_checked_in_capability_matrix_matches_reviewed_evidence(
    tmp_path: Path,
) -> None:
    evidence = _fixtures_evidence(tmp_path)

    assert MATRIX.read_text(encoding="utf-8") == render_capability_matrix(
        load_reports(EVIDENCE, fixtures_evidence=evidence),
        fixtures_evidence=evidence,
    )


def test_initial_matrix_makes_absence_of_model_evidence_explicit(tmp_path: Path) -> None:
    evidence = _fixtures_evidence(tmp_path)
    reports = load_reports(EVIDENCE, fixtures_evidence=evidence)
    matrix = MATRIX.read_text(encoding="utf-8")

    assert reports == ()
    assert matrix.count("**not_evaluated**") == 4
    assert "A parser profile without reviewed evidence is not a model capability claim" in matrix
    assert "Family-name matching does not create a capability claim" in matrix
