"""Fail-closed deterministic fixture evidence for tool-profile publication."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppmlx.local_runtime.deterministic_fixtures import (
    FIXTURE_SCHEMA,
    DeterministicFixtureError,
    load_fixture_evidence,
    run_deterministic_fixtures,
)
from ppmlx.local_runtime.profile_publication import (
    ProfilePublicationError,
    PublishedSupportStatus,
    load_reports,
    render_capability_matrix,
    validate_report,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "capabilities" / "tool-profile-evidence"
COMMIT = "a" * 40


def _artifact() -> dict[str, object]:
    return {
        "schema_version": FIXTURE_SCHEMA,
        "ppmlx_commit": COMMIT,
        "fixture_count": 4,
        "passed": True,
        "suite_sha256": "b40c4c6328d8b5d8f46cfb568ac6da9a2c540085e6d2d29046c7c8e811011dfc",
    }


def test_runner_executes_the_real_fixture_suite_and_records_a_content_free_artifact() -> None:
    artifact = run_deterministic_fixtures(
        repository_root=ROOT,
        ppmlx_commit=COMMIT,
    )

    assert artifact["schema_version"] == FIXTURE_SCHEMA
    assert artifact["ppmlx_commit"] == COMMIT
    assert artifact["passed"] is True
    assert artifact["fixture_count"] >= 1
    serialized = json.dumps(artifact)
    assert "tool_call" not in serialized
    assert "arguments" not in serialized


def test_runner_rejects_a_dirty_checkout(tmp_path: Path) -> None:
    marker = ROOT / "uncommitted-fixture-probe.txt"
    marker.write_text("probe", encoding="utf-8")
    try:
        with pytest.raises(DeterministicFixtureError) as captured:
            run_deterministic_fixtures(repository_root=ROOT)
    finally:
        marker.unlink()

    assert captured.value.code == "dirty_evaluation_checkout"


def test_runner_records_only_the_requested_commit_without_content() -> None:
    artifact = run_deterministic_fixtures(
        repository_root=ROOT,
        ppmlx_commit="e" * 40,
    )

    assert artifact["ppmlx_commit"] == "e" * 40
    # Publication, not the runner, binds the report commit to the artifact.


def test_load_fixture_evidence_rejects_tampered_or_foreign_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    assert load_fixture_evidence(path) == _artifact()

    for mutation, code in (
        ({"passed": False}, "fixtures_not_passed"),
        ({"schema_version": "tool-profile-fixtures/v2"}, "unsupported_fixture_schema"),
        ({"ppmlx_commit": "z" * 40}, "mutable_ppmlx_revision"),
        ({"suite_sha256": "short"}, "invalid_fixture_evidence"),
        ({"extra": 1}, "invalid_fixture_evidence"),
    ):
        tampered = _artifact()
        tampered.pop(next(iter(mutation)), None)
        tampered.update(mutation)
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(DeterministicFixtureError) as captured:
            load_fixture_evidence(path)
        assert captured.value.code == code

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DeterministicFixtureError) as captured:
        load_fixture_evidence(path)
    assert captured.value.code == "invalid_fixture_evidence_file"


def test_publication_fails_closed_without_recorded_fixture_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_tool_profile_publication import _report

    report = _report()

    with pytest.raises(ProfilePublicationError) as missing:
        validate_report(report)
    assert missing.value.code == "fixtures_evidence_required"

    with pytest.raises(ProfilePublicationError) as foreign:
        validate_report(
            report,
            fixtures_evidence={**_artifact(), "ppmlx_commit": "e" * 40},
        )
    assert foreign.value.code == "fixture_evidence_commit_mismatch"

    assert (
        validate_report(report, fixtures_evidence=_artifact())
        is PublishedSupportStatus.STABLE
    )


def test_load_reports_requires_fixture_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_tool_profile_publication import _report

    monkeypatch.setattr(
        "ppmlx.local_runtime.profile_publication.json.loads",
        lambda _: _report(),
    )

    with pytest.raises(ProfilePublicationError) as captured:
        load_reports(EVIDENCE_DIR)

    assert captured.value.code == "fixtures_evidence_required"


def test_matrix_renderer_requires_fixture_evidence() -> None:
    with pytest.raises(ProfilePublicationError) as captured:
        render_capability_matrix(())

    assert captured.value.code == "fixtures_evidence_required"


def test_checked_in_evidence_directory_ships_no_report() -> None:
    # Without a recorded fixture artifact no checked-in file can pass
    # publication; the repository must keep shipping none.
    reports = load_reports(EVIDENCE_DIR, fixtures_evidence=_artifact())

    assert reports == ()


def test_fixture_flag_alone_is_not_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_tool_profile_publication import _report

    report = _report()
    assert report["deterministic_fixtures_passed"] is True

    with pytest.raises(ProfilePublicationError) as captured:
        validate_report(report)

    assert captured.value.code == "fixtures_evidence_required"
