"""Tests for ppmlx.memory_eval — temporal-memory anti-garbage eval suite."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.memory_eval import (
    CaseRun,
    MemoryEvalRunner,
    STATUS_ACTIVE,
    STATUS_REJECTED,
    ValidatedMemory,
    load_builtin_cases,
    save_report,
)


def test_builtin_memory_eval_passes_reference_gate():
    report = MemoryEvalRunner().run()

    assert report.passed is True
    assert report.summary["false_active_count"] == 0
    assert report.summary["secret_leak_count"] == 0
    assert report.summary["scope_leakage_count"] == 0
    assert report.summary["bad_injection_count"] == 0
    assert report.summary["manual_review_burden"] == 0
    assert report.summary["latency_ms"]["validation_p95"] < 50
    assert report.summary["latency_ms"]["retrieval_p95"] < 50


def test_omitted_bad_candidate_counts_as_rejected():
    cases = load_builtin_cases()
    omitted_secret_run = CaseRun(
        case_id="secret_rejection",
        validated=[],
        retrieved_ids=[],
        timings_ms={"validation": 1.0, "retrieval": 1.0, "total": 2.0},
    )

    report = MemoryEvalRunner().run(cases=cases, case_runs={"secret_rejection": omitted_secret_run})

    assert report.passed is True
    assert report.summary["secret_leak_count"] == 0
    assert report.summary["false_active_count"] == 0


def test_secret_active_prediction_fails_suite():
    cases = load_builtin_cases()
    bad_secret_run = CaseRun(
        case_id="secret_rejection",
        validated=[ValidatedMemory(id="c-secret", status=STATUS_ACTIVE, scope="global", confidence=0.99)],
        retrieved_ids=["c-secret"],
        timings_ms={"validation": 1.0, "retrieval": 1.0, "total": 2.0},
    )

    report = MemoryEvalRunner().run(cases=cases, case_runs={"secret_rejection": bad_secret_run})

    assert report.passed is False
    assert report.summary["secret_leak_count"] == 1
    assert report.summary["bad_injection_count"] == 1
    assert "c-secret" in report.summary["ids"]["secret_leaks"]


def test_scope_leakage_prediction_fails_suite():
    cases = load_builtin_cases()
    wrong_scope_run = CaseRun(
        case_id="project_decision_scope",
        validated=[
            ValidatedMemory(id="c-ppmlx-position", status=STATUS_ACTIVE, scope="global", confidence=0.93),
            ValidatedMemory(id="c-ppmlx-position-global", status=STATUS_REJECTED, scope="global", confidence=0.88),
        ],
        retrieved_ids=["c-ppmlx-position"],
        timings_ms={"validation": 1.0, "retrieval": 1.0, "total": 2.0},
    )

    report = MemoryEvalRunner().run(cases=cases, case_runs={"project_decision_scope": wrong_scope_run})

    assert report.passed is False
    assert report.summary["scope_leakage_count"] == 1
    assert "c-ppmlx-position" in report.summary["ids"]["scope_leaks"]


def test_save_report_writes_json(tmp_path):
    report = MemoryEvalRunner().run()
    path = save_report(report, tmp_path / "memory-eval" / "report.json")

    data = json.loads(path.read_text())
    assert data["passed"] is True
    assert data["summary"]["cases"] >= 1
    assert "thresholds" in data


def test_memory_eval_cli_json_output():
    result = CliRunner().invoke(app, ["memory-eval", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["passed"] is True
    assert data["summary"]["secret_leak_count"] == 0
