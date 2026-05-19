"""Tests for ppmlx.compact_eval long-session compact quality gates."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.compact_eval import CompactEvalRunner, save_report


def test_compact_eval_builtin_tv_session_passes_and_realistic_case_stays_clean():
    report = CompactEvalRunner().run()

    assert report.passed is True
    assert report.summary["wrong_terms"] == 0
    assert report.summary["context_missed_terms"] == 0
    assert report.summary["avg_session_context_coverage"] == 1.0
    by_id = {case.case_id: case for case in report.cases}
    case = by_id["tv_buying_long_session"]
    assert case.compression_ratio >= 4.0
    assert case.reduced_tokens <= 10_000
    assert "LG OLED C4" in case.session_context
    assert "Samsung S90D" in case.session_context
    assert "HDMI 2.1" in case.session_context
    assert "Rejected Samsung CU8000" in case.session_context

    json_case = by_id["tv_buying_json_tool_trace"]
    assert json_case.passed is True
    assert json_case.compression_ratio >= 4.0
    assert "LG OLED C4 price: 4599 PLN" in json_case.session_context
    assert "LG OLED C4 spec hdmi_2_1 = 4" in json_case.session_context
    assert "Rejected Samsung CU8000" in json_case.session_context

    realistic_case = by_id["ppmlx_real_project_handoff"]
    assert realistic_case.passed is True
    assert realistic_case.session_context_coverage == 1.0
    assert realistic_case.session_context_missed_terms == []
    assert realistic_case.wrong_terms == []
    assert "ppmlx todo: add context coverage metrics to compact-eval" in realistic_case.session_context
    assert "budget = 800 PLN" not in realistic_case.session_context


def test_save_compact_eval_report(tmp_path):
    report = CompactEvalRunner().run()
    path = save_report(report, tmp_path / "compact" / "report.json")

    data = json.loads(path.read_text())
    assert data["passed"] is True
    assert data["summary"]["cases"] == 3
    assert data["summary"]["context_missed_terms"] == 0
    assert all("session_context_coverage" in case for case in data["cases"])


def test_compact_eval_cli_json_output():
    result = CliRunner().invoke(app, ["compact-eval", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["passed"] is True
    assert data["summary"]["context_missed_terms"] == 0
