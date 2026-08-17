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
from ppmlx.memory_engine import MemoryEngine, ShadowMemoryCandidate
from ppmlx.memory_store import MemoryStore


class _ProjectStatusExtractor:
    max_candidates = 1

    def extract(self, event):
        project_id = event.get("project_id")
        label = str(project_id) if project_id is not None else "unscoped"
        return [ShadowMemoryCandidate(
            type="fact",
            subject="ppmlx",
            predicate="status",
            object=f"status for {label}",
            text=f"ppmlx status for {label}.",
            scope="project",
            confidence=0.96,
            source_quote=f"ppmlx status for {label}",
            salience=0.95,
        )]


class _SessionTaskExtractor:
    max_candidates = 1

    def extract(self, event):
        task = str(event["messages"][0]["content"])
        return [ShadowMemoryCandidate(
            type="workflow_state",
            subject="session",
            predicate="current_task",
            object=task,
            text=f"Current task: {task}.",
            scope="session",
            confidence=0.96,
            source_quote=task,
            salience=0.95,
        )]


def test_builtin_memory_eval_passes_reference_gate():
    report = MemoryEvalRunner().run()

    assert report.passed is True
    assert report.summary["status_accuracy"] == 1.0
    assert report.summary["active_recall"] == 1.0
    assert report.summary["retrieval_recall"] == 1.0
    assert report.summary["false_active_count"] == 0
    assert report.summary["secret_leak_count"] == 0
    assert report.summary["scope_leakage_count"] == 0
    assert report.summary["bad_injection_count"] == 0
    assert report.summary["manual_review_burden"] == 0
    graph_quality = report.summary["graph_quality"]
    assert graph_quality["passed"] is True
    assert graph_quality["failure_count"] == 0
    assert all(graph_quality["checks"].values())
    assert graph_quality["metrics"]["processed_jobs"] == graph_quality["metrics"]["queued_jobs"]
    assert graph_quality["metrics"]["jobs_per_second"] > 0
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


def test_missing_active_prediction_fails_suite():
    cases = load_builtin_cases()
    missing_active_run = CaseRun(
        case_id="global_preference_valid",
        validated=[],
        retrieved_ids=[],
        timings_ms={"validation": 1.0, "retrieval": 1.0, "total": 2.0},
    )

    report = MemoryEvalRunner().run(cases=cases, case_runs={"global_preference_valid": missing_active_run})

    assert report.passed is False
    assert report.summary["active_recall"] < 1.0
    assert report.summary["retrieval_recall"] < 1.0
    assert "c-pref-short" in report.summary["ids"]["retrieval_misses"]


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
    assert data["summary"]["graph_quality"]["passed"] is True
    assert "thresholds" in data
    assert "min_active_recall" in data["thresholds"]
    assert "min_retrieval_recall" in data["thresholds"]
    assert "max_graph_quality_failures" in data["thresholds"]


def test_memory_eval_cli_json_output():
    result = CliRunner().invoke(app, ["memory-eval", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["passed"] is True
    assert data["summary"]["secret_leak_count"] == 0
    assert data["summary"]["graph_quality"]["failure_count"] == 0


def test_project_memory_does_not_supersede_a_different_project(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store, extractor=_ProjectStatusExtractor())
    for project_id in ("first-project", "second-project"):
        engine.capture_chat(
            request_id=project_id,
            endpoint="/v1/chat/completions",
            model_alias="test-model",
            model_repo="local/test",
            messages=[{"role": "user", "content": f"ppmlx status for {project_id}."}],
            response_text="",
            project_id=project_id,
        )

    active = store.query_candidates(status="active")

    assert {row["project_id"] for row in active} == {"first-project", "second-project"}


def test_missing_project_id_does_not_match_all_project_slots(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store, extractor=_ProjectStatusExtractor())
    engine.capture_chat(
        request_id="project-memory",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{"role": "user", "content": "ppmlx status for first-project."}],
        response_text="",
        project_id="first-project",
    )
    engine.capture_chat(
        request_id="unscoped-memory",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{"role": "user", "content": "ppmlx status for unscoped."}],
        response_text="",
    )

    legacy_slots = store.find_active_slot(
        type="fact",
        subject="ppmlx",
        predicate="status",
        scope="project",
    )
    exact_unscoped_slots = store.find_active_slot(
        type="fact",
        subject="ppmlx",
        predicate="status",
        scope="project",
        exact_namespace=True,
    )

    assert {row["object"] for row in legacy_slots} == {
        "status for first-project",
        "status for unscoped",
    }
    assert [row["object"] for row in exact_unscoped_slots] == ["status for unscoped"]


def test_preferences_and_requirements_are_additive_without_a_correction(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="additive-memory",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{
            "role": "user",
            "content": (
                "I prefer OLED if burn-in risk is acceptable. "
                "I prefer concise comparison tables. "
                "Need HDMI 2.1 for PS5. Need a 120Hz panel."
            ),
        }],
        response_text="",
        project_id="tv-shopping",
    )

    active = store.query_candidates(status="active", project_id="tv-shopping")
    objects = {row["object"] for row in active}

    assert "OLED if burn-in risk is acceptable" in objects
    assert "concise comparison tables" in objects
    assert "HDMI 2.1 for PS5" in objects
    assert "a 120Hz panel" in objects


def test_preference_correction_only_supersedes_a_related_preference(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="initial-preferences",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{
            "role": "user",
            "content": "I prefer verbose explanations. I prefer OLED if burn-in risk is acceptable.",
        }],
        response_text="",
    )
    engine.capture_chat(
        request_id="preference-correction",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{"role": "user", "content": "Actually, from now on I prefer concise answers."}],
        response_text="",
    )

    active_objects = {
        row["object"] for row in store.query_candidates(status="active")
        if row["type"] == "preference"
    }
    superseded_objects = {
        row["object"] for row in store.query_candidates(status="superseded")
        if row["type"] == "preference"
    }

    assert active_objects == {"concise answers", "OLED if burn-in risk is acceptable"}
    assert superseded_objects == {"verbose explanations"}


def test_constraint_correction_only_supersedes_a_related_requirement(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="initial-requirements",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{
            "role": "user",
            "content": "Need HDMI 2.1 for PS5. Need a 120Hz panel.",
        }],
        response_text="",
        project_id="tv-shopping",
    )
    engine.capture_chat(
        request_id="requirement-correction",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{
            "role": "user",
            "content": "I no longer need HDMI 2.1 for PS5. Need DisplayPort.",
        }],
        response_text="",
        project_id="tv-shopping",
    )

    active_objects = {
        row["object"] for row in store.query_candidates(status="active")
        if row["type"] == "constraint"
    }
    superseded_objects = {
        row["object"] for row in store.query_candidates(status="superseded")
        if row["type"] == "constraint"
    }

    assert active_objects == {"a 120Hz panel", "DisplayPort"}
    assert superseded_objects == {"HDMI 2.1 for PS5"}


def test_unmatched_todo_correction_keeps_existing_todos(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store)
    engine.capture_chat(
        request_id="initial-todos",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{
            "role": "user",
            "content": "Todo: add tests\nTodo: update docs",
        }],
        response_text="",
        project_id="ppmlx",
    )
    engine.capture_chat(
        request_id="todo-correction",
        endpoint="/v1/chat/completions",
        model_alias="test-model",
        model_repo="local/test",
        messages=[{
            "role": "user",
            "content": "Actually, todo: add benchmarks instead.",
        }],
        response_text="",
        project_id="ppmlx",
    )

    active_objects = {
        row["object"] for row in store.query_candidates(status="active")
        if row["type"] == "todo"
    }
    superseded_objects = {
        row["object"] for row in store.query_candidates(status="superseded")
        if row["type"] == "todo"
    }

    assert active_objects == {"add tests", "update docs", "add benchmarks instead"}
    assert superseded_objects == set()


def test_session_slot_uses_project_and_harness_identity(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store=store, extractor=_SessionTaskExtractor())

    def capture(request_id, task, project_id, app_id):
        engine.capture_chat(
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias="test-model",
            model_repo="local/test",
            messages=[{"role": "user", "content": task}],
            response_text="",
            app_id=app_id,
            project_id=project_id,
            session_id="shared-session",
        )

    capture("project-a-codex-old", "task-a-old", "project-a", "codex")
    capture("project-b-codex", "task-b", "project-b", "codex")
    capture("project-a-claude", "task-c", "project-a", "claude")
    capture("project-a-codex-new", "task-a-new", "project-a", "codex")

    active_objects = {
        row["object"] for row in store.query_candidates(status="active")
        if row["type"] == "workflow_state"
    }
    superseded_objects = {
        row["object"] for row in store.query_candidates(status="superseded")
        if row["type"] == "workflow_state"
    }

    assert active_objects == {"task-a-new", "task-b", "task-c"}
    assert superseded_objects == {"task-a-old"}
