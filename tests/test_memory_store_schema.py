from __future__ import annotations

import sqlite3

from ppmlx.memory_store import MemoryStore


def _event(event_id: str, *, project_id: str, session_id: str = "shared-session") -> dict:
    return {
        "event_id": event_id,
        "endpoint": "/v1/chat/completions",
        "project_id": project_id,
        "session_id": session_id,
        "request": {"messages": [{"role": "user", "content": f"status for {project_id}"}]},
        "response_text": "ok",
        "metadata": {},
    }


def _candidate(candidate_id: str, event_id: str, project_id: str, object_: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "event_id": event_id,
        "type": "workflow_state",
        "subject": "shared service",
        "predicate": "latest_status",
        "object": object_,
        "text": f"shared service latest status is {object_}",
        "scope": "project",
        "confidence": 0.95,
        "source_quote": object_,
        "salience": 0.9,
        "metadata": {"project_id": project_id},
    }


def _fact_candidate(candidate_id: str, event_id: str, project_id: str, object_: str) -> dict:
    candidate = _candidate(candidate_id, event_id, project_id, object_)
    candidate["type"] = "fact"
    candidate["predicate"] = "supports"
    return candidate


def _session_candidate(
    candidate_id: str,
    event_id: str,
    project_id: str,
    app_id: str,
    object_: str,
) -> dict:
    candidate = _fact_candidate(candidate_id, event_id, project_id, object_)
    candidate["scope"] = "session"
    candidate["metadata"] = {"project_id": project_id, "app_id": app_id, "session_id": "shared-session"}
    return candidate


def _tables(path):
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def test_init_creates_extraction_atom_alias_tables(tmp_path):
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)

    store.init()

    tables = _tables(db_path)
    assert "memory_extraction_jobs" in tables
    assert "memory_atoms" in tables
    assert "memory_entity_aliases" in tables


def test_enqueue_list_claim_complete_and_fail_extraction_jobs(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    enqueued = store.enqueue_extraction_job(
        {"messages": ["remember this"]},
        job_id="job-test-1",
        source_event_id=None,
        priority=10,
        valid_at="2026-01-01T00:00:00.000",
        metadata={"kind": "unit"},
    )
    assert enqueued["job_id"] == "job-test-1"
    assert enqueued["status"] == "queued"
    assert enqueued["payload"] == {"messages": ["remember this"]}
    assert enqueued["metadata"] == {"kind": "unit"}

    queued = store.list_extraction_jobs(status="queued")
    assert [job["job_id"] for job in queued] == ["job-test-1"]

    claimed = store.claim_extraction_job("worker-a")
    assert claimed is not None
    assert claimed["job_id"] == "job-test-1"
    assert claimed["status"] == "claimed"
    assert claimed["worker_id"] == "worker-a"
    assert claimed["attempts"] == 1
    assert claimed["claimed_at"] is not None
    assert store.claim_extraction_job("worker-b") is None

    assert store.complete_extraction_job("job-test-1", result={"atoms": 2}) is True
    completed = store.get_extraction_job("job-test-1")
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == {"atoms": 2}
    assert completed["completed_at"] is not None
    assert completed["invalid_at"] is not None

    store.enqueue_extraction_job({"messages": ["bad"]}, job_id="job-test-2")
    claimed_failed = store.claim_extraction_job("worker-a")
    assert claimed_failed is not None
    assert claimed_failed["job_id"] == "job-test-2"
    assert store.fail_extraction_job("job-test-2", "boom") is True
    failed = store.get_extraction_job("job-test-2")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"
    assert failed["failed_at"] is not None
    assert failed["invalid_at"] is not None


def test_renew_extraction_job_claim_refreshes_claimed_at(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    store.enqueue_extraction_job({"messages": ["slow"]}, job_id="job-renew")
    claimed = store.claim_extraction_job("worker-a")
    assert claimed is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE memory_extraction_jobs SET claimed_at = '2000-01-01T00:00:00.000' WHERE job_id = ?",
            ("job-renew",),
        )
        conn.commit()

    assert store.renew_extraction_job_claim("job-renew", "worker-a") is True

    renewed = store.get_extraction_job("job-renew")
    assert renewed is not None
    assert renewed["status"] == "claimed"
    assert renewed["claimed_at"] != "2000-01-01T00:00:00.000"
    assert store.renew_extraction_job_claim("job-renew", "worker-b") is False


def test_reused_session_does_not_cross_project_set_fact_or_batch(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.set_fact(
        subject="shared service", predicate="latest_status", object="first status",
        text="shared service latest_status is first status", scope="project",
        project_id="project-a", session_id="shared-session",
    )
    second = store.set_fact(
        subject="shared service", predicate="latest_status", object="second status",
        text="shared service latest_status is second status", scope="project",
        project_id="project-b", session_id="shared-session",
    )
    assert second["superseded_ids"] == []
    assert {row["object"] for row in store.query_candidates(project_id="project-a")} == {"first status"}
    assert {row["object"] for row in store.query_candidates(project_id="project-b")} == {"second status"}

    for event in (_event("batch-a", project_id="project-a"), _event("batch-b", project_id="project-b")):
        store.record_event(event)
    store.store_candidates_batch([
        (_candidate("batch-candidate-a", "batch-a", "project-a", "batch a"), {"status": "active"}),
        (_candidate("batch-candidate-b", "batch-b", "project-b", "batch b"), {"status": "active"}),
    ])
    assert {row["object"] for row in store.query_candidates(project_id="project-a")} >= {"batch a"}
    assert {row["object"] for row in store.query_candidates(project_id="project-b")} >= {"batch b"}


def test_full_session_identity_is_required_for_set_fact_query_and_search(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.set_fact(
        subject="shared service", predicate="current_mode", object="mode a",
        text="shared service current_mode is mode a", scope="session",
        project_id="project-a", app_id="app-a", session_id="shared-session",
    )
    second = store.set_fact(
        subject="shared service", predicate="current_mode", object="mode b",
        text="shared service current_mode is mode b", scope="session",
        project_id="project-b", app_id="app-b", session_id="shared-session",
    )
    correction = store.set_fact(
        subject="shared service", predicate="current_mode", object="mode a-new",
        text="shared service current_mode is mode a-new", scope="session",
        project_id="project-a", app_id="app-a", session_id="shared-session",
    )
    assert correction["superseded_ids"] == [first["candidate_id"]]
    assert second["superseded_ids"] == []

    for event_id, project_id, app_id in (
        ("session-a", "project-a", "app-a"),
        ("session-b", "project-b", "app-b"),
    ):
        store.record_event({**_event(event_id, project_id=project_id), "app_id": app_id})
    store.store_candidates_batch([
        (_session_candidate("session-candidate-a", "session-a", "project-a", "app-a", "alpha memory"), {"status": "active"}),
        (_session_candidate("session-candidate-b", "session-b", "project-b", "app-b", "beta memory"), {"status": "active"}),
    ])
    assert [row["object"] for row in store.query_candidates(
        scope="session", session_id="shared-session", project_id="project-a", app_id="app-a",
    )] == ["alpha memory", "mode a-new"]
    assert [row["object"] for row in store.search(
        "memory", scope="session", session_id="shared-session", project_id="project-b", app_id="app-b",
    )] == ["beta memory"]
    assert store.query_candidates(
        scope="session", session_id="shared-session", project_id="project-a",
    ) == []
    assert store.search(
        "memory", scope="session", session_id="shared-session", project_id="project-a",
    ) == []


def test_batch_exact_dedup_and_single_value_collapse_are_project_scoped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for event in (_event("batch-exact-a", project_id="project-a"), _event("batch-exact-b", project_id="project-b")):
        store.record_event(event)
    results = store.store_candidates_batch([
        (_candidate("batch-exact-a1", "batch-exact-a", "project-a", "same value"), {"status": "active"}),
        (_candidate("batch-exact-a2", "batch-exact-a", "project-a", "same value"), {"status": "active"}),
        (_candidate("batch-exact-b1", "batch-exact-b", "project-b", "other value"), {"status": "active"}),
    ])
    assert results[1]["action"] == "updated"
    assert results[1]["superseded_ids"] == ["batch-exact-a1"]
    assert {row["object"] for row in store.query_candidates(project_id="project-a")} == {"same value"}

    collapse = store.store_candidates_batch([
        (_candidate("batch-collapse-a", "batch-exact-a", "project-a", "new value"), {"status": "active"}),
        (_candidate("batch-collapse-b", "batch-exact-b", "project-b", "new value"), {"status": "active"}),
    ])
    assert collapse[0]["superseded_ids"] == ["batch-exact-a2"]
    assert collapse[1]["superseded_ids"] == ["batch-exact-b1"]
    assert {row["object"] for row in store.query_candidates(project_id="project-a")} == {"new value"}
    assert {row["object"] for row in store.query_candidates(project_id="project-b")} == {"new value"}


def test_temporal_conflicts_and_migration_are_namespace_scoped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for event_id, project_id in (("conflict-a", "project-a"), ("conflict-b", "project-b")):
        store.record_event(_event(event_id, project_id=project_id))
    store.store_candidates_batch([
        (_fact_candidate("conflict-a-1", "conflict-a", "project-a", "old a"), {"status": "active"}),
        (_fact_candidate("conflict-a-2", "conflict-a", "project-a", "new a"), {"status": "active"}),
        (_fact_candidate("conflict-b-1", "conflict-b", "project-b", "only b"), {"status": "active"}),
    ])

    conflicts = store.temporal_conflicts()
    assert {(group["project_id"], group["object_count"]) for group in conflicts} == {("project-a", 2)}
    preview = store.migrate_temporal_conflicts(dry_run=True)
    assert preview["would_supersede"] == 1
    applied = store.migrate_temporal_conflicts(dry_run=False)
    assert applied["superseded"] == 1
    assert {row["object"] for row in store.query_candidates(project_id="project-b")} == {"only b"}


def test_query_search_and_compaction_keep_project_and_session_boundaries(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for event_id, project_id in (("search-a", "project-a"), ("search-b", "project-b")):
        store.record_event(_event(event_id, project_id=project_id))
    for candidate_id, event_id, project_id, object_ in (
        ("search-candidate-a", "search-a", "project-a", "alpha deployment"),
        ("search-candidate-b", "search-b", "project-b", "beta deployment"),
    ):
        store.store_candidate(_fact_candidate(candidate_id, event_id, project_id, object_), {"status": "active"})

    assert [row["object"] for row in store.search("deployment", project_id="project-a")] == ["alpha deployment"]
    assert [row["object"] for row in store.search("deployment", project_id="project-b")] == ["beta deployment"]
    assert store.query_candidates(session_id="shared-session") == []

    compacted = store.compact_candidates_to_atoms(project_id="project-a", dry_run=False)
    assert compacted["atoms_written"] == 1
    assert {atom["metadata"].get("project_id") for atom in store.query_atoms(active_only=True)} == {"project-a"}


def test_compaction_forget_sources_does_not_cross_projects(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for event_id, project_id in (("compact-a", "project-a"), ("compact-b", "project-b")):
        store.record_event(_event(event_id, project_id=project_id))
    items = []
    for project_id, event_id in (("project-a", "compact-a"), ("project-b", "compact-b")):
        items.extend([
            (_fact_candidate(f"{project_id}-old", event_id, project_id, f"old {project_id}"), {"status": "active"}),
            (_fact_candidate(f"{project_id}-new", event_id, project_id, f"new {project_id}"), {"status": "active"}),
        ])
    store.store_candidates_batch(items)

    result = store.compact_candidates_to_atoms(dry_run=False, forget_sources=True)
    assert result["atoms_written"] == 2
    assert result["sources_forgotten"] == 2
    assert {atom["metadata"].get("project_id") for atom in store.query_atoms(active_only=True)} == {
        "project-a", "project-b",
    }
    active = store.query_candidates(status="active")
    assert len(active) == 2
    assert {row["project_id"] for row in active} == {"project-a", "project-b"}


def test_atom_ids_and_supersession_are_namespace_scoped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.store_atom({
        "type": "decision", "subject": "shared service", "predicate": "decided",
        "object": "use alpha", "scope": "project", "confidence": 0.9,
        "metadata": {"project_id": "project-a"},
    })
    duplicate = store.store_atom({
        "type": "decision", "subject": "shared service", "predicate": "decided",
        "object": "use alpha", "scope": "project", "confidence": 0.9,
        "metadata": {"project_id": "project-a"},
    })
    other_project = store.store_atom({
        "type": "decision", "subject": "shared service", "predicate": "decided",
        "object": "use beta", "scope": "project", "confidence": 0.9,
        "metadata": {"project_id": "project-b"},
    })
    correction = store.store_atom({
        "type": "decision", "subject": "shared service", "predicate": "decided",
        "object": "use gamma", "scope": "project", "confidence": 0.9,
        "metadata": {"project_id": "project-a", "supersedes_prior": True},
    })

    assert duplicate["atom_id"] == first["atom_id"]
    assert store.get_atom(first["atom_id"])["invalid_at"] is not None
    assert store.get_atom(other_project["atom_id"])["invalid_at"] is None
    assert {atom["object"] for atom in store.query_atoms(project_id="project-a")} == {"use gamma"}
    assert {atom["object"] for atom in store.query_atoms(project_id="project-b")} == {"use beta"}
    assert correction["atom_id"] != first["atom_id"]

    session_atom = store.store_atom({
        "type": "fact", "subject": "session", "predicate": "notes",
        "object": "private app memory", "scope": "session", "confidence": 0.9,
        "metadata": {
            "project_id": "project-a",
            "app_id": "app-a",
            "session_id": "shared-session",
        },
    })
    assert store.query_atoms(
        scope="session", project_id="project-a", session_id="shared-session",
    ) == []
    assert [atom["atom_id"] for atom in store.query_atoms(
        scope="session", project_id="project-a", app_id="app-a", session_id="shared-session",
    )] == [session_atom["atom_id"]]


def test_atom_source_event_namespace_is_authoritative(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.record_event(_event("atom-source", project_id="project-a"))

    atom = store.store_atom({
        "source_event_id": "atom-source",
        "type": "fact",
        "subject": "shared service",
        "predicate": "supports",
        "object": "authoritative namespace",
        "scope": "project",
        "confidence": 0.9,
        "metadata": {"project_id": "project-b"},
    })

    assert atom["metadata"]["project_id"] == "project-a"
    assert store.query_atoms(project_id="project-b") == []


def test_secret_redaction_covers_events_jobs_candidates_and_fts(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    secret = "super-secret-value-123"
    github_token = "ghp_" + "A" * 36
    github_pat = "github_pat_" + "B" * 40
    hugging_face_token = "hf_" + "C" * 32
    slack_token = "xoxb-" + "D" * 32
    store.record_event({
        **_event("secret-event", project_id="project-a"),
        "request": {
            "api_key": secret,
            "environment": {
                "GH_TOKEN": github_token,
                "HUGGING_FACE_HUB_TOKEN": hugging_face_token,
            },
            "messages": [{"role": "tool", "content": f"tokens: {github_pat} {slack_token}"}],
        },
        "response_text": f"tool output: {github_token} {hugging_face_token}",
        "metadata": {"authorization": f"Bearer {secret}"},
    })
    store.enqueue_extraction_job(
        {"event_id": "secret-event", "token": secret, "nested": {"password": secret}},
        job_id="secret-job",
        metadata={"secret": secret},
    )
    assert store.complete_extraction_job(
        "secret-job", result={"api_key": secret, "atoms": [{"token": secret}]},
    ) is True
    store.enqueue_extraction_job({"messages": ["safe"]}, job_id="secret-failure-job")
    assert store.fail_extraction_job("secret-failure-job", f"token={secret}") is True
    store.store_candidate(
        {
            **_fact_candidate("secret-candidate", "secret-event", "project-a", f"api_key: {secret}"),
            "text": f"remember api_key: {secret}", "source_quote": f"api_key: {secret}",
            "metadata": {"api_key": secret},
        },
        {"status": "active"},
    )
    store.store_candidates_batch([
        (
            {
                **_fact_candidate("secret-batch", "secret-event", "project-a", f"api_key: {secret}"),
                "metadata": {"authorization": f"Bearer {secret}"},
            },
            {"status": "active"},
        ),
    ])
    store.store_atom({
        "type": "fact", "subject": "secret graph", "predicate": "api_key",
        "object": f"api_key: {secret}", "text": f"api_key: {secret}", "scope": "project",
        "metadata": {"project_id": "project-a", "token": secret},
    })

    assert secret not in repr(store.query_events(project_id="project-a"))
    assert secret not in repr(store.get_extraction_job("secret-job"))
    assert secret not in repr(store.query_candidates(project_id="project-a"))
    assert store.search(secret, project_id="project-a") == []
    with store._connect() as conn:
        table_names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        persisted = " ".join(
            str(value)
            for table_name in table_names
            for row in conn.execute(f"SELECT * FROM [{table_name}]")
            for value in row
        )
    for credential in (secret, github_token, github_pat, hugging_face_token, slack_token):
        assert credential not in persisted


def test_requeue_stale_claimed_extraction_jobs(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    store.enqueue_extraction_job({"messages": ["slow"]}, job_id="job-stale")
    claimed = store.claim_extraction_job("worker-a")
    assert claimed is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE memory_extraction_jobs SET claimed_at = '2000-01-01T00:00:00.000' WHERE job_id = ?",
            ("job-stale",),
        )
        conn.commit()

    recovered = store.requeue_stale_claimed_extraction_jobs(stale_after_seconds=1)

    assert recovered == {"requeued": 1, "failed": 0}
    requeued = store.get_extraction_job("job-stale")
    assert requeued is not None
    assert requeued["status"] == "queued"
    assert requeued["worker_id"] is None
    assert requeued["attempts"] == 1
    assert "stale claim requeued" in requeued["error"]
    reclaimed = store.claim_extraction_job("worker-b")
    assert reclaimed is not None
    assert reclaimed["attempts"] == 2


def test_atom_same_slot_without_supersession_does_not_invalidate_prior_atom(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    store.store_atom(
        {
            "atom_id": "atom-tv-budget-old",
            "type": "constraint",
            "subject": "TV Purchase",
            "predicate": "budget",
            "object": "5000 PLN",
            "scope": "project",
            "valid_at": "2026-01-01T00:00:00.000",
        }
    )
    store.store_atom(
        {
            "atom_id": "atom-tv-budget-new",
            "type": "constraint",
            "subject": "tv purchase",
            "predicate": "budget",
            "object": "6000 PLN",
            "scope": "project",
            "valid_at": "2026-01-02T00:00:00.000",
        }
    )

    atoms = store.query_atoms(type="constraint", predicate="budget", scope="project", active_only=True)
    assert {atom["atom_id"] for atom in atoms} == {"atom-tv-budget-old", "atom-tv-budget-new"}
    assert all(atom["invalid_at"] is None for atom in atoms)
    assert all(atom["expired_at"] is None for atom in atoms)


def test_superseding_atom_closes_prior_conflicting_slot(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    store.store_atom(
        {
            "atom_id": "atom-tv-budget-old",
            "type": "constraint",
            "subject": "Project TV Purchase",
            "predicate": "budget",
            "object": "5000 PLN",
            "scope": "project",
            "valid_at": "2026-01-01T00:00:00.000",
        }
    )
    current = store.store_atom(
        {
            "atom_id": "atom-tv-budget-new",
            "type": "constraint",
            "subject": "tv purchase",
            "predicate": "budget",
            "object": "6000 PLN",
            "scope": "project",
            "valid_at": "2026-01-02T00:00:00.000",
            "metadata": {"from_now_on": True},
        }
    )

    old = store.get_atom("atom-tv-budget-old")
    assert old is not None
    assert old["invalid_at"] == "2026-01-02T00:00:00.000"
    assert old["expired_at"] == "2026-01-02T00:00:00.000"
    assert current["invalid_at"] is None
    assert current["expired_at"] is None

    active_atoms = store.query_atoms(type="constraint", predicate="budget", scope="project", active_only=True)
    assert [atom["atom_id"] for atom in active_atoms] == ["atom-tv-budget-new"]
    assert active_atoms[0]["object"] == "6000 PLN"


def test_atoms_and_aliases_can_be_stored_and_read(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    atom = store.store_atom(
        {
            "atom_id": "atom-tv-budget",
            "type": "constraint",
            "subject": "tv purchase",
            "predicate": "budget",
            "object": "5000 PLN",
            "text": "Budget is 5000 PLN",
            "scope": "project",
            "confidence": 0.92,
            "valid_at": "2026-01-01T00:00:00.000",
            "metadata": {"project_id": "tv-shopping"},
        }
    )
    assert atom["atom_id"] == "atom-tv-budget"
    assert atom["metadata"] == {"project_id": "tv-shopping"}

    atoms = store.query_atoms(type="constraint", subject="tv purchase", scope="project")
    assert len(atoms) == 1
    assert atoms[0]["predicate"] == "budget"
    assert atoms[0]["object"] == "5000 PLN"

    alias = store.store_entity_alias(
        {
            "alias_id": "alias-lg-c4",
            "entity_id": "ent_lg_oled_c4",
            "alias": "LG C4",
            "type": "product",
            "scope": "project",
            "confidence": 0.99,
            "metadata": {"brand": "LG"},
        }
    )
    assert alias["alias_id"] == "alias-lg-c4"
    assert alias["metadata"] == {"brand": "LG"}

    aliases = store.query_entity_aliases(alias="LG C4", type="product", scope="project")
    assert len(aliases) == 1
    assert aliases[0]["entity_id"] == "ent_lg_oled_c4"
