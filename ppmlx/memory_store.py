"""SQLite storage for ppmlx's local temporal memory graph."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from hashlib import sha1
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = {"active"}
REDACTION_MARKER = "[REDACTED]"
_PERSISTED_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\b(?:xox[baprs]-|glpat-)[A-Za-z0-9_-]{10,}\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{10,}", re.IGNORECASE),
)
_PERSISTED_SECRET_FIELDS = re.compile(
    r"^(?:.*[_-])?(?:api[_-]?key|authorization|password|secret|token|credential|private[_-]?key)$",
    re.IGNORECASE,
)
MAX_GRAPH_ENTITY_LABEL_CHARS = 80
MAX_GRAPH_ENTITY_LABEL_WORDS = 12
_SAFE_ENTITY_PREFIXES = (
    "project",
    "repo",
    "repository",
    "app",
    "application",
    "package",
    "module",
    "workspace",
)


def _redact_persisted_value(value: Any, *, field: str | None = None) -> Any:
    """Return a copy that does not contain common secret values."""
    if field and _PERSISTED_SECRET_FIELDS.fullmatch(field):
        return REDACTION_MARKER if value not in (None, "") else value
    if isinstance(value, str):
        redacted = value
        for pattern in _PERSISTED_SECRET_PATTERNS:
            redacted = pattern.sub(REDACTION_MARKER, redacted)
        return redacted
    if isinstance(value, list):
        return [_redact_persisted_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_persisted_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _redact_persisted_value(item, field=str(key))
            for key, item in value.items()
        }
    return value


def contains_persisted_secret(text: str) -> bool:
    """Return True when text contains a credential that must not be stored."""
    return any(pattern.search(text) for pattern in _PERSISTED_SECRET_PATTERNS)


def _namespace_fields(
    scope: str,
    *,
    app_id: str | None,
    project_id: str | None,
    session_id: str | None,
) -> tuple[tuple[str, str | None], ...]:
    """Return the identifiers that define one memory namespace."""
    if scope == "project":
        return (("project_id", project_id),)
    if scope == "app":
        return (("app_id", app_id),)
    if scope == "session":
        return (
            ("session_id", session_id),
            ("project_id", project_id),
            ("app_id", app_id),
        )
    return ()


def _namespace_identity(
    scope: str,
    *,
    app_id: str | None,
    project_id: str | None,
    session_id: str | None,
) -> tuple[str | None, str | None, str | None]:
    values = dict(
        _namespace_fields(
            scope,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
        )
    )
    return values.get("app_id"), values.get("project_id"), values.get("session_id")

# Retrieval ranking: higher is better. workflow_state is intentionally low so
# durable decisions/preferences win over transient task chatter.
_TYPE_RANK_BOOST = {
    "preference": 1.25,
    "constraint": 1.2,
    "decision": 1.15,
    "instruction": 1.1,
    "fact": 1.0,
    "entity_note": 0.95,
    "relationship": 0.9,
    "blocker": 0.85,
    "risk": 0.85,
    "todo": 0.7,
    "workflow_state": 0.45,
}

# Predicates that represent a single current value for subject+scope.
# Newer values should supersede older active ones instead of stacking.
SINGLE_VALUE_PREDICATES = {
    "current_task",
    "next_action",
    "blocker",
    "status",
    "core_status",
    "current_phase",
    "latest_commit",
    "latest_local_commit",
    "latest_pushed_commit",
    "latest pushed commit",
    "pushed_commit",
    "is",
    "equals",
    "uses",
    "use",
    "supports_engine",
    "configuration",
    "decided",
    "decided_as",
    "goal",
    "mode",
    "default_model",
    "release_target",
    "platform_strategy",
    "visual_direction",
}

# workflow_state predicates that remain append-only history.
ADDITIVE_WORKFLOW_PREDICATES = {
    "command_run",
    "file_changed",
    "file_updated",
    "patched_issue",
    "rebuilt_after",
    "updated_from",
    "implemented",
    "validation",
    "commit",
    "commit_pushed",
}


def _default_memory_db_path() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir

        return get_ppmlx_dir() / "memory.db"
    except Exception:
        return Path.home() / ".ppmlx" / "memory.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL UNIQUE,
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    endpoint        TEXT,
    app_id          TEXT,
    project_id      TEXT,
    session_id      TEXT,
    model_alias     TEXT,
    model_repo      TEXT,
    request_json    TEXT,
    response_text   TEXT,
    metadata_json   TEXT
);

CREATE TABLE IF NOT EXISTS memory_candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id    TEXT NOT NULL UNIQUE,
    event_id        TEXT NOT NULL,
    type            TEXT NOT NULL,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    text            TEXT NOT NULL,
    scope           TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0,
    source_quote    TEXT,
    salience        REAL NOT NULL DEFAULT 1,
    status          TEXT NOT NULL,
    reasons_json    TEXT,
    invalidates_json TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    valid_from      TEXT,
    valid_to        TEXT,
    metadata_json   TEXT,
    FOREIGN KEY(event_id) REFERENCES memory_events(event_id)
);

CREATE TABLE IF NOT EXISTS memory_entities (
    entity_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'concept',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    UNIQUE(name, type)
);

CREATE TABLE IF NOT EXISTS memory_edges (
    edge_id             TEXT PRIMARY KEY,
    from_entity_id      TEXT NOT NULL,
    relation            TEXT NOT NULL,
    to_entity_id        TEXT NOT NULL,
    source_candidate_id TEXT NOT NULL,
    confidence          REAL NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'active',
    valid_from          TEXT,
    valid_to            TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    FOREIGN KEY(from_entity_id) REFERENCES memory_entities(entity_id),
    FOREIGN KEY(to_entity_id) REFERENCES memory_entities(entity_id)
);

CREATE TABLE IF NOT EXISTS memory_compactions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    request_id              TEXT,
    endpoint                TEXT,
    app_id                  TEXT,
    project_id              TEXT,
    session_id              TEXT,
    mode                    TEXT NOT NULL,
    original_tokens         INTEGER NOT NULL DEFAULT 0,
    reduced_tokens          INTEGER NOT NULL DEFAULT 0,
    compression_ratio       REAL NOT NULL DEFAULT 0,
    hot_tail_tokens         INTEGER NOT NULL DEFAULT 0,
    session_context_tokens  INTEGER NOT NULL DEFAULT 0,
    cold_messages           INTEGER NOT NULL DEFAULT 0,
    context_items           INTEGER NOT NULL DEFAULT 0,
    compacted               INTEGER NOT NULL DEFAULT 0,
    injected                INTEGER NOT NULL DEFAULT 0,
    latency_ms              REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_extraction_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL UNIQUE,
    source_event_id TEXT,
    status          TEXT NOT NULL DEFAULT 'queued',
    priority        INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    worker_id       TEXT,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    result_json     TEXT,
    error           TEXT,
    valid_at        TEXT,
    invalid_at      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    claimed_at      TEXT,
    completed_at    TEXT,
    failed_at       TEXT,
    expired_at      TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(source_event_id) REFERENCES memory_events(event_id)
);

CREATE TABLE IF NOT EXISTS memory_atoms (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id         TEXT NOT NULL UNIQUE,
    source_event_id TEXT,
    source_job_id   TEXT,
    type            TEXT NOT NULL,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    text            TEXT NOT NULL,
    scope           TEXT NOT NULL DEFAULT 'global',
    confidence      REAL NOT NULL DEFAULT 0,
    valid_at        TEXT,
    invalid_at      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    expired_at      TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(source_event_id) REFERENCES memory_events(event_id),
    FOREIGN KEY(source_job_id) REFERENCES memory_extraction_jobs(job_id)
);

CREATE TABLE IF NOT EXISTS memory_entity_aliases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alias_id        TEXT NOT NULL UNIQUE,
    entity_id       TEXT NOT NULL,
    alias           TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'concept',
    scope           TEXT NOT NULL DEFAULT 'global',
    confidence      REAL NOT NULL DEFAULT 1,
    valid_at        TEXT,
    invalid_at      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    expired_at      TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    UNIQUE(entity_id, alias, type, scope)
);

CREATE INDEX IF NOT EXISTS idx_memory_events_timestamp ON memory_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_events_project ON memory_events(project_id);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_event ON memory_candidates(event_id);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_status ON memory_candidates(status);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_scope ON memory_candidates(scope);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_slot ON memory_candidates(type, subject, predicate, scope, status);
CREATE INDEX IF NOT EXISTS idx_memory_edges_source ON memory_edges(source_candidate_id);
CREATE INDEX IF NOT EXISTS idx_memory_edges_status ON memory_edges(status);
CREATE INDEX IF NOT EXISTS idx_memory_compactions_timestamp ON memory_compactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_compactions_project_session ON memory_compactions(project_id, session_id);
CREATE INDEX IF NOT EXISTS idx_memory_extraction_jobs_status ON memory_extraction_jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_extraction_jobs_source_event ON memory_extraction_jobs(source_event_id);
CREATE INDEX IF NOT EXISTS idx_memory_atoms_slot ON memory_atoms(type, subject, predicate, scope);
CREATE INDEX IF NOT EXISTS idx_memory_atoms_source_event ON memory_atoms(source_event_id);
CREATE INDEX IF NOT EXISTS idx_memory_atoms_valid ON memory_atoms(valid_at, invalid_at, expired_at);
CREATE INDEX IF NOT EXISTS idx_memory_entity_aliases_alias ON memory_entity_aliases(alias, type, scope);
CREATE INDEX IF NOT EXISTS idx_memory_entity_aliases_entity ON memory_entity_aliases(entity_id);

CREATE TABLE IF NOT EXISTS memory_inferred (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    inferred_id     TEXT NOT NULL UNIQUE,
    from_entity_id  TEXT NOT NULL,
    relation        TEXT NOT NULL,
    to_entity_id    TEXT NOT NULL,
    inference_method TEXT NOT NULL,
    source_edge_ids TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active',
    valid_from      TEXT,
    valid_to        TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(from_entity_id) REFERENCES memory_entities(entity_id),
    FOREIGN KEY(to_entity_id) REFERENCES memory_entities(entity_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_inferred_entities ON memory_inferred(from_entity_id, to_entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_inferred_method ON memory_inferred(inference_method);
CREATE INDEX IF NOT EXISTS idx_memory_inferred_status ON memory_inferred(status);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_candidates_fts USING fts5(
    candidate_id UNINDEXED,
    text,
    subject,
    predicate,
    object,
    scope
);
"""


class MemoryStore:
    """Small synchronous SQLite store for temporal-memory events and graph projection."""

    def __init__(self, path: Path | None = None):
        self.path = path or _default_memory_db_path()
        self._lock = threading.Lock()
        self._fts_available: bool | None = None

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            try:
                conn.executescript(_FTS_SCHEMA)
                self._fts_available = True
            except sqlite3.Error:
                self._fts_available = False
            conn.commit()

    def record_event(self, event: dict[str, Any]) -> None:
        self.init()
        with self._lock, self._connect() as conn:
            self._record_event_conn(conn, event)
            conn.commit()

    @staticmethod
    def _record_event_conn(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
        event = _redact_persisted_value(event)
        conn.execute(
            """INSERT OR REPLACE INTO memory_events (
                event_id, endpoint, app_id, project_id, session_id,
                model_alias, model_repo, request_json, response_text, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                event["event_id"],
                event.get("endpoint"),
                event.get("app_id"),
                event.get("project_id"),
                event.get("session_id"),
                event.get("model_alias"),
                event.get("model_repo"),
                json.dumps(event.get("request", {}), ensure_ascii=False),
                event.get("response_text"),
                json.dumps(event.get("metadata", {}), ensure_ascii=False),
            ),
        )

    def enqueue_extraction_job(
        self,
        payload: dict[str, Any],
        *,
        job_id: str | None = None,
        source_event_id: str | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        valid_at: str | None = None,
        expired_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or replace a queued asynchronous memory extraction job."""
        self.init()
        payload = _redact_persisted_value(payload)
        metadata = _redact_persisted_value(metadata or {})
        resolved_job_id = job_id or self._job_id(source_event_id, payload)
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memory_extraction_jobs (
                    job_id, source_event_id, status, priority, attempts, max_attempts,
                    worker_id, payload_json, result_json, error, valid_at, invalid_at,
                    claimed_at, completed_at, failed_at, expired_at, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    resolved_job_id,
                    source_event_id,
                    "queued",
                    int(priority),
                    0,
                    int(max_attempts),
                    None,
                    json.dumps(payload, ensure_ascii=False),
                    None,
                    None,
                    valid_at,
                    None,
                    None,
                    None,
                    None,
                    expired_at,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            conn.commit()
        job = self.get_extraction_job(resolved_job_id)
        if job is None:  # defensive; the insert above should always make this available.
            raise RuntimeError(f"failed to enqueue extraction job {resolved_job_id}")
        return job

    def get_extraction_job(self, job_id: str) -> dict[str, Any] | None:
        self.init()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memory_extraction_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_extraction_job(row) if row else None

    def list_extraction_jobs(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT * FROM memory_extraction_jobs{where}
                    ORDER BY priority DESC, created_at ASC LIMIT ?""",
                params,
            ).fetchall()
        return [self._row_to_extraction_job(row) for row in rows]

    def claim_extraction_job(self, worker_id: str, *, include_expired: bool = False) -> dict[str, Any] | None:
        """Atomically claim the next queued extraction job for a worker."""
        self.init()
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            conditions = ["status = 'queued'", "attempts < max_attempts"]
            if not include_expired:
                conditions.append("(expired_at IS NULL OR expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
            row = conn.execute(
                f"""SELECT * FROM memory_extraction_jobs
                    WHERE {' AND '.join(conditions)}
                    ORDER BY priority DESC, created_at ASC LIMIT 1""",
            ).fetchone()
            if not row:
                conn.commit()
                return None
            conn.execute(
                """UPDATE memory_extraction_jobs
                   SET status = 'claimed', worker_id = ?, attempts = attempts + 1,
                       claimed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'), error = NULL
                   WHERE job_id = ?""",
                (worker_id, row["job_id"]),
            )
            updated = conn.execute(
                "SELECT * FROM memory_extraction_jobs WHERE job_id = ?",
                (row["job_id"],),
            ).fetchone()
            conn.commit()
        return self._row_to_extraction_job(updated) if updated else None

    def requeue_stale_claimed_extraction_jobs(self, *, stale_after_seconds: float) -> dict[str, int]:
        """Recover extraction jobs left claimed by crashed or interrupted workers."""
        self.init()
        try:
            seconds = max(0.0, float(stale_after_seconds))
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds <= 0:
            return {"requeued": 0, "failed": 0}

        # Scale the staleness threshold so a claim that is still being
        # renewed by a live worker (heartbeat interval is ~timeout/3) is
        # never stolen by another worker's sweep.
        effective_seconds = seconds * 2
        modifier = f"-{effective_seconds:.3f} seconds"
        stale_condition = """status = 'claimed'
            AND claimed_at IS NOT NULL
            AND julianday(claimed_at) <= julianday('now', ?)"""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            requeued = conn.execute(
                f"""UPDATE memory_extraction_jobs
                    SET status = 'queued', worker_id = NULL, claimed_at = NULL,
                        error = ?
                    WHERE {stale_condition}
                      AND attempts < max_attempts""",
                (f"stale claim requeued after {effective_seconds:g}s", modifier),
            ).rowcount
            failed = conn.execute(
                f"""UPDATE memory_extraction_jobs
                    SET status = 'failed', error = ?, failed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                        invalid_at = COALESCE(invalid_at, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                    WHERE {stale_condition}
                      AND attempts >= max_attempts""",
                (f"stale claim exceeded max attempts after {effective_seconds:g}s", modifier),
            ).rowcount
            conn.commit()
        return {"requeued": int(requeued), "failed": int(failed)}

    def renew_extraction_job_claim(self, job_id: str, worker_id: str) -> bool:
        """Refresh a claimed extraction job lease while its worker is alive."""
        self.init()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE memory_extraction_jobs
                   SET claimed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                   WHERE job_id = ? AND worker_id = ? AND status = 'claimed'""",
                (job_id, worker_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def complete_extraction_job(self, job_id: str, worker_id: str, *, result: dict[str, Any] | None = None) -> bool:
        """Mark a job completed; only succeeds if ``worker_id`` still owns the claim."""
        self.init()
        result = _redact_persisted_value(result or {})
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE memory_extraction_jobs
                   SET status = 'completed', result_json = ?, error = NULL,
                       completed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                       invalid_at = COALESCE(invalid_at, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                   WHERE job_id = ? AND worker_id = ? AND status = 'claimed'""",
                (json.dumps(result, ensure_ascii=False), job_id, worker_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def fail_extraction_job(self, job_id: str, error: str, *, worker_id: str | None = None, retry: bool = False) -> bool:
        self.init()
        error = str(_redact_persisted_value(error))
        status_expr = "CASE WHEN ? AND attempts < max_attempts THEN 'queued' ELSE 'failed' END"
        owner_condition = "AND worker_id = ?" if worker_id is not None else ""
        # Placeholder order in the statement: retry flag (status), error,
        # retry flag (invalid_at CASE), job_id, then owner.
        params: list[Any] = [1 if retry else 0, error]
        params.extend([1 if retry else 0, job_id])
        if worker_id is not None:
            params.append(worker_id)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"""UPDATE memory_extraction_jobs
                   SET status = {status_expr}, error = ?, failed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                       invalid_at = CASE WHEN ? AND attempts < max_attempts THEN invalid_at
                                         ELSE COALESCE(invalid_at, strftime('%Y-%m-%dT%H:%M:%f', 'now')) END
                   WHERE job_id = ? {owner_condition}""",
                params,
            )
            conn.commit()
        return cur.rowcount > 0

    def store_atom(self, atom: dict[str, Any]) -> dict[str, Any]:
        self.init()
        atom = _redact_persisted_value(atom)
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            atom = dict(atom)
            metadata = dict(atom.get("metadata") or {})
            source_event_id = atom.get("source_event_id")
            if source_event_id:
                source_event = conn.execute(
                    "SELECT app_id, project_id, session_id FROM memory_events WHERE event_id = ?",
                    (source_event_id,),
                ).fetchone()
                if source_event is not None:
                    for field in ("app_id", "project_id", "session_id"):
                        metadata[field] = source_event[field]
            atom["metadata"] = metadata
            atom_id = str(atom.get("atom_id") or self._atom_id(atom))
            conn.execute(
                """INSERT OR REPLACE INTO memory_atoms (
                    atom_id, source_event_id, source_job_id, type, subject, predicate, object,
                    text, scope, confidence, valid_at, invalid_at, expired_at, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    atom_id,
                    atom.get("source_event_id"),
                    atom.get("source_job_id"),
                    atom["type"],
                    atom["subject"],
                    atom["predicate"],
                    atom["object"],
                    atom.get("text") or f"{atom['subject']} {atom['predicate']} {atom['object']}",
                    atom.get("scope", "global"),
                    float(atom.get("confidence", 0.0)),
                    atom.get("valid_at"),
                    atom.get("invalid_at"),
                    atom.get("expired_at"),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            if self._atom_has_supersession_signal(atom):
                self._close_superseded_atom_slots_conn(conn, atom_id=atom_id, atom=atom)
            conn.commit()
        stored = self.get_atom(atom_id)
        if stored is None:
            raise RuntimeError(f"failed to store atom {atom_id}")
        return stored

    def get_atom(self, atom_id: str) -> dict[str, Any] | None:
        self.init()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM memory_atoms WHERE atom_id = ?", (atom_id,)).fetchone()
        return self._row_to_atom(row) if row else None

    def query_atoms(
        self,
        *,
        type: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        scope: str | None = None,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        for column, value in (("type", type), ("subject", subject), ("predicate", predicate), ("scope", scope)):
            if value:
                conditions.append(f"{column} = ?")
                params.append(value)
        if active_only:
            conditions.append("invalid_at IS NULL")
            conditions.append("(expired_at IS NULL OR expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
        namespace_condition, namespace_params = self._atom_namespace_condition(
            scope=scope,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
        )
        if namespace_condition:
            conditions.append(namespace_condition)
            params.extend(namespace_params)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memory_atoms{where} ORDER BY confidence DESC, created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def _close_superseded_atom_slots_conn(
        self,
        conn: sqlite3.Connection,
        *,
        atom_id: str,
        atom: dict[str, Any],
    ) -> None:
        """Close older conflicting active atoms in the same semantic slot.

        Supersession is intentionally opt-in: callers must include an explicit
        correction signal before same-slot atoms with a different object are
        closed. Slot matching uses a canonicalized subject so punctuation/case
        changes do not prevent correction, while preserving exact storage.
        """
        scope = str(atom.get("scope") or "global")
        rows = conn.execute(
            """SELECT atom_id, subject, object, scope, metadata_json FROM memory_atoms
               WHERE type = ? AND predicate = ? AND scope = ?
                 AND atom_id != ? AND invalid_at IS NULL
                 AND (expired_at IS NULL OR expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))""",
            (atom["type"], atom["predicate"], scope, atom_id),
        ).fetchall()
        if not rows:
            return

        canonical_subject = _canonical_atom_subject(atom["subject"])
        object_norm = _norm(str(atom["object"]))
        metadata_value = atom.get("metadata")
        metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
        namespace = _namespace_identity(
            scope,
            app_id=metadata.get("app_id"),
            project_id=metadata.get("project_id"),
            session_id=metadata.get("session_id"),
        )
        superseded_ids = [
            row["atom_id"]
            for row in rows
            if _canonical_atom_subject(row["subject"]) == canonical_subject
            and _norm(str(row["object"])) != object_norm
            and self._atom_row_namespace(row) == namespace
        ]
        if not superseded_ids:
            return

        cutoff = atom.get("valid_at")
        if cutoff is None:
            cutoff = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%f', 'now')").fetchone()[0]
        conn.executemany(
            """UPDATE memory_atoms
               SET invalid_at = COALESCE(invalid_at, ?), expired_at = COALESCE(expired_at, ?)
               WHERE atom_id = ?""",
            [(cutoff, cutoff, superseded_id) for superseded_id in superseded_ids],
        )

    @staticmethod
    def _atom_has_supersession_signal(atom: dict[str, Any]) -> bool:
        metadata = atom.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        for container in (atom, metadata):
            for key in ("supersedes_prior", "from_now_on", "actually", "supersedes", "supersedes_atom_ids"):
                if _truthy_supersession_value(container.get(key)):
                    return True
        return False

    def store_alias(self, alias: dict[str, Any]) -> dict[str, Any]:
        return self.store_entity_alias(alias)

    def store_entity_alias(self, alias: dict[str, Any]) -> dict[str, Any]:
        self.init()
        alias_id = str(alias.get("alias_id") or self._alias_id(alias))
        entity_id = str(alias.get("entity_id") or self._entity_id(alias["alias"], alias.get("type", "concept")))
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memory_entity_aliases (
                    alias_id, entity_id, alias, type, scope, confidence,
                    valid_at, invalid_at, expired_at, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    alias_id,
                    entity_id,
                    alias["alias"],
                    alias.get("type", "concept"),
                    alias.get("scope", "global"),
                    float(alias.get("confidence", 1.0)),
                    alias.get("valid_at"),
                    alias.get("invalid_at"),
                    alias.get("expired_at"),
                    json.dumps(alias.get("metadata", {}), ensure_ascii=False),
                ),
            )
            conn.commit()
        stored = self.get_entity_alias(alias_id)
        if stored is None:
            raise RuntimeError(f"failed to store entity alias {alias_id}")
        return stored

    def get_entity_alias(self, alias_id: str) -> dict[str, Any] | None:
        self.init()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM memory_entity_aliases WHERE alias_id = ?", (alias_id,)).fetchone()
        return self._row_to_entity_alias(row) if row else None

    def query_aliases(
        self,
        *,
        entity_id: str | None = None,
        alias: str | None = None,
        type: str | None = None,
        scope: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.query_entity_aliases(
            entity_id=entity_id,
            alias=alias,
            type=type,
            scope=scope,
            active_only=active_only,
            limit=limit,
        )

    def query_entity_aliases(
        self,
        *,
        entity_id: str | None = None,
        alias: str | None = None,
        type: str | None = None,
        scope: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        for column, value in (("entity_id", entity_id), ("alias", alias), ("type", type), ("scope", scope)):
            if value:
                conditions.append(f"{column} = ?")
                params.append(value)
        if active_only:
            conditions.append("invalid_at IS NULL")
            conditions.append("(expired_at IS NULL OR expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memory_entity_aliases{where} ORDER BY confidence DESC, created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_entity_alias(row) for row in rows]

    def query_events(
        self,
        *,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        since_hours: float | None = None,
        limit: int = 100,
        include_internal: bool = False,
    ) -> list[dict[str, Any]]:
        """Return raw memory events for local trace export.

        Exported events may contain prompts, responses, and tool outputs; this is
        intentionally a local-only API used by the CLI trace exporter.
        """
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        if app_id:
            conditions.append("app_id = ?")
            params.append(app_id)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if since_hours is not None:
            conditions.append("timestamp >= strftime('%Y-%m-%dT%H:%M:%f', 'now', ?)")
            params.append(f"-{since_hours} hours")
        if not include_internal:
            conditions.append("(endpoint IS NULL OR endpoint NOT LIKE '%#compact%')")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memory_events{where} ORDER BY timestamp ASC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def store_candidate(self, candidate: dict[str, Any], validation: dict[str, Any]) -> None:
        self.init()
        with self._lock, self._connect() as conn:
            self._store_candidate_conn(conn, candidate, validation)
            conn.commit()

    def _store_candidate_conn(self, conn: sqlite3.Connection, candidate: dict[str, Any], validation: dict[str, Any]) -> None:
        candidate = _redact_persisted_value(candidate)
        conn.execute(
            """INSERT OR REPLACE INTO memory_candidates (
                candidate_id, event_id, type, subject, predicate, object, text, scope,
                confidence, source_quote, salience, status, reasons_json,
                invalidates_json, valid_from, valid_to, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                candidate["candidate_id"],
                candidate["event_id"],
                candidate["type"],
                candidate["subject"],
                candidate["predicate"],
                candidate["object"],
                candidate["text"],
                candidate["scope"],
                float(candidate.get("confidence", 0.0)),
                candidate.get("source_quote"),
                float(candidate.get("salience", 1.0)),
                validation.get("status", "rejected"),
                json.dumps(validation.get("reasons", []), ensure_ascii=False),
                json.dumps(validation.get("invalidates", []), ensure_ascii=False),
                validation.get("valid_from"),
                validation.get("valid_to"),
                json.dumps(candidate.get("metadata", {}), ensure_ascii=False),
            ),
        )
        self._upsert_fts(conn, candidate)

    def store_candidates_batch(
        self,
        items: list[tuple[dict[str, Any], dict[str, Any]]],
        *,
        event: dict[str, Any] | None = None,
        force: bool = False,
        dedup: bool = True,
    ) -> list[dict[str, Any]]:
        """Store candidates in one transaction, with namespace-scoped SPO dedup."""
        self.init()
        results: list[dict[str, Any]] = []
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if event is not None:
                self._record_event_conn(conn, event)
            for candidate, validation in items:
                action = "added"
                superseded_ids: list[str] = []
                now = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%f', 'now')").fetchone()[0]
                if dedup and not force:
                    event_namespace = event
                    if event_namespace is None or event_namespace.get("event_id") != candidate.get("event_id"):
                        event_row = conn.execute(
                            "SELECT app_id, project_id, session_id FROM memory_events WHERE event_id = ?",
                            (candidate.get("event_id"),),
                        ).fetchone()
                        event_namespace = dict(event_row) if event_row is not None else {}
                    candidate_scope = str(candidate.get("scope") or "global")
                    namespace_condition, namespace_params = self._exact_namespace_condition(
                        scope=candidate_scope,
                        app_id=event_namespace.get("app_id"),
                        project_id=event_namespace.get("project_id"),
                        session_id=event_namespace.get("session_id"),
                    )

                    # 1) Exact candidate dedup in one namespace.
                    exact_conditions = [
                        "c.type = ?",
                        "c.subject = ?",
                        "c.predicate = ?",
                        "c.object = ?",
                        "c.scope = ?",
                        "c.status = 'active'",
                    ]
                    exact_params: list[Any] = [
                        candidate.get("type"),
                        candidate.get("subject"),
                        candidate.get("predicate"),
                        candidate.get("object"),
                        candidate_scope,
                    ]
                    if namespace_condition:
                        exact_conditions.append(namespace_condition)
                        exact_params.extend(namespace_params)
                    existing_rows = conn.execute(
                        f"""SELECT c.* FROM memory_candidates c
                            LEFT JOIN memory_events e ON e.event_id = c.event_id
                            WHERE {' AND '.join(exact_conditions)}
                            ORDER BY c.created_at DESC""",
                        exact_params,
                    ).fetchall()
                    if existing_rows:
                        superseded_ids = [row["candidate_id"] for row in existing_rows]
                        old_confidence_sum = sum(float(row["confidence"] or 0.0) for row in existing_rows)
                        new_confidence = float(candidate.get("confidence", 0.0))
                        candidate["confidence"] = (old_confidence_sum + new_confidence) / (len(existing_rows) + 1)
                        validation = {**validation, "valid_from": validation.get("valid_from") or now}
                        for superseded_id in superseded_ids:
                            self._supersede_candidate_conn(conn, superseded_id, valid_to=now)
                        action = "updated"
                    # 2) Single-value temporal slot collapse (different object).
                    elif _is_single_value_predicate(str(candidate.get("type") or ""), str(candidate.get("predicate") or "")):
                        slot_conditions = [
                            "c.type = ?",
                            "c.subject = ?",
                            "c.predicate = ?",
                            "c.scope = ?",
                            "c.status = 'active'",
                        ]
                        slot_params: list[Any] = [
                            candidate.get("type"),
                            candidate.get("subject"),
                            candidate.get("predicate"),
                            candidate_scope,
                        ]
                        if namespace_condition:
                            slot_conditions.append(namespace_condition)
                            slot_params.extend(namespace_params)
                        slot_rows = conn.execute(
                            f"""SELECT c.* FROM memory_candidates c
                                LEFT JOIN memory_events e ON e.event_id = c.event_id
                                WHERE {' AND '.join(slot_conditions)}
                                ORDER BY c.created_at DESC""",
                            slot_params,
                        ).fetchall()
                        if slot_rows:
                            superseded_ids = [row["candidate_id"] for row in slot_rows]
                            validation = {
                                **validation,
                                "valid_from": validation.get("valid_from") or now,
                                "reasons": list(validation.get("reasons") or []) + ["supersedes_prior"],
                                "invalidates": list(validation.get("invalidates") or []) + superseded_ids,
                            }
                            for superseded_id in superseded_ids:
                                self._supersede_candidate_conn(conn, superseded_id, valid_to=now)
                            action = "updated"
                candidate_for_edge = dict(candidate)
                candidate_for_edge["valid_from"] = validation.get("valid_from")
                candidate_for_edge["valid_to"] = validation.get("valid_to")
                self._store_candidate_conn(conn, candidate, validation)
                self._upsert_memory_edge_conn(conn, candidate_for_edge)
                results.append({
                    "action": "forced" if force else action,
                    "candidate_id": candidate["candidate_id"],
                    "superseded_id": superseded_ids[0] if superseded_ids else None,
                    "superseded_ids": superseded_ids,
                    "type": candidate.get("type"),
                    "subject": candidate.get("subject"),
                    "predicate": candidate.get("predicate"),
                    "object": candidate.get("object"),
                    "scope": candidate.get("scope"),
                    "confidence": candidate.get("confidence"),
                    "valid_from": validation.get("valid_from"),
                    "valid_to": validation.get("valid_to"),
                })
            conn.commit()
        return results

    def dedup_scan(self, *, active_only: bool = True, limit: int = 1000) -> list[dict[str, Any]]:
        """Return candidate groups that share the same subject/predicate/object."""
        self.init()
        where = "WHERE status = 'active'" if active_only else ""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            groups = conn.execute(
                f"""SELECT subject, predicate, object, COUNT(*) AS count
                    FROM memory_candidates
                    {where}
                    GROUP BY subject, predicate, object
                    HAVING COUNT(*) > 1
                    ORDER BY count DESC, subject ASC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for group in groups:
                candidates = conn.execute(
                    """SELECT candidate_id, type, scope, confidence, status, created_at, valid_from, valid_to
                       FROM memory_candidates
                       WHERE subject = ? AND predicate = ? AND object = ?
                       ORDER BY created_at DESC""",
                    (group["subject"], group["predicate"], group["object"]),
                ).fetchall()
                out.append({
                    "subject": group["subject"],
                    "predicate": group["predicate"],
                    "object": group["object"],
                    "count": group["count"],
                    "candidates": [dict(row) for row in candidates],
                })
        return out

    def mark_invalidated(self, candidate_ids: list[str], *, invalidated_by: str) -> None:
        if not candidate_ids:
            return
        self.init()
        with self._lock, self._connect() as conn:
            now = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%f', 'now')").fetchone()[0]
            for candidate_id in candidate_ids:
                self._supersede_candidate_conn(conn, candidate_id, valid_to=now)
            conn.commit()

    @staticmethod
    def _supersede_candidate_conn(conn: sqlite3.Connection, candidate_id: str, *, valid_to: str) -> None:
        conn.execute(
            """UPDATE memory_candidates
               SET status = 'superseded', valid_to = ?
               WHERE candidate_id = ? AND status = 'active'""",
            (valid_to, candidate_id),
        )
        conn.execute(
            """UPDATE memory_edges
               SET status = 'superseded', valid_to = ?
               WHERE source_candidate_id = ? AND status = 'active'""",
            (valid_to, candidate_id),
        )

    def upsert_memory_edge(self, candidate: dict[str, Any]) -> None:
        self.init()
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            self._upsert_memory_edge_conn(conn, candidate)
            conn.commit()

    def _upsert_memory_edge_conn(self, conn: sqlite3.Connection, candidate: dict[str, Any]) -> None:
        candidate = _redact_persisted_value(candidate)
        edge_id = self._edge_id(candidate["candidate_id"], candidate["predicate"])
        namespace = self._candidate_namespace_conn(conn, candidate)
        subject_projection = self._resolve_graph_entity_conn(
            conn,
            raw_name=str(candidate["subject"]),
            candidate=candidate,
            namespace=namespace,
            side="subject",
        )
        object_projection = self._resolve_graph_entity_conn(
            conn,
            raw_name=str(candidate["object"]),
            candidate=candidate,
            namespace=namespace,
            side="object",
        )
        conn.execute("DELETE FROM memory_edges WHERE source_candidate_id = ?", (candidate["candidate_id"],))
        for raw_name, projection in ((candidate["subject"], subject_projection), (candidate["object"], object_projection)):
            if projection is None:
                continue
            self._upsert_entity_conn(conn, projection["entity_id"], projection["name"], "concept")
            self._upsert_canonical_alias_conn(
                conn,
                entity_id=projection["entity_id"],
                raw_name=str(raw_name),
                canonical_name=projection["name"],
                entity_type="concept",
                scope=str(candidate.get("scope") or "global"),
                candidate_id=str(candidate.get("candidate_id") or ""),
            )
        if subject_projection is None or object_projection is None:
            return

        from_entity_id = subject_projection["entity_id"]
        to_entity_id = object_projection["entity_id"]
        conn.execute(
            """INSERT OR REPLACE INTO memory_edges (
                edge_id, from_entity_id, relation, to_entity_id,
                source_candidate_id, confidence, status, valid_from, valid_to
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                edge_id,
                from_entity_id,
                candidate["predicate"],
                to_entity_id,
                candidate["candidate_id"],
                float(candidate.get("confidence", 0.0)),
                "active",
                candidate.get("valid_from"),
                candidate.get("valid_to"),
            ),
        )

    def _candidate_namespace_conn(self, conn: sqlite3.Connection, candidate: dict[str, Any]) -> dict[str, str | None]:
        row = conn.execute(
            "SELECT app_id, project_id, session_id FROM memory_events WHERE event_id = ?",
            (candidate.get("event_id"),),
        ).fetchone()
        if row is None:
            return {"app_id": None, "project_id": None, "session_id": None}
        return {
            "app_id": row["app_id"],
            "project_id": row["project_id"],
            "session_id": row["session_id"],
        }

    def _resolve_graph_entity_conn(
        self,
        conn: sqlite3.Connection,
        *,
        raw_name: str,
        candidate: dict[str, Any],
        namespace: dict[str, str | None],
        side: str,
    ) -> dict[str, str] | None:
        projection = self._project_anchor_projection(raw_name, candidate=candidate, namespace=namespace, side=side)
        if projection is None:
            projection = canonicalize_graph_entity(raw_name)
        if projection is None:
            return None

        # Local partition search first (preserves project/session isolation for
        # namespace-scoped entities).
        scoped_candidates = self._graph_resolution_candidates_conn(conn, namespace)
        match = _select_graph_entity_match(raw_name, projection, scoped_candidates)
        if match is not None:
            return match

        entity_id = self._entity_id(projection, "concept")

        # Global alias fallback: if the projection text appears globally as an
        # alias for an existing entity (from any partition), reuse that entity
        # so cross-project mentions of the same real-world thing merge into one
        # graph node instead of creating duplicate entities.
        global_match = self._resolve_entity_by_alias_conn(conn, raw_name, projection, entity_id)
        if global_match is not None:
            return global_match

        return {"entity_id": entity_id, "name": projection}

    @staticmethod
    def _project_anchor_projection(
        raw_name: str,
        *,
        candidate: dict[str, Any],
        namespace: dict[str, str | None],
        side: str,
    ) -> str | None:
        if side != "subject":
            return None
        project_id = namespace.get("project_id")
        if not project_id:
            return None
        scope = str(candidate.get("scope") or "").lower()
        candidate_type = str(candidate.get("type") or "").lower()
        predicate = str(candidate.get("predicate") or "").lower()
        raw_norm = _norm(raw_name)
        generic_subjects = {
            "session",
            "current session",
            "current task",
            "task",
            "workflow",
            "assistant",
            "agent",
            "quality-bench",
            "shopping_session",
            # First-person and model-reply subjects — when a candidate says
            # "I need to audit" or "The assistant should X", the real-world
            # subject is the project/user, not the pronoun.
            "i",
            "me",
            "we",
            "the assistant",
            "the user",
        }
        if raw_norm in generic_subjects and (scope == "project" or candidate_type in {"workflow_state", "todo", "decision", "instruction", "entity_note"}):
            return canonicalize_graph_entity(project_id)
        if raw_norm == "user" and candidate_type in {"todo", "decision", "workflow_state", "entity_note", "instruction"}:
            return canonicalize_graph_entity(project_id)
        if raw_norm == "session" and predicate in {"validation", "commit", "commit_pushed", "file_changed", "command_run"}:
            return canonicalize_graph_entity(project_id)
        # Short names (2-3 chars) that look like tool/project abbreviations
        # are likely project-scoped entities, not global concepts.
        if len(raw_norm) <= 3 and candidate_type in {"fact", "entity_note", "workflow_state"} and scope == "global":
            return canonicalize_graph_entity(project_id)
        return None

    def _graph_resolution_candidates_conn(
        self,
        conn: sqlite3.Connection,
        namespace: dict[str, str | None],
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["c.status = 'active'"]
        params: list[Any] = []
        # Project is the useful graph partition; session-only is a fallback for
        # unprojected traces. Avoid global fuzzy linking when no namespace exists.
        if namespace.get("project_id"):
            clauses.append("ev.project_id = ?")
            params.append(namespace["project_id"])
        elif namespace.get("session_id"):
            clauses.append("ev.session_id = ?")
            params.append(namespace["session_id"])
        elif namespace.get("app_id"):
            clauses.append("ev.app_id = ?")
            params.append(namespace["app_id"])
        else:
            return []
        where = " AND ".join(clauses)
        rows = conn.execute(
            f"""SELECT DISTINCT ent.entity_id, ent.name
                FROM memory_edges edge
                JOIN memory_candidates c ON c.candidate_id = edge.source_candidate_id
                JOIN memory_events ev ON ev.event_id = c.event_id
                JOIN memory_entities ent ON ent.entity_id IN (edge.from_entity_id, edge.to_entity_id)
                WHERE {where}""",
            params,
        ).fetchall()
        candidates = [{"entity_id": row["entity_id"], "name": row["name"]} for row in rows]
        if not candidates:
            return []
        entity_ids = {candidate["entity_id"] for candidate in candidates}
        placeholders = ",".join("?" for _ in entity_ids)
        alias_rows = conn.execute(
            f"""SELECT entity_id, alias
                FROM memory_entity_aliases
                WHERE invalid_at IS NULL
                  AND (expired_at IS NULL OR expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                  AND entity_id IN ({placeholders})""",
            list(entity_ids),
        ).fetchall()
        aliases_by_entity: dict[str, list[str]] = {}
        for row in alias_rows:
            aliases_by_entity.setdefault(row["entity_id"], []).append(row["alias"])
        for candidate in candidates:
            candidate["aliases"] = aliases_by_entity.get(candidate["entity_id"], [])
        return candidates

    def _resolve_entity_by_alias_conn(
        self,
        conn: sqlite3.Connection,
        raw_name: str,
        projection: str,
        entity_id: str,
    ) -> dict[str, str] | None:
        """Search globally for an existing entity that has *projection* as an alias.

        When two different events mention the same real-world thing using different
        names (e.g. "ppmlx" vs "the ppmlx project"), the deterministic entity_id
        will differ.  This method checks whether *projection* is already an alias
        for an existing entity from any partition so cross-project mentions merge
        into a single graph node.
        """
        # Exact alias match: projection is a registered alias for another entity.
        row = conn.execute(
            """SELECT ea.entity_id, ent.name
               FROM memory_entity_aliases ea
               JOIN memory_entities ent ON ent.entity_id = ea.entity_id
               WHERE ea.alias = ? AND ea.invalid_at IS NULL
                 AND (ea.expired_at IS NULL OR ea.expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                 AND ea.entity_id != ?
               LIMIT 1""",
            (projection, entity_id),
        ).fetchone()
        if row is not None:
            return {"entity_id": row["entity_id"], "name": row["name"]}

        # Approximate match: search all active aliases globally for a fuzzy hit.
        alias_rows = conn.execute(
            """SELECT ea.entity_id, ea.alias, ent.name
               FROM memory_entity_aliases ea
               JOIN memory_entities ent ON ent.entity_id = ea.entity_id
               WHERE ea.invalid_at IS NULL
                 AND (ea.expired_at IS NULL OR ea.expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                 AND ea.entity_id != ?""",
            (entity_id,),
        ).fetchall()
        if not alias_rows:
            return None
        candidates = [
            {"entity_id": row["entity_id"], "name": row["name"], "aliases": [row["alias"]]}
            for row in alias_rows
        ]
        return _select_graph_entity_match(raw_name, projection, candidates)

    def rebuild_graph_projection(
        self,
        *,
        status: str | None = "active",
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Rebuild graph edges for candidates matching exact namespace filters.

        Dry runs are non-destructive and return counts. Confirmed rebuilds delete
        only edges whose source candidates match the supplied filters, then route
        each candidate through ``upsert_memory_edge`` so canonical graph safety
        checks continue to apply.
        """
        self.init()
        status_filter = None if status in {None, "", "all"} else status
        candidates = self._projection_candidates(
            status=status_filter,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
        )
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        existing_edges = self._count_edges_for_candidate_ids(candidate_ids)
        projectable = sum(
            1
            for candidate in candidates
            if canonicalize_graph_entity(str(candidate.get("subject") or "")) is not None
            and canonicalize_graph_entity(str(candidate.get("object") or "")) is not None
        )
        result: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "status": status_filter or "all",
            "app_id": app_id,
            "project_id": project_id,
            "session_id": session_id,
            "candidates": len(candidates),
            "existing_edges": existing_edges,
            "projectable_candidates": projectable,
            "deleted_edges": 0,
            "rebuilt_edges": 0,
        }
        if dry_run or not candidate_ids:
            return result

        with self._lock, self._connect() as conn:
            deleted_edges = self._delete_edges_for_candidate_ids_conn(conn, candidate_ids)
            conn.commit()
        for candidate in candidates:
            self.upsert_memory_edge(candidate)
        result["deleted_edges"] = deleted_edges
        result["rebuilt_edges"] = self._count_edges_for_candidate_ids(candidate_ids)
        return result

    def enqueue_extraction_jobs_from_events(
        self,
        *,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Queue extraction jobs for already-recorded memory events without deleting data."""
        self.init()
        events = self._events_for_extraction_jobs(
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
            limit=limit,
        )
        result: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "app_id": app_id,
            "project_id": project_id,
            "session_id": session_id,
            "limit": limit,
            "events": len(events),
            "queued": 0,
            "job_ids": [],
        }
        if dry_run:
            return result
        for event in events:
            job = self.enqueue_extraction_job(
                event,
                source_event_id=str(event.get("event_id") or ""),
                priority=priority,
                max_attempts=max_attempts,
            )
            result["queued"] += 1
            result["job_ids"].append(job["job_id"])
        return result

    def prune_noisy_namespaces(
        self,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Safely forget active memories from known eval/test namespaces."""
        self.init()
        candidates = self._noisy_namespace_candidates(project_id=project_id, session_id=session_id)
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        edges = self._count_edges_for_candidate_ids(candidate_ids, status="active")
        result: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "project_id": project_id,
            "session_id": session_id,
            "candidates": len(candidates),
            "edges": edges,
            "forgotten_candidates": 0,
            "candidate_ids": candidate_ids,
        }
        if dry_run:
            return result
        for candidate_id in candidate_ids:
            if self.forget_candidate(candidate_id):
                result["forgotten_candidates"] += 1
        return result

    def query_candidates(
        self,
        *,
        status: str | None = "active",
        scope: str | None = None,
        limit: int = 20,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("c.status = ?")
            params.append(status)
        if scope:
            conditions.append("c.scope = ?")
            params.append(scope)
        ns_condition, ns_params = self._namespace_condition(
            scope=scope, app_id=app_id, project_id=project_id, session_id=session_id
        )
        if ns_condition:
            conditions.append(ns_condition)
            params.extend(ns_params)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id, e.endpoint, e.model_alias, e.model_repo
                    FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    {where}
                    ORDER BY c.salience DESC, c.confidence DESC, c.created_at DESC LIMIT ?""",
                params,
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def find_active_slot(
        self,
        *,
        type: str,
        subject: str,
        predicate: str,
        scope: str,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        exact_namespace: bool = False,
    ) -> list[dict[str, Any]]:
        """Find active values in one slot, with optional exact namespace matching.

        The default keeps the old unfiltered result for callers that omit a namespace.
        The validator uses exact matching so a missing identifier only matches an empty namespace.
        An exact session identity includes the session, project, and app identifiers.
        """
        self.init()
        conditions = [
            "c.type = ?",
            "c.subject = ?",
            "c.predicate = ?",
            "c.scope = ?",
            "c.status = 'active'",
        ]
        params: list[Any] = [type, subject, predicate, scope]
        namespace_fields: tuple[tuple[str, str | None], ...] = ()
        if scope == "project":
            namespace_fields = (("project_id", project_id),)
        elif scope == "session":
            namespace_fields = (("session_id", session_id),)
            if exact_namespace:
                namespace_fields += (("project_id", project_id), ("app_id", app_id))
        elif scope == "app":
            namespace_fields = (("app_id", app_id),)

        for namespace_column, namespace_value in namespace_fields:
            if not exact_namespace and namespace_value is None:
                continue
            if namespace_value is None:
                conditions.append(f"e.{namespace_column} IS NULL")
            else:
                conditions.append(f"e.{namespace_column} = ?")
                params.append(namespace_value)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id
                    FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY c.created_at DESC""",
                params,
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def find_active_spo(
        self, *, subject: str, predicate: str, object_: str
    ) -> list[dict[str, Any]]:
        """Find active candidate(s) matching exact SPO."""
        self.init()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM memory_candidates
                   WHERE subject = ? AND predicate = ? AND object = ? AND status = 'active'
                   ORDER BY created_at DESC""",
                (subject, predicate, object_),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        status: str | None = "active",
        scope: str | None = None,
        limit: int = 20,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        exclude_noisy: bool = True,
    ) -> list[dict[str, Any]]:
        """Search candidates with hybrid lexical ranking.

        Ranking blends FTS/LIKE match strength, type prior, recency, confidence,
        salience, and optional project affinity. Returns a ``score`` field on each
        row so callers stop mistaking confidence for relevance.
        """
        self.init()
        terms = _search_terms(query)
        if not terms:
            return []
        # Over-fetch then rerank so durable facts can outrank high-confidence noise.
        fetch_limit = max(limit * 8, 40)
        rows: list[dict[str, Any]] = []
        if self._fts_available is not False:
            try:
                rows = self._search_fts(
                    terms, status=status, scope=scope, limit=fetch_limit,
                    app_id=app_id, project_id=project_id, session_id=session_id,
                )
            except sqlite3.Error:
                self._fts_available = False
                rows = []
        if not rows:
            rows = self._search_like(
                terms, status=status, scope=scope, limit=fetch_limit,
                app_id=app_id, project_id=project_id, session_id=session_id,
            )
        if exclude_noisy and not (app_id or project_id or session_id):
            from ppmlx.context_reducer import is_noisy_context_namespace

            rows = [row for row in rows if not is_noisy_context_namespace(row)]
        dense_vectors = self._load_candidate_embedding_vectors(
            [str(row.get("candidate_id") or "") for row in rows if row.get("candidate_id")]
        )
        ranked = _rank_search_results(
            rows,
            query=query,
            terms=terms,
            project_id=project_id,
            session_id=session_id,
            app_id=app_id,
            dense_vectors=dense_vectors,
        )
        return ranked[: max(1, int(limit))]

    def set_fact(
        self,
        *,
        type: str = "fact",
        subject: str,
        predicate: str,
        object: str,
        text: str,
        scope: str = "project",
        confidence: float = 0.9,
        salience: float = 0.9,
        valid_from: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        app_id: str | None = None,
        source_quote: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Set one active temporal value for subject+predicate+scope.

        Same object => unchanged/deduped. Different object => supersede prior actives.
        """
        self.init()
        subject_n = str(subject).strip()
        predicate_n = str(predicate).strip()
        object_n = str(object).strip()
        text_n = str(text).strip()
        scope_n = str(scope or "project").strip() or "project"
        type_n = str(type or "fact").strip() or "fact"
        if not subject_n or not predicate_n or not object_n or not text_n:
            raise ValueError("set_fact requires subject, predicate, object, and text")

        namespace_identity = _namespace_identity(
            scope_n,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
        )
        namespace_key = json.dumps(namespace_identity, ensure_ascii=False, separators=(",", ":"))
        event_id = f"set-fact-{sha1(f'{type_n}:{subject_n}:{predicate_n}:{scope_n}:{namespace_key}:{object_n}'.encode()).hexdigest()[:12]}"
        candidate_id = f"mem_{sha1(f'{event_id}:{type_n}:{_norm(subject_n)}:{_norm(predicate_n)}:{_norm(object_n)}:{scope_n}'.encode()).hexdigest()[:16]}"
        event = {
            "event_id": event_id,
            "endpoint": "/cli/set-fact",
            "app_id": app_id,
            "project_id": project_id,
            "session_id": session_id,
            "model_alias": "cli",
            "model_repo": "cli",
            "request": {"source": "set_fact"},
            "response_text": "",
            "metadata": {"source": "set_fact"},
        }
        candidate = {
            "candidate_id": candidate_id,
            "event_id": event_id,
            "type": type_n,
            "subject": subject_n,
            "predicate": predicate_n,
            "object": object_n,
            "text": text_n,
            "scope": scope_n,
            "confidence": float(confidence),
            "source_quote": source_quote or text_n,
            "salience": float(salience),
            "metadata": dict(metadata or {}),
        }

        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            self._record_event_conn(conn, event)
            now = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%f', 'now')").fetchone()[0]
            valid_from_n = valid_from or now
            active_conditions = [
                "c.type = ?",
                "c.subject = ?",
                "c.predicate = ?",
                "c.scope = ?",
                "c.status = 'active'",
            ]
            active_params: list[Any] = [type_n, subject_n, predicate_n, scope_n]
            namespace_condition, namespace_params = self._exact_namespace_condition(
                scope=scope_n,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            )
            if namespace_condition:
                active_conditions.append(namespace_condition)
                active_params.extend(namespace_params)
            active_rows = conn.execute(
                f"""SELECT c.* FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    WHERE {' AND '.join(active_conditions)}
                    ORDER BY c.created_at DESC, c.id DESC""",
                active_params,
            ).fetchall()

            same = [row for row in active_rows if _norm(row["object"]) == _norm(object_n)]
            others = [row for row in active_rows if _norm(row["object"]) != _norm(object_n)]

            if same and not others:
                kept = same[0]
                extra = same[1:]
                superseded_ids = [row["candidate_id"] for row in extra]
                for row in extra:
                    self._supersede_candidate_conn(conn, row["candidate_id"], valid_to=now)
                conn.commit()
                return {
                    "action": "unchanged" if not superseded_ids else "deduped",
                    "candidate_id": kept["candidate_id"],
                    "superseded_ids": superseded_ids,
                    "type": type_n,
                    "subject": subject_n,
                    "predicate": predicate_n,
                    "object": object_n,
                    "scope": scope_n,
                }

            superseded_ids = [row["candidate_id"] for row in active_rows]
            for row in active_rows:
                self._supersede_candidate_conn(conn, row["candidate_id"], valid_to=now)
            validation = {
                "status": "active",
                "reasons": ["set_fact"],
                "invalidates": superseded_ids,
                "valid_from": valid_from_n,
                "valid_to": None,
            }
            self._store_candidate_conn(conn, candidate, validation)
            edge_candidate = dict(candidate)
            edge_candidate["valid_from"] = valid_from_n
            edge_candidate["valid_to"] = None
            self._upsert_memory_edge_conn(conn, edge_candidate)
            conn.commit()

        return {
            "action": "updated" if superseded_ids else "added",
            "candidate_id": candidate_id,
            "superseded_ids": superseded_ids,
            "type": type_n,
            "subject": subject_n,
            "predicate": predicate_n,
            "object": object_n,
            "scope": scope_n,
        }

    def fact_history(
        self,
        *,
        subject: str,
        predicate: str,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return temporal history for one subject+predicate slot."""
        self.init()
        conditions = ["subject = ?", "predicate = ?"]
        params: list[Any] = [subject, predicate]
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT * FROM memory_candidates
                   WHERE {' AND '.join(conditions)}
                   ORDER BY COALESCE(valid_from, created_at) DESC, created_at DESC, id DESC
                   LIMIT ?""",
                params,
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def temporal_conflicts(
        self,
        *,
        scope: str | None = None,
        limit: int = 1000,
        type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List active subject+predicate+scope slots with multiple object values."""
        self.init()
        conditions = ["status = 'active'"]
        params: list[Any] = []
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if type:
            conditions.append("type = ?")
            params.append(type)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id
                    FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    WHERE {' AND '.join(f'c.{condition}' for condition in conditions)}
                    ORDER BY COALESCE(c.valid_from, c.created_at) DESC, c.created_at DESC, c.id DESC""",
                params,
            ).fetchall()

        grouped: dict[tuple[Any, ...], list[sqlite3.Row]] = {}
        for row in rows:
            namespace = _namespace_identity(
                str(row["scope"]),
                app_id=row["app_id"],
                project_id=row["project_id"],
                session_id=row["session_id"],
            )
            key = (row["subject"], row["predicate"], row["scope"], row["type"], *namespace)
            grouped.setdefault(key, []).append(row)

        out: list[dict[str, Any]] = []
        for key, candidates in grouped.items():
            objects = list(dict.fromkeys(str(row["object"]) for row in candidates))
            if len(objects) < 2:
                continue
            out.append({
                "subject": key[0],
                "predicate": key[1],
                "scope": key[2],
                "type": key[3],
                "app_id": key[4],
                "project_id": key[5],
                "session_id": key[6],
                "object_count": len(objects),
                "count": len(candidates),
                "objects": objects,
                "candidates": [dict(row) for row in candidates],
            })
        out.sort(key=lambda group: (-group["object_count"], -group["count"], str(group["subject"])))
        return out[:max(1, int(limit))]

    def migrate_temporal_conflicts(
        self,
        *,
        scope: str | None = None,
        limit: int = 1000,
        dry_run: bool = True,
        only_single_value_predicates: bool = False,
    ) -> dict[str, Any]:
        """Keep newest active value per conflict slot; supersede older actives."""
        groups = self.temporal_conflicts(scope=scope, limit=limit)
        if only_single_value_predicates:
            groups = [
                group for group in groups
                if _is_single_value_predicate(str(group.get("type") or ""), str(group.get("predicate") or ""))
            ]
        would_supersede = 0
        plan: list[dict[str, Any]] = []
        for group in groups:
            candidates = list(group.get("candidates") or [])
            if len(candidates) < 2:
                continue
            keep = candidates[0]["candidate_id"]
            drop = [row["candidate_id"] for row in candidates[1:]]
            would_supersede += len(drop)
            plan.append({
                "subject": group["subject"],
                "predicate": group["predicate"],
                "scope": group["scope"],
                "type": group["type"],
                "app_id": group.get("app_id"),
                "project_id": group.get("project_id"),
                "session_id": group.get("session_id"),
                "keep": keep,
                "supersede": drop,
            })
        result: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "conflict_groups": len(plan),
            "would_supersede": would_supersede,
            "superseded": 0,
            "groups": plan,
        }
        if dry_run or not plan:
            return result

        superseded = 0
        with self._lock, self._connect() as conn:
            now = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%f', 'now')").fetchone()[0]
            for group in plan:
                for candidate_id in group["supersede"]:
                    self._supersede_candidate_conn(conn, candidate_id, valid_to=now)
                    superseded += 1
            conn.commit()
        result["superseded"] = superseded
        return result

    def expire_stale_candidates(
        self,
        *,
        older_than_days: float = 30.0,
        types: list[str] | None = None,
        dry_run: bool = True,
        limit: int = 5000,
    ) -> dict[str, Any]:
        """Forget stale active workflow/todo style memories past a TTL."""
        self.init()
        selected_types = types or ["workflow_state", "todo"]
        days = max(0.0, float(older_than_days))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in selected_types)
            rows = conn.execute(
                f"""SELECT candidate_id, type, subject, predicate, object, created_at, valid_from
                   FROM memory_candidates
                   WHERE status = 'active'
                     AND type IN ({placeholders})
                     AND COALESCE(valid_from, created_at) < strftime('%Y-%m-%dT%H:%M:%f', 'now', ?)
                   ORDER BY COALESCE(valid_from, created_at) ASC
                   LIMIT ?""",
                (*selected_types, f"-{days} days", max(1, int(limit))),
            ).fetchall()
        candidate_ids = [row["candidate_id"] for row in rows]
        result = {
            "dry_run": bool(dry_run),
            "older_than_days": days,
            "types": selected_types,
            "candidates": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "forgotten": 0,
        }
        if dry_run:
            return result
        forgotten = 0
        for candidate_id in candidate_ids:
            if self.forget_candidate(candidate_id):
                forgotten += 1
        result["forgotten"] = forgotten
        return result

    def doctor(self) -> dict[str, Any]:
        """Operational health snapshot for memory quality maintenance."""
        self.init()
        stats = self.stats()
        conflicts = self.temporal_conflicts(limit=5000)
        dups = self.dedup_scan(active_only=True, limit=5000)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            active = int(conn.execute("SELECT COUNT(*) FROM memory_candidates WHERE status = 'active'").fetchone()[0])
            workflow_active = int(conn.execute(
                "SELECT COUNT(*) FROM memory_candidates WHERE status = 'active' AND type = 'workflow_state'"
            ).fetchone()[0])
            missing_edge_rows = conn.execute(
                """SELECT c.candidate_id, c.subject, c.object
                   FROM memory_candidates c
                   WHERE c.status = 'active'
                     AND NOT EXISTS (
                        SELECT 1 FROM memory_edges e WHERE e.source_candidate_id = c.candidate_id
                     )"""
            ).fetchall()
            active_no_edge = 0
            for row in missing_edge_rows:
                if canonicalize_graph_entity(str(row["subject"] or "")) is None:
                    continue
                if canonicalize_graph_entity(str(row["object"] or "")) is None:
                    continue
                active_no_edge += 1
            orphan_entities = int(conn.execute(
                """SELECT COUNT(*) FROM memory_entities ent
                   WHERE NOT EXISTS (
                        SELECT 1 FROM memory_edges e
                        WHERE e.from_entity_id = ent.entity_id OR e.to_entity_id = ent.entity_id
                   )
                   AND NOT EXISTS (
                        SELECT 1 FROM memory_inferred i
                        WHERE i.from_entity_id = ent.entity_id OR i.to_entity_id = ent.entity_id
                   )"""
            ).fetchone()[0])
            failed_jobs = int(conn.execute(
                "SELECT COUNT(*) FROM memory_extraction_jobs WHERE status = 'failed'"
            ).fetchone()[0])
            noisy_candidates = self._noisy_namespace_candidates(project_id=None, session_id=None)

        single_value_conflicts = [
            g for g in conflicts
            if _is_single_value_predicate(str(g.get("type") or ""), str(g.get("predicate") or ""))
        ]
        issues: list[str] = []
        if conflicts:
            issues.append(f"temporal_conflicts={len(conflicts)}")
        if single_value_conflicts:
            issues.append(f"single_value_conflicts={len(single_value_conflicts)}")
        if dups:
            issues.append(f"exact_spo_dup_groups={len(dups)}")
        if active and workflow_active / active > 0.25:
            issues.append(f"workflow_state_ratio={workflow_active / active:.2f}")
        if active_no_edge:
            issues.append(f"active_without_edge={active_no_edge}")
        if orphan_entities:
            issues.append(f"orphan_entities={orphan_entities}")
        if failed_jobs:
            issues.append(f"failed_jobs={failed_jobs}")
        if noisy_candidates:
            issues.append(f"noisy_namespace_active={len(noisy_candidates)}")

        return {
            **stats,
            "active": active,
            "workflow_active": workflow_active,
            "workflow_active_ratio": round((workflow_active / active), 4) if active else 0.0,
            "temporal_conflicts": len(conflicts),
            "single_value_conflicts": len(single_value_conflicts),
            "exact_spo_dup_groups": len(dups),
            "active_without_edge": active_no_edge,
            "orphan_entities": orphan_entities,
            "failed_jobs": failed_jobs,
            "noisy_namespace_active": len(noisy_candidates),
            "issues": issues,
            "healthy": not issues,
            "recommendations": _doctor_recommendations(
                conflicts=len(conflicts),
                single_value_conflicts=len(single_value_conflicts),
                dups=len(dups),
                active_no_edge=active_no_edge,
                noisy=len(noisy_candidates),
                failed_jobs=failed_jobs,
            ),
        }

    def inspect_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        self.init()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if not row:
                return None
            candidate = self._row_to_candidate(row)
            edges = conn.execute(
                """SELECT e.*, ef.name as from_name, et.name as to_name
                   FROM memory_edges e
                   JOIN memory_entities ef ON ef.entity_id = e.from_entity_id
                   JOIN memory_entities et ON et.entity_id = e.to_entity_id
                   WHERE e.source_candidate_id = ?""",
                (candidate_id,),
            ).fetchall()
        candidate["edges"] = [dict(edge) for edge in edges]
        return candidate

    def forget_candidate(self, candidate_id: str) -> bool:
        self.init()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE memory_candidates
                   SET status = 'forgotten', valid_to = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                   WHERE candidate_id = ?""",
                (candidate_id,),
            )
            conn.execute(
                """UPDATE memory_edges
                   SET status = 'forgotten', valid_to = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                   WHERE source_candidate_id = ?""",
                (candidate_id,),
            )
            conn.commit()
        return cur.rowcount > 0

    def record_compaction(self, data: dict[str, Any]) -> None:
        """Record privacy-safe local compact-mode observability metrics."""
        self.init()
        original_tokens = int(data.get("original_tokens") or 0)
        reduced_tokens = int(data.get("reduced_tokens") or 0)
        compression_ratio = float(data.get("compression_ratio") or _compression_ratio(original_tokens, reduced_tokens))
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO memory_compactions (
                    request_id, endpoint, app_id, project_id, session_id, mode,
                    original_tokens, reduced_tokens, compression_ratio,
                    hot_tail_tokens, session_context_tokens, cold_messages,
                    context_items, compacted, injected, latency_ms
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data.get("request_id"),
                    data.get("endpoint"),
                    data.get("app_id"),
                    data.get("project_id"),
                    data.get("session_id"),
                    str(data.get("mode") or "compact"),
                    original_tokens,
                    reduced_tokens,
                    compression_ratio,
                    int(data.get("hot_tail_tokens") or 0),
                    int(data.get("session_context_tokens") or 0),
                    int(data.get("cold_messages") or 0),
                    int(data.get("context_items") or 0),
                    1 if data.get("compacted") else 0,
                    1 if data.get("injected") else 0,
                    float(data.get("latency_ms") or 0.0),
                ),
            )
            conn.commit()

    def compact_stats(
        self,
        *,
        since_hours: float | None = 24,
        project_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        if since_hours is not None:
            conditions.append("timestamp >= strftime('%Y-%m-%dT%H:%M:%f', 'now', ?)")
            params.append(f"-{since_hours} hours")
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memory_compactions{where} ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            aggregate = conn.execute(
                f"""SELECT COUNT(*) as total,
                           SUM(compacted) as compacted,
                           SUM(injected) as injected,
                           AVG(original_tokens) as avg_original_tokens,
                           AVG(reduced_tokens) as avg_reduced_tokens,
                           AVG(compression_ratio) as avg_compression_ratio,
                           AVG(cold_messages) as avg_cold_messages,
                           AVG(context_items) as avg_context_items,
                           AVG(latency_ms) as avg_latency_ms,
                           MAX(original_tokens) as max_original_tokens,
                           MAX(reduced_tokens) as max_reduced_tokens
                    FROM memory_compactions{where}""",
                params,
            ).fetchone()
            latency_rows = conn.execute(
                f"SELECT latency_ms FROM memory_compactions{where} ORDER BY latency_ms",
                params,
            ).fetchall()
        latencies = [float(row[0]) for row in latency_rows]
        total = int(aggregate["total"] or 0)
        recent = [dict(row) for row in rows]
        return {
            "path": str(self.path),
            "since_hours": since_hours,
            "project_id": project_id,
            "session_id": session_id,
            "total": total,
            "compacted": int(aggregate["compacted"] or 0),
            "injected": int(aggregate["injected"] or 0),
            "avg_original_tokens": _round_or_none(aggregate["avg_original_tokens"]),
            "avg_reduced_tokens": _round_or_none(aggregate["avg_reduced_tokens"]),
            "avg_compression_ratio": _round_or_none(aggregate["avg_compression_ratio"]),
            "avg_cold_messages": _round_or_none(aggregate["avg_cold_messages"]),
            "avg_context_items": _round_or_none(aggregate["avg_context_items"]),
            "avg_latency_ms": _round_or_none(aggregate["avg_latency_ms"]),
            "p95_latency_ms": _round_or_none(_percentile(latencies, 95)),
            "max_original_tokens": int(aggregate["max_original_tokens"] or 0),
            "max_reduced_tokens": int(aggregate["max_reduced_tokens"] or 0),
            "recent": recent,
        }

    def graph_snapshot(
        self,
        *,
        status: str | None = "active",
        query: str | None = None,
        app_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        limit: int = 120,
    ) -> dict[str, Any]:
        """Return a graph snapshot of memory entities, edges, facts, and events."""
        self.init()
        safe_limit = max(1, min(int(limit), 500))
        status_filter = None if status in {None, "", "all"} else status
        if query and query.strip():
            candidates = self.search(
                query,
                status=status_filter,
                limit=safe_limit,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            )
        else:
            candidates = self.query_candidates(
                status=status_filter,
                limit=safe_limit,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            )
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        edges = self._graph_edges(candidate_ids, status=status_filter)
        nodes = self._graph_nodes(candidates, edges)
        events = self._graph_events(app_id=app_id, project_id=project_id, session_id=session_id, limit=min(safe_limit, 100))
        return {
            "path": str(self.path),
            "filters": {
                "status": status_filter or "all",
                "query": query or "",
                "app_id": app_id,
                "project_id": project_id,
                "session_id": session_id,
                "limit": safe_limit,
            },
            "stats": self.stats(),
            "nodes": nodes,
            "edges": edges,
            "candidates": candidates,
            "events": events,
        }

    def _graph_edges(self, candidate_ids: list[str], *, status: str | None) -> list[dict[str, Any]]:
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        conditions = [f"me.source_candidate_id IN ({placeholders})"]
        params: list[Any] = list(candidate_ids)
        if status:
            conditions.append("me.status = ?")
            params.append(status)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT me.*, ef.name AS from_name, et.name AS to_name,
                           c.type AS candidate_type, c.text AS candidate_text,
                           c.source_quote, c.salience, ev.app_id, ev.project_id, ev.session_id
                    FROM memory_edges me
                    JOIN memory_entities ef ON ef.entity_id = me.from_entity_id
                    JOIN memory_entities et ON et.entity_id = me.to_entity_id
                    LEFT JOIN memory_candidates c ON c.candidate_id = me.source_candidate_id
                    LEFT JOIN memory_events ev ON ev.event_id = c.event_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY me.created_at DESC""",
                params,
            ).fetchall()
        enriched_edges = []
        for row in rows:
            edge = dict(row)
            # Graph-friendly aliases; preserve existing edge_id/from_entity_id/to_entity_id/relation fields.
            edge["id"] = edge.get("edge_id")
            edge["source"] = edge.get("from_entity_id")
            edge["target"] = edge.get("to_entity_id")
            edge["label"] = edge.get("relation")
            enriched_edges.append(edge)
        return enriched_edges

    def _graph_nodes(self, candidates: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}

        def add(name: str, *, role: str, candidate: dict[str, Any] | None = None) -> None:
            cleaned = canonicalize_graph_entity(name)
            if not cleaned:
                return
            entity_id = self._entity_id(cleaned, "concept")
            node = nodes.setdefault(entity_id, {
                "id": entity_id,
                "label": cleaned,
                "name": cleaned,
                "type": "concept",
                "roles": [],
                "candidate_count": 0,
                "salience": 0.0,
                "degree": 0,
                "size": 20,
            })
            if role not in node["roles"]:
                node["roles"].append(role)
            if candidate:
                node["candidate_count"] += 1
                try:
                    node["salience"] = max(float(node.get("salience") or 0.0), float(candidate.get("salience") or 0.0))
                except (TypeError, ValueError):
                    pass

        edge_candidate_ids = {str(edge.get("source_candidate_id") or "") for edge in edges}
        for candidate in candidates:
            # When an edge projection exists, use the resolved graph entity names
            # from that edge instead of re-introducing raw extracted labels as
            # disconnected nodes in the snapshot.
            if str(candidate.get("candidate_id") or "") in edge_candidate_ids:
                continue
            add(str(candidate.get("subject") or ""), role="subject", candidate=candidate)
            add(str(candidate.get("object") or ""), role="object", candidate=candidate)
        for edge in edges:
            edge_candidate = {"salience": edge.get("salience")}
            add(str(edge.get("from_name") or ""), role="edge_from", candidate=edge_candidate)
            add(str(edge.get("to_name") or ""), role="edge_to", candidate=edge_candidate)

        for edge in edges:
            for node_id in (edge.get("source") or edge.get("from_entity_id"), edge.get("target") or edge.get("to_entity_id")):
                if node_id in nodes:
                    nodes[node_id]["degree"] += 1

        for node in nodes.values():
            salience = float(node.get("salience") or 0.0)
            degree = int(node.get("degree") or 0)
            candidate_count = int(node.get("candidate_count") or 0)
            node["size"] = min(48, max(16, round(20 + degree * 4 + candidate_count * 3 + salience * 6, 2)))

        return sorted(nodes.values(), key=lambda node: (-int(node.get("candidate_count") or 0), str(node.get("name") or "").lower()))

    def _graph_events(
        self,
        *,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if app_id:
            conditions.append("app_id = ?")
            params.append(app_id)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT event_id, timestamp, endpoint, app_id, project_id, session_id, model_alias, model_repo
                    FROM memory_events{where}
                    ORDER BY timestamp DESC LIMIT ?""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def _projection_candidates(
        self,
        *,
        status: str | None,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("c.status = ?")
            params.append(status)
        if app_id:
            conditions.append("e.app_id = ?")
            params.append(app_id)
        if project_id:
            conditions.append("e.project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("e.session_id = ?")
            params.append(session_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id, e.endpoint, e.model_alias, e.model_repo
                    FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    {where}
                    ORDER BY c.created_at ASC, c.id ASC""",
                params,
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def _events_for_extraction_jobs(
        self,
        *,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if app_id:
            conditions.append("app_id = ?")
            params.append(app_id)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memory_events{where} ORDER BY timestamp ASC, id ASC{limit_sql}",
                params,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _noisy_namespace_candidates(
        self,
        *,
        project_id: str | None,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        from ppmlx.context_reducer import is_noisy_context_namespace

        conditions = ["c.status = 'active'"]
        params: list[Any] = []
        if project_id:
            conditions.append("e.project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("e.session_id = ?")
            params.append(session_id)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id, e.endpoint, e.model_alias, e.model_repo
                    FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY c.created_at ASC, c.id ASC""",
                params,
            ).fetchall()
        candidates = [self._row_to_candidate(row) for row in rows]
        return [candidate for candidate in candidates if is_noisy_context_namespace(candidate)]

    def _count_edges_for_candidate_ids(self, candidate_ids: list[str], *, status: str | None = None) -> int:
        if not candidate_ids:
            return 0
        placeholders = ",".join("?" for _ in candidate_ids)
        conditions = [f"source_candidate_id IN ({placeholders})"]
        params: list[Any] = list(candidate_ids)
        if status:
            conditions.append("status = ?")
            params.append(status)
        with self._connect() as conn:
            return int(conn.execute(
                f"SELECT COUNT(*) FROM memory_edges WHERE {' AND '.join(conditions)}",
                params,
            ).fetchone()[0])

    @staticmethod
    def _delete_edges_for_candidate_ids_conn(conn: sqlite3.Connection, candidate_ids: list[str]) -> int:
        if not candidate_ids:
            return 0
        placeholders = ",".join("?" for _ in candidate_ids)
        cur = conn.execute(
            f"DELETE FROM memory_edges WHERE source_candidate_id IN ({placeholders})",
            candidate_ids,
        )
        return int(cur.rowcount or 0)

    def stats(self) -> dict[str, Any]:
        self.init()
        with self._connect() as conn:
            events = conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
            candidates = conn.execute("SELECT COUNT(*) FROM memory_candidates").fetchone()[0]
            by_status = conn.execute(
                "SELECT status, COUNT(*) FROM memory_candidates GROUP BY status ORDER BY status"
            ).fetchall()
            edges = conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
            entities = conn.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0]
            atoms = conn.execute("SELECT COUNT(*) FROM memory_atoms").fetchone()[0]
            compactions = conn.execute("SELECT COUNT(*) FROM memory_compactions").fetchone()[0]
            extraction_jobs = conn.execute("SELECT COUNT(*) FROM memory_extraction_jobs").fetchone()[0]
            jobs_by_status = conn.execute(
                "SELECT status, COUNT(*) FROM memory_extraction_jobs GROUP BY status ORDER BY status"
            ).fetchall()
            inferred = conn.execute("SELECT COUNT(*) FROM memory_inferred").fetchone()[0]
            inferred_by_method = conn.execute(
                "SELECT inference_method, COUNT(*) FROM memory_inferred GROUP BY inference_method"
            ).fetchall()
        return {
            "path": str(self.path),
            "events": events,
            "candidates": candidates,
            "entities": entities,
            "edges": edges,
            "atoms": atoms,
            "compactions": compactions,
            "extraction_jobs": extraction_jobs,
            "inferred": inferred,
            "by_status": {row[0]: row[1] for row in by_status},
            "jobs_by_status": {row[0]: row[1] for row in jobs_by_status},
            "inferred_by_method": {row[0]: row[1] for row in inferred_by_method},
        }

    # ------------------------------------------------------------------
    # Graph inference (L3): deterministic rule-based edge inference that
    # connects isolated triples into a richer graph without model calls.
    # ------------------------------------------------------------------

    def run_inference(self, *, scope: str | None = None) -> dict[str, int]:
        """Run all deterministic graph inference rules.

        Returns counts of inferred edges by method.  Safe to call repeatedly;
        each run upserts inferred edges so deduplication is automatic.
        """
        self.init()
        result: dict[str, int] = {}
        with self._lock, self._connect() as conn:
            result["transitive"] = self._infer_transitive_edges_conn(conn, scope=scope)
            result["cooccurrence"] = self._infer_cooccurrence_edges_conn(conn, scope=scope)
            result["temporal"] = self._infer_temporal_chains_conn(conn, scope=scope)
            conn.commit()
        return result

    def query_inferred(
        self,
        *,
        status: str | None = "active",
        method: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return inferred edges with resolved entity names."""
        self.init()
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("inf.status = ?")
            params.append(status)
        if method:
            conditions.append("inf.inference_method = ?")
            params.append(method)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT inf.*, ef.name AS from_name, et.name AS to_name
                    FROM memory_inferred inf
                    JOIN memory_entities ef ON ef.entity_id = inf.from_entity_id
                    JOIN memory_entities et ON et.entity_id = inf.to_entity_id
                    {where}
                    ORDER BY inf.confidence DESC, inf.created_at DESC
                    LIMIT ?""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_walk(
        self,
        entity_name: str,
        *,
        max_hops: int = 3,
        include_inferred: bool = True,
    ) -> dict[str, Any]:
        """Multi-hop graph traversal from an entity using recursive CTE.

        Returns all entities reachable within ``max_hops`` edges, with paths,
        depths, and confidence decay.  Handles cycles via path tracking.
        """
        self.init()
        max_hops = max(1, min(int(max_hops), 5))  # cap at 5 hops for safety

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            # Resolve starting entity
            root = conn.execute(
                "SELECT entity_id, name FROM memory_entities WHERE name = ? LIMIT 1",
                (canonicalize_entity_name(entity_name) or entity_name,),
            ).fetchone()
            if not root:
                return {"entity": entity_name, "found": False, "nodes": [], "edges": []}

            # Build the edge set: direct edges + optionally inferred edges
            # We use a UNION to treat them as one traversable graph.
            edge_sql = """
                SELECT from_entity_id, relation, to_entity_id, confidence, 'direct' AS kind
                FROM memory_edges WHERE status = 'active'
            """
            if include_inferred:
                edge_sql += """
                UNION ALL
                SELECT from_entity_id, relation, to_entity_id, confidence, 'inferred' AS kind
                FROM memory_inferred WHERE status = 'active'
                """

            # Recursive CTE: walk the graph up to max_hops
            rows = conn.execute(f"""
                WITH RECURSIVE walk AS (
                    -- Seed: all edges from the root entity
                    SELECT
                        e.from_entity_id,
                        e.relation,
                        e.to_entity_id,
                        e.confidence,
                        e.kind,
                        1 AS depth,
                        json_array(e.from_entity_id, e.to_entity_id) AS path
                    FROM ({edge_sql}) e
                    WHERE e.from_entity_id = ?

                    UNION ALL

                    -- Recursive: follow edges from reached entities
                    SELECT
                        e.from_entity_id,
                        e.relation,
                        e.to_entity_id,
                        ROUND(w.confidence * e.confidence, 4) AS confidence,
                        e.kind,
                        w.depth + 1,
                        json_insert(w.path, '$[#]', e.to_entity_id)
                    FROM ({edge_sql}) e
                    JOIN walk w ON e.from_entity_id = w.to_entity_id
                    WHERE w.depth < ?
                      -- Cycle prevention: don't follow back to entities already in path
                      AND e.to_entity_id NOT IN (
                          SELECT json_extract(w.path, '$[' || idx || ']')
                          FROM (SELECT 0 AS idx UNION ALL SELECT 1 UNION ALL SELECT 2
                                UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5)
                          WHERE idx < json_array_length(w.path)
                      )
                )
                SELECT DISTINCT
                    ef.name AS from_name,
                    w.relation,
                    et.name AS to_name,
                    w.depth,
                    w.confidence,
                    w.kind,
                    w.path
                FROM walk w
                JOIN memory_entities ef ON ef.entity_id = w.from_entity_id
                JOIN memory_entities et ON et.entity_id = w.to_entity_id
                ORDER BY w.depth, w.confidence DESC
            """, (root["entity_id"], max_hops)).fetchall()

            # Build response
            nodes_set: set[str] = {root["name"]}
            edges_out: list[dict[str, Any]] = []
            for row in rows:
                nodes_set.add(row["from_name"])
                nodes_set.add(row["to_name"])
                edges_out.append({
                    "from": row["from_name"],
                    "relation": row["relation"],
                    "to": row["to_name"],
                    "depth": row["depth"],
                    "confidence": round(row["confidence"], 4),
                    "kind": row["kind"],
                })

            return {
                "entity": root["name"],
                "found": True,
                "max_hops": max_hops,
                "node_count": len(nodes_set),
                "edge_count": len(edges_out),
                "nodes": sorted(nodes_set),
                "edges": edges_out,
            }

    def _infer_transitive_edges_conn(
        self, conn: sqlite3.Connection, *, scope: str | None = None
    ) -> int:
        """A → R1 → B  +  B → R2 → C  ⇒  A → (via B) → C.

        Combines two active edges that share a middle entity.  Confidence is
        min(c1, c2) × 0.8 so inferred edges never outrank direct ones.
        """
        conn.row_factory = sqlite3.Row
        conditions = ["e1.status = 'active'", "e2.status = 'active'"]
        params: list[Any] = []
        if scope:
            conditions.append("c1.scope = ? AND c2.scope = ?")
            params.extend([scope, scope])
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"""SELECT e1.from_entity_id, e1.relation AS rel1, e1.to_entity_id AS mid_entity,
                       e2.relation AS rel2, e2.to_entity_id AS to_entity,
                       MIN(e1.confidence, e2.confidence) * 0.8 AS conf,
                       e1.edge_id AS src1, e2.edge_id AS src2
                FROM memory_edges e1
                JOIN memory_edges e2 ON e2.from_entity_id = e1.to_entity_id
                JOIN memory_candidates c1 ON c1.candidate_id = e1.source_candidate_id
                JOIN memory_candidates c2 ON c2.candidate_id = e2.source_candidate_id
                WHERE {where}
                  AND e1.from_entity_id != e2.to_entity_id""",
            params,
        ).fetchall()
        count = 0
        for row in rows:
            relation = f"{row['rel1']} › {row['rel2']}"
            inferred_id = self._inferred_id(row["from_entity_id"], relation, row["to_entity"], "transitive")
            source_ids = json.dumps([row["src1"], row["src2"]], ensure_ascii=False)
            conn.execute(
                """INSERT OR REPLACE INTO memory_inferred (
                    inferred_id, from_entity_id, relation, to_entity_id,
                    inference_method, source_edge_ids, confidence
                ) VALUES (?,?,?,?,?,?,?)""",
                (inferred_id, row["from_entity_id"], relation, row["to_entity"],
                 "transitive", source_ids, round(row["conf"], 4)),
            )
            count += 1
        return count

    def _infer_cooccurrence_edges_conn(
        self, conn: sqlite3.Connection, *, scope: str | None = None
    ) -> int:
        """Entity pairs appearing in ≥3 co-scoped candidates get a co_occurs_with edge."""
        conn.row_factory = sqlite3.Row
        conditions = ["c.status = 'active'"]
        params: list[Any] = []
        if scope:
            conditions.append("c.scope = ?")
            params.append(scope)
        where = " AND ".join(conditions)
        # Build co-occurrence from edges sharing the same scope: entities that
        # appear as from_entity in edges from the same scope's candidates.
        rows = conn.execute(
            f"""SELECT e1.from_entity_id AS ent_a, e2.from_entity_id AS ent_b,
                       COUNT(DISTINCT c.candidate_id) AS cnt
                FROM memory_candidates c
                JOIN memory_edges e1 ON e1.source_candidate_id = c.candidate_id
                JOIN memory_edges e2 ON e2.source_candidate_id = c.candidate_id
                WHERE {where}
                  AND e1.from_entity_id < e2.from_entity_id
                GROUP BY 1, 2
                HAVING cnt >= 3""",
            params,
        ).fetchall()
        count = 0
        for row in rows:
            inferred_id = self._inferred_id(row["ent_a"], "co_occurs_with", row["ent_b"], "cooccurrence")
            confidence = min(1.0, round(row["cnt"] * 0.15, 4))
            conn.execute(
                """INSERT OR REPLACE INTO memory_inferred (
                    inferred_id, from_entity_id, relation, to_entity_id,
                    inference_method, source_edge_ids, confidence
                ) VALUES (?,?,?,?,?,?,?)""",
                (inferred_id, row["ent_a"], "co_occurs_with", row["ent_b"],
                 "cooccurrence", "[]", confidence),
            )
            count += 1
        return count

    def _infer_temporal_chains_conn(
        self, conn: sqlite3.Connection, *, scope: str | None = None
    ) -> int:
        """Link sequential candidates in the same session with _precedes_ edges.

        Candidates from the same session ordered by created_at form a temporal
        DAG.  We project each candidate's from_entity as a proxy and chain them.
        """
        conn.row_factory = sqlite3.Row
        conditions = ["c1.status = 'active'", "c2.status = 'active'",
                      "ev1.session_id IS NOT NULL", "ev1.session_id = ev2.session_id"]
        params: list[Any] = []
        if scope:
            conditions.append("c1.scope = ? AND c2.scope = ?")
            params.extend([scope, scope])
        where = " AND ".join(conditions)
        # Self-join candidates on same session; pick the immediate predecessor.
        rows = conn.execute(
            f"""SELECT e1.from_entity_id AS from_ent, e2.from_entity_id AS to_ent,
                       c1.candidate_id AS prev_cid, c2.candidate_id AS curr_cid
                FROM memory_candidates c1
                JOIN memory_candidates c2 ON c2.created_at > c1.created_at
                JOIN memory_events ev1 ON ev1.event_id = c1.event_id
                JOIN memory_events ev2 ON ev2.event_id = c2.event_id
                JOIN memory_edges e1 ON e1.source_candidate_id = c1.candidate_id
                JOIN memory_edges e2 ON e2.source_candidate_id = c2.candidate_id
                WHERE {where}
                  AND e1.from_entity_id != e2.from_entity_id
                  AND NOT EXISTS (
                    SELECT 1 FROM memory_candidates c3
                    JOIN memory_events ev3 ON ev3.event_id = c3.event_id
                    WHERE ev3.session_id = ev1.session_id
                      AND c3.created_at > c1.created_at
                      AND c3.created_at < c2.created_at
                  )
                ORDER BY c1.created_at
                LIMIT 500""",
            params,
        ).fetchall()
        count = 0
        for row in rows:
            inferred_id = self._inferred_id(row["from_ent"], "precedes", row["to_ent"], "temporal")
            source_ids = json.dumps([row["prev_cid"], row["curr_cid"]], ensure_ascii=False)
            conn.execute(
                """INSERT OR REPLACE INTO memory_inferred (
                    inferred_id, from_entity_id, relation, to_entity_id,
                    inference_method, source_edge_ids, confidence
                ) VALUES (?,?,?,?,?,?,?)""",
                (inferred_id, row["from_ent"], "precedes", row["to_ent"],
                 "temporal", source_ids, 0.7),
            )
            count += 1
        return count

    @staticmethod
    def _inferred_id(from_entity_id: str, relation: str, to_entity_id: str, method: str) -> str:
        digest = sha1(f"{from_entity_id}:{_norm(relation)}:{to_entity_id}:{method}".encode()).hexdigest()[:16]
        return f"inf_{digest}"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def _upsert_fts(self, conn: sqlite3.Connection, candidate: dict[str, Any]) -> None:
        if self._fts_available is False:
            return
        try:
            conn.execute("DELETE FROM memory_candidates_fts WHERE candidate_id = ?", (candidate["candidate_id"],))
            conn.execute(
                """INSERT INTO memory_candidates_fts (candidate_id, text, subject, predicate, object, scope)
                   VALUES (?,?,?,?,?,?)""",
                (
                    candidate["candidate_id"],
                    candidate["text"],
                    candidate["subject"],
                    candidate["predicate"],
                    candidate["object"],
                    candidate["scope"],
                ),
            )
            self._fts_available = True
        except sqlite3.Error:
            self._fts_available = False

    def _search_fts(
        self,
        terms: list[str],
        *,
        status: str | None,
        scope: str | None,
        limit: int,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        match = " OR ".join(terms)
        conditions = ["memory_candidates_fts MATCH ?"]
        params: list[Any] = [match]
        if status:
            conditions.append("c.status = ?")
            params.append(status)
        if scope:
            conditions.append("c.scope = ?")
            params.append(scope)
        ns_condition, ns_params = self._namespace_condition(
            scope=scope, app_id=app_id, project_id=project_id, session_id=session_id
        )
        if ns_condition:
            conditions.append(ns_condition)
            params.extend(ns_params)
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id, e.endpoint, e.model_alias, e.model_repo
                    FROM memory_candidates_fts
                    JOIN memory_candidates c ON c.candidate_id = memory_candidates_fts.candidate_id
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY c.salience DESC, c.confidence DESC, c.created_at DESC
                    LIMIT ?""",
                params,
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def _search_like(
        self,
        terms: list[str],
        *,
        status: str | None,
        scope: str | None,
        limit: int,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        term_conditions = []
        for term in terms:
            like = f"%{term}%"
            term_conditions.append("(c.text LIKE ? OR c.subject LIKE ? OR c.predicate LIKE ? OR c.object LIKE ?)")
            params.extend([like, like, like, like])
        conditions.append("(" + " OR ".join(term_conditions) + ")")
        if status:
            conditions.append("c.status = ?")
            params.append(status)
        if scope:
            conditions.append("c.scope = ?")
            params.append(scope)
        ns_condition, ns_params = self._namespace_condition(
            scope=scope, app_id=app_id, project_id=project_id, session_id=session_id
        )
        if ns_condition:
            conditions.append(ns_condition)
            params.extend(ns_params)
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT c.*, e.app_id, e.project_id, e.session_id, e.endpoint, e.model_alias, e.model_repo
                    FROM memory_candidates c
                    LEFT JOIN memory_events e ON e.event_id = c.event_id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY c.salience DESC, c.confidence DESC, c.created_at DESC LIMIT ?""",
                params,
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    @staticmethod
    def _atom_row_namespace(
        row: sqlite3.Row,
    ) -> tuple[str | None, str | None, str | None]:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return _namespace_identity(
            str(row["scope"]),
            app_id=metadata.get("app_id"),
            project_id=metadata.get("project_id"),
            session_id=metadata.get("session_id"),
        )

    @staticmethod
    def _atom_namespace_condition(
        *,
        scope: str | None,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> tuple[str | None, list[Any]]:
        def metadata_clause(field: str, value: str | None) -> tuple[str, list[Any]]:
            expression = f"json_extract(metadata_json, '$.{field}')"
            if value is None:
                return f"{expression} IS NULL", []
            return f"{expression} = ?", [value]

        if scope == "global":
            return None, []
        if scope:
            clauses: list[str] = []
            scoped_params: list[Any] = []
            for field, value in _namespace_fields(
                scope,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            ):
                if scope != "session" and value is None:
                    continue
                clause, values = metadata_clause(field, value)
                clauses.append(clause)
                scoped_params.extend(values)
            return (" AND ".join(clauses) if clauses else None), scoped_params
        if not (app_id or project_id or session_id):
            return None, []

        clauses = ["scope = 'global'"]
        params = []
        if project_id:
            project_clause, project_params = metadata_clause("project_id", project_id)
            clauses.append(f"(scope = 'project' AND {project_clause})")
            params.extend(project_params)
        if session_id:
            session_clauses = ["scope = 'session'"]
            session_params: list[Any] = []
            for field, value in _namespace_fields(
                "session",
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            ):
                clause, values = metadata_clause(field, value)
                session_clauses.append(clause)
                session_params.extend(values)
            clauses.append("(" + " AND ".join(session_clauses) + ")")
            params.extend(session_params)
        if app_id:
            app_clause, app_params = metadata_clause("app_id", app_id)
            clauses.append(f"(scope = 'app' AND {app_clause})")
            params.extend(app_params)
        return "(" + " OR ".join(clauses) + ")", params

    @staticmethod
    def _exact_namespace_condition(
        *,
        scope: str,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> tuple[str | None, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in _namespace_fields(
            scope,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
        ):
            if value is None:
                clauses.append(f"e.{column} IS NULL")
            else:
                clauses.append(f"e.{column} = ?")
                params.append(value)
        return (" AND ".join(clauses) if clauses else None), params

    @staticmethod
    def _namespace_condition(
        *,
        scope: str | None,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> tuple[str | None, list[Any]]:
        if scope == "global":
            return None, []
        if scope == "session":
            return MemoryStore._exact_namespace_condition(
                scope=scope,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            )
        if scope:
            clauses: list[str] = []
            scoped_params: list[Any] = []
            for column, value in _namespace_fields(
                scope,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            ):
                if value is not None:
                    clauses.append(f"e.{column} = ?")
                    scoped_params.append(value)
            return (" AND ".join(clauses) if clauses else None), scoped_params
        if not (app_id or project_id or session_id):
            return None, []

        clauses = ["c.scope = 'global'"]
        params: list[Any] = []
        if project_id:
            clauses.append("(c.scope = 'project' AND e.project_id = ?)")
            params.append(project_id)
        if session_id:
            session_clauses = ["c.scope = 'session'"]
            session_params: list[Any] = []
            for column, value in _namespace_fields(
                "session",
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
            ):
                if value is None:
                    session_clauses.append(f"e.{column} IS NULL")
                else:
                    session_clauses.append(f"e.{column} = ?")
                    session_params.append(value)
            clauses.append("(" + " AND ".join(session_clauses) + ")")
            params.extend(session_params)
        if app_id:
            clauses.append("(c.scope = 'app' AND e.app_id = ?)")
            params.append(app_id)
        return "(" + " OR ".join(clauses) + ")", params

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        request_raw = out.pop("request_json", None)
        metadata_raw = out.pop("metadata_json", None)
        try:
            request = json.loads(request_raw) if request_raw else {}
        except json.JSONDecodeError:
            request = {}
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except json.JSONDecodeError:
            metadata = {}
        out["request"] = request
        out["messages"] = request.get("messages", []) if isinstance(request, dict) else []
        out["metadata"] = metadata
        return out

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        for key in ("reasons_json", "invalidates_json", "metadata_json"):
            raw = out.pop(key, None)
            target = key.removesuffix("_json")
            try:
                out[target] = json.loads(raw) if raw else ([] if target in {"reasons", "invalidates"} else {})
            except json.JSONDecodeError:
                out[target] = [] if target in {"reasons", "invalidates"} else {}
        return out

    @staticmethod
    def _row_to_extraction_job(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        for key in ("payload_json", "result_json", "metadata_json"):
            raw = out.pop(key, None)
            target = key.removesuffix("_json")
            try:
                out[target] = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                out[target] = {}
        return out

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        raw = out.pop("metadata_json", None)
        try:
            out["metadata"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            out["metadata"] = {}
        return out

    @staticmethod
    def _row_to_entity_alias(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        raw = out.pop("metadata_json", None)
        try:
            out["metadata"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            out["metadata"] = {}
        return out

    @staticmethod
    def _job_id(source_event_id: str | None, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = sha1(f"{source_event_id or ''}:{payload_json}".encode()).hexdigest()[:16]
        return f"job_{digest}"

    @staticmethod
    def _atom_id(atom: dict[str, Any]) -> str:
        metadata_value = atom.get("metadata")
        metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
        namespace = _namespace_identity(
            str(atom.get("scope") or "global"),
            app_id=metadata.get("app_id"),
            project_id=metadata.get("project_id"),
            session_id=metadata.get("session_id"),
        )
        parts = (
            atom.get("source_event_id") or "",
            atom.get("source_job_id") or "",
            atom.get("type") or "",
            atom.get("subject") or "",
            atom.get("predicate") or "",
            atom.get("object") or "",
            atom.get("scope") or "global",
            *namespace,
        )
        digest = sha1(":".join(_norm(str(part)) for part in parts).encode()).hexdigest()[:16]
        return f"atom_{digest}"

    @staticmethod
    def _alias_id(alias: dict[str, Any]) -> str:
        entity_id = alias.get("entity_id") or ""
        parts = (entity_id, alias.get("alias") or "", alias.get("type") or "concept", alias.get("scope") or "global")
        digest = sha1(":".join(_norm(str(part)) for part in parts).encode()).hexdigest()[:16]
        return f"alias_{digest}"

    @staticmethod
    def _entity_id(name: str, entity_type: str) -> str:
        canonical = canonicalize_entity_name(name) or _norm(name)
        digest = sha1(f"{entity_type}:{canonical}".encode()).hexdigest()[:16]
        return f"ent_{digest}"

    @staticmethod
    def _edge_id(candidate_id: str, relation: str) -> str:
        digest = sha1(f"{candidate_id}:{_norm(relation)}".encode()).hexdigest()[:16]
        return f"edge_{digest}"

    @staticmethod
    def _upsert_entity_conn(conn: sqlite3.Connection, entity_id: str, name: str, entity_type: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities (entity_id, name, type) VALUES (?,?,?)",
            (entity_id, name, entity_type),
        )

    @classmethod
    def _upsert_canonical_alias_conn(
        cls,
        conn: sqlite3.Connection,
        *,
        entity_id: str,
        raw_name: str,
        canonical_name: str,
        entity_type: str,
        scope: str,
        candidate_id: str,
    ) -> None:
        alias = _clean_entity_label(raw_name)
        if not alias or _norm(alias) == _norm(canonical_name):
            return
        alias_record = {
            "entity_id": entity_id,
            "alias": alias,
            "type": entity_type,
            "scope": scope,
        }
        alias_id = cls._alias_id(alias_record)
        conn.execute(
            """INSERT OR REPLACE INTO memory_entity_aliases (
                alias_id, entity_id, alias, type, scope, confidence,
                valid_at, invalid_at, expired_at, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                alias_id,
                entity_id,
                alias,
                entity_type,
                scope,
                1.0,
                None,
                None,
                None,
                json.dumps({"source": "canonical_graph_projection", "candidate_id": candidate_id}, ensure_ascii=False),
            ),
        )


    def _load_candidate_embedding_vectors(self, candidate_ids: list[str]) -> dict[str, dict[int, float]]:
        """Load cached candidate vectors (hash or model) keyed by candidate_id."""
        ids = [cid for cid in candidate_ids if cid]
        if not ids:
            return {}
        try:
            aliases = self.query_entity_aliases(type="embedding_cache", scope="system", active_only=True, limit=10000)
        except Exception:
            return {}
        wanted = set(ids)
        out: dict[str, dict[int, float]] = {}
        for alias in aliases:
            meta = alias.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    continue
            cid = str(meta.get("candidate_id") or "")
            if cid not in wanted:
                continue
            vec = meta.get("sparse") or meta.get("vector")
            parsed = _coerce_vector(vec)
            if parsed:
                out[cid] = parsed
        return out

    def cache_candidate_embeddings(
        self,
        *,
        status: str = "active",
        limit: int = 2000,
        force: bool = False,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Cache lightweight semantic vectors for active candidates.

        Uses a deterministic hashed bag-of-words embedding so search can fuse a
        dense-like signal without requiring a local embedding model download.
        """
        self.init()
        rows = self.query_candidates(status=status, limit=max(1, int(limit)), project_id=project_id)
        existing = set() if force else set(self._load_candidate_embedding_vectors(
            [str(r.get("candidate_id") or "") for r in rows]
        ))
        written = 0
        skipped = 0
        for row in rows:
            cid = str(row.get("candidate_id") or "")
            if not cid:
                continue
            if cid in existing and not force:
                skipped += 1
                continue
            text = " ".join(
                str(row.get(key) or "")
                for key in ("type", "subject", "predicate", "object", "text")
            )
            sparse = _hash_embed(text)
            self.store_entity_alias({
                "entity_id": f"embedding:{cid}",
                "alias": "embedding_vector",
                "type": "embedding_cache",
                "scope": "system",
                "confidence": 1.0,
                "metadata": {
                    "candidate_id": cid,
                    "model": "hash-bow-v1",
                    "dims": HASH_EMBED_DIMS,
                    "sparse": {str(k): v for k, v in sparse.items()},
                },
            })
            written += 1
        return {
            "candidates": len(rows),
            "written": written,
            "skipped": skipped,
            "force": bool(force),
            "model": "hash-bow-v1",
        }

    def compact_candidates_to_atoms(
        self,
        *,
        project_id: str | None = None,
        types: list[str] | None = None,
        min_confidence: float = 0.9,
        limit: int = 2000,
        dry_run: bool = True,
        forget_sources: bool = False,
    ) -> dict[str, Any]:
        """Compact durable active candidates into memory_atoms summaries.

        Groups by slot and exact namespace. It keeps the newest high-confidence
        value as an atom. Optional forget_sources archives older values in that group.
        """
        self.init()
        selected_types = types or ["decision", "preference", "constraint", "instruction", "fact"]
        rows = self.query_candidates(status="active", limit=max(1, int(limit)), project_id=project_id)
        durable = [
            row for row in rows
            if str(row.get("type") or "") in selected_types
            and float(row.get("confidence") or 0.0) >= float(min_confidence)
            and str(row.get("scope") or "") in {"global", "project"}
        ]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in durable:
            namespace = _namespace_identity(
                str(row.get("scope") or "project"),
                app_id=row.get("app_id"),
                project_id=row.get("project_id"),
                session_id=row.get("session_id"),
            )
            key = (
                str(row.get("type") or ""),
                str(row.get("subject") or ""),
                str(row.get("predicate") or ""),
                str(row.get("scope") or "project"),
                *namespace,
            )
            groups.setdefault(key, []).append(row)

        planned: list[dict[str, Any]] = []
        for key, items in groups.items():
            items_sorted = sorted(
                items,
                key=lambda r: (
                    float(r.get("confidence") or 0.0),
                    str(r.get("valid_from") or r.get("created_at") or ""),
                ),
                reverse=True,
            )
            keep = items_sorted[0]
            planned.append({
                "type": key[0],
                "subject": key[1],
                "predicate": key[2],
                "scope": key[3],
                "app_id": key[4],
                "project_id": key[5],
                "session_id": key[6],
                "object": keep.get("object"),
                "text": keep.get("text"),
                "confidence": keep.get("confidence"),
                "source_candidate_id": keep.get("candidate_id"),
                "source_count": len(items_sorted),
                "source_candidate_ids": [str(i.get("candidate_id")) for i in items_sorted],
            })

        result: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "project_id": project_id,
            "types": selected_types,
            "min_confidence": float(min_confidence),
            "candidate_inputs": len(durable),
            "atom_groups": len(planned),
            "atoms_written": 0,
            "sources_forgotten": 0,
            "groups": planned[:50],
        }
        if dry_run or not planned:
            return result

        atoms_written = 0
        forgotten = 0
        for item in planned:
            atom = {
                "type": item["type"],
                "subject": item["subject"],
                "predicate": item["predicate"],
                "object": item["object"],
                "text": item["text"],
                "scope": item["scope"],
                "confidence": float(item.get("confidence") or 0.0),
                "source_event_id": None,
                "metadata": {
                    "source": "compact_candidates_to_atoms",
                    "source_candidate_id": item.get("source_candidate_id"),
                    "source_candidate_ids": item.get("source_candidate_ids"),
                    "app_id": item.get("app_id"),
                    "project_id": item.get("project_id"),
                    "session_id": item.get("session_id"),
                    "supersedes_prior": True,
                },
            }
            self.store_atom(atom)
            atoms_written += 1
            if forget_sources:
                # Keep the canonical newest candidate; forget older duplicates in the slot.
                for cid in list(item.get("source_candidate_ids") or [])[1:]:
                    if self.forget_candidate(str(cid)):
                        forgotten += 1
        result["atoms_written"] = atoms_written
        result["sources_forgotten"] = forgotten
        return result

    def get_namespaces(self) -> dict[str, list[str]]:
        """Return distinct app_ids, project_ids, session_ids, and scopes in the memory database."""
        self.init()
        res: dict[str, list[str]] = {"app_ids": [], "project_ids": [], "session_ids": [], "scopes": []}
        with self._connect() as conn:
            # Get distinct app_ids
            rows = conn.execute("SELECT DISTINCT app_id FROM memory_events WHERE app_id IS NOT NULL AND app_id != ''").fetchall()
            res["app_ids"] = sorted([r[0] for r in rows if r[0]])

            # Get distinct project_ids
            rows = conn.execute("SELECT DISTINCT project_id FROM memory_events WHERE project_id IS NOT NULL AND project_id != ''").fetchall()
            res["project_ids"] = sorted([r[0] for r in rows if r[0]])

            # Get distinct session_ids
            rows = conn.execute("SELECT DISTINCT session_id FROM memory_events WHERE session_id IS NOT NULL AND session_id != ''").fetchall()
            res["session_ids"] = sorted([r[0] for r in rows if r[0]])

            # Get distinct scopes
            rows = conn.execute("SELECT DISTINCT scope FROM memory_candidates WHERE scope IS NOT NULL AND scope != ''").fetchall()
            res["scopes"] = sorted([r[0] for r in rows if r[0]])
        return res


_store_instance: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store(path: Path | None = None) -> MemoryStore:
    global _store_instance
    if path is not None:
        store = MemoryStore(path)
        store.init()
        return store
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = MemoryStore()
                _store_instance.init()
    return _store_instance


def reset_memory_store() -> None:
    global _store_instance
    _store_instance = None


def canonicalize_entity_name(value: str) -> str | None:
    """Return a short deterministic entity label for graph projection.

    Memory candidates keep their raw subject/object text for retrieval. Graph
    nodes use this safer form so legacy facts do not turn arbitrary prose into
    node identifiers.
    """
    cleaned = _clean_entity_label(value)
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for prefix in _SAFE_ENTITY_PREFIXES:
        match = re.fullmatch(rf"{re.escape(prefix)}\s+(.+)", lowered)
        if match:
            lowered = match.group(1).strip()
            break
    return lowered


def canonicalize_graph_entity(value: str) -> str | None:
    canonical = canonicalize_entity_name(value)
    if not canonical or _looks_like_long_text_entity(canonical):
        return None
    return canonical


def _select_graph_entity_match(raw_name: str, projection: str, candidates: list[dict[str, Any]]) -> dict[str, str] | None:
    if not candidates:
        return None
    target_keys = {_entity_resolution_key(raw_name), _entity_resolution_key(projection), _norm(projection)}
    target_keys = {key for key in target_keys if key}
    best: tuple[float, str, str] | None = None
    for candidate in candidates:
        names = [str(candidate.get("name") or ""), *[str(alias) for alias in candidate.get("aliases", [])]]
        name_keys = {_entity_resolution_key(name) for name in names}
        name_keys.update(_norm(name) for name in names)
        name_keys = {key for key in name_keys if key}
        if target_keys & name_keys:
            return {"entity_id": str(candidate["entity_id"]), "name": str(candidate["name"])}
        score = max((_entity_resolution_similarity(target_key, name_key) for target_key in target_keys for name_key in name_keys), default=0.0)
        if score >= 0.9:
            contender = (score, str(candidate.get("name") or ""), str(candidate.get("entity_id") or ""))
            if best is None or contender > best:
                best = contender
    if best is None:
        return None
    return {"entity_id": best[2], "name": best[1]}


def _entity_resolution_key(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    text = _clean_entity_label(text).lower()
    for prefix in _SAFE_ENTITY_PREFIXES:
        if text.startswith(prefix + " "):
            text = text[len(prefix) + 1 :].strip()
            break
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _entity_resolution_similarity(left: str, right: str) -> float:
    if not left or not right or left == right:
        return 1.0 if left and right else 0.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return 0.0
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    if token_score < 0.8:
        return token_score
    left_chars = _char_shingles(left)
    right_chars = _char_shingles(right)
    if not left_chars or not right_chars:
        return token_score
    char_score = len(left_chars & right_chars) / len(left_chars | right_chars)
    return max(token_score, char_score)


def _char_shingles(value: str) -> set[str]:
    cleaned = value.replace(" ", "")
    if len(cleaned) < 3:
        return {cleaned} if cleaned else set()
    return {cleaned[idx : idx + 3] for idx in range(len(cleaned) - 2)}


def _clean_entity_label(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().strip("'\"").split())
    return cleaned.strip(" .;:-")


def _canonical_atom_subject(value: str) -> str:
    return canonicalize_entity_name(value) or _norm(value)


def _truthy_supersession_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _looks_like_long_text_entity(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    words = re.findall(r"\w+", text)
    if len(text) > MAX_GRAPH_ENTITY_LABEL_CHARS or len(words) > MAX_GRAPH_ENTITY_LABEL_WORDS:
        return True
    if "\n" in text or "\r" in text:
        return True
    if any(char in text for char in "{}[]"):
        return True
    if len(re.findall(r"[.!?]", text)) >= 2:
        return True
    if re.search(r"[.!?]\s+\w", text):
        return True
    # Legacy remembered facts often place a complete clause in the object; keep
    # the fact searchable, but do not project that clause as a graph node.
    if len(words) >= 7 and re.search(r"\b(is|are|was|were|will|should|must|need|needs|prefer|prefers|decided|uses|use|has|have)\b", text):
        return True
    return False



HASH_EMBED_DIMS = 256


def _hash_embed(text: str, *, dims: int = HASH_EMBED_DIMS) -> dict[int, float]:
    """Deterministic hashed bag-of-words embedding (sparse)."""
    tokens = re.findall(r"[a-z0-9_]{2,}", (text or "").lower())
    if not tokens:
        return {}
    vec: dict[int, float] = {}
    for token in tokens:
        digest = sha1(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] = vec.get(idx, 0.0) + sign
    # L2 normalize sparse
    norm = sum(v * v for v in vec.values()) ** 0.5
    if norm <= 0:
        return {}
    return {k: (v / norm) for k, v in vec.items()}


def _cosine_sparse(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        if k in b:
            dot += v * b[k]
    return max(0.0, min(1.0, float(dot)))


def _coerce_vector(value: Any) -> dict[int, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        out: dict[int, float] = {}
        for k, v in value.items():
            try:
                out[int(k)] = float(v)
            except Exception:
                continue
        return out or None
    if isinstance(value, list):
        # Dense list -> sparse nonzero map
        out = {i: float(v) for i, v in enumerate(value) if abs(float(v)) > 1e-12}
        return out or None
    return None


def _semantic_similarity(query: str, document: str) -> float:
    """Cheap semantic-ish similarity via token + char-ngram overlap."""
    q = (query or "").lower().strip()
    d = (document or "").lower().strip()
    if not q or not d:
        return 0.0
    q_tokens = set(re.findall(r"[a-z0-9_]{2,}", q))
    d_tokens = set(re.findall(r"[a-z0-9_]{2,}", d))
    token_score = 0.0
    if q_tokens and d_tokens:
        token_score = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
    q_ng = _char_shingles(q)
    d_ng = _char_shingles(d)
    ngram_score = 0.0
    if q_ng and d_ng:
        ngram_score = len(q_ng & d_ng) / max(len(q_ng | d_ng), 1)
    # Soft synonym-ish bridges for common infra/product terms.
    bridges = {
        ("cpu", "governor"), ("governor", "powersave"), ("cart", "checkout"),
        ("rewrite", "refactor"), ("commit", "pushed"), ("convex", "backend"),
    }
    bridge_hits = 0.0
    blob = f"{q} {d}"
    for a, b in bridges:
        if a in q_tokens and b in d_tokens:
            bridge_hits += 1
        elif b in q_tokens and a in d_tokens:
            bridge_hits += 1
        elif a in q and b in blob:
            bridge_hits += 0.5
    bridge_score = min(0.4, bridge_hits * 0.15)
    return max(0.0, min(1.0, 0.55 * token_score + 0.30 * ngram_score + bridge_score))


def _search_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(term) >= 2][:12]


def _is_single_value_predicate(type_: str, predicate: str) -> bool:
    pred = _norm(predicate)
    if not pred:
        return False
    if pred in {_norm(item) for item in SINGLE_VALUE_PREDICATES}:
        return True
    # Common mutable "current X" slots, including commit/status history noise.
    if re.search(r"\b(latest|current|active|default)\b", pred):
        return True
    if pred.endswith("_status") or pred.endswith(" status"):
        return True
    if "commit" in pred and any(token in pred for token in ("latest", "pushed", "local", "head")):
        return True
    if type_ == "workflow_state" and pred not in {_norm(item) for item in ADDITIVE_WORKFLOW_PREDICATES}:
        # Default workflow slots are single-value unless explicitly additive.
        return pred in {_norm(item) for item in ("current_task", "next_action", "blocker", "status", "core_status", "current_phase")}
    if type_ == "decision" and pred in {"decided", "decided_as", "mode", "uses", "use"}:
        return True
    return False


def _rank_search_results(
    rows: list[dict[str, Any]],
    *,
    query: str,
    terms: list[str],
    project_id: str | None = None,
    session_id: str | None = None,
    app_id: str | None = None,
    dense_vectors: dict[str, dict[int, float]] | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    query_l = (query or "").lower()
    term_set = set(terms)
    query_vec = _hash_embed(query_l)
    dense_vectors = dense_vectors or {}
    for idx, row in enumerate(rows):
        subject = str(row.get("subject") or "")
        predicate = str(row.get("predicate") or "")
        obj = str(row.get("object") or "")
        text = str(row.get("text") or "")
        hay = f"{subject} {predicate} {obj} {text}".lower()
        tokens = set(re.findall(r"[a-z0-9_]+", hay))

        overlap = len(term_set & tokens) / max(len(term_set), 1)
        phrase = 1.0 if query_l and query_l in hay else 0.0
        field_hits = 0.0
        for term in terms:
            if term in subject.lower():
                field_hits += 0.35
            if term in predicate.lower():
                field_hits += 0.2
            if term in obj.lower():
                field_hits += 0.25
            if term in text.lower():
                field_hits += 0.1
        lexical = min(1.5, overlap + phrase + min(field_hits, 1.0))
        semantic = _semantic_similarity(query_l, hay)

        type_boost = float(_TYPE_RANK_BOOST.get(str(row.get("type") or ""), 0.8))
        confidence = float(row.get("confidence") or 0.0)
        salience = float(row.get("salience") or 0.0)

        # Recency: ~1.0 for now, decays toward 0.55 over ~90 days.
        age_days = _age_days(str(row.get("valid_from") or row.get("created_at") or ""))
        recency = max(0.55, 1.0 - min(age_days, 90.0) / 180.0)

        ns_boost = 1.0
        if project_id and str(row.get("project_id") or "") == str(project_id):
            ns_boost += 0.25
        if session_id and str(row.get("session_id") or "") == str(session_id):
            ns_boost += 0.15
        if app_id and str(row.get("app_id") or "") == str(app_id):
            ns_boost += 0.1
        if project_id and str(row.get("project_id") or "") and str(row.get("project_id") or "") != str(project_id):
            if str(row.get("scope") or "") == "project":
                ns_boost -= 0.35

        dense = 0.0
        cid = str(row.get("candidate_id") or "")
        if dense_vectors and cid in dense_vectors:
            dense = _cosine_sparse(query_vec, dense_vectors[cid])

        score = (
            0.40 * lexical
            + 0.16 * semantic
            + 0.12 * dense
            + 0.14 * type_boost
            + 0.10 * recency
            + 0.08 * confidence
            + 0.06 * min(max(salience, 0.0), 1.5)
        ) * max(ns_boost, 0.4)
        # Stable tie-break: original retrieval order.
        score -= idx * 1e-6

        out = dict(row)
        out["score"] = round(score, 6)
        out["rank_components"] = {
            "lexical": round(lexical, 4),
            "semantic": round(semantic, 4),
            "dense": round(dense, 4),
            "type_boost": round(type_boost, 4),
            "recency": round(recency, 4),
            "confidence": round(confidence, 4),
            "salience": round(salience, 4),
            "namespace_boost": round(ns_boost, 4),
        }
        ranked.append(out)
    ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked


def _age_days(ts: str) -> float:
    if not ts:
        return 30.0
    try:
        # Accept both YYYY-MM-DDTHH:MM:SS(.fff) and date-only.
        raw = ts.strip().replace("Z", "")
        if "T" in raw:
            date_part, time_part = raw.split("T", 1)
            year, month, day = [int(x) for x in date_part.split("-")[:3]]
            hour = minute = second = 0
            if time_part:
                hm = time_part.split(":")
                hour = int(hm[0]) if len(hm) > 0 and hm[0] else 0
                minute = int(hm[1]) if len(hm) > 1 and hm[1] else 0
                second = int(float(hm[2])) if len(hm) > 2 and hm[2] else 0
            from datetime import datetime, timezone

            then = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (now - then).total_seconds() / 86400.0)
        year, month, day = [int(x) for x in raw.split("-")[:3]]
        from datetime import datetime, timezone

        then = datetime(year, month, day, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - then).total_seconds() / 86400.0)
    except Exception:
        return 30.0


def _doctor_recommendations(
    *,
    conflicts: int,
    single_value_conflicts: int,
    dups: int,
    active_no_edge: int,
    noisy: int,
    failed_jobs: int,
) -> list[str]:
    tips: list[str] = []
    if single_value_conflicts:
        tips.append("ppmlx memory migrate-temporal-conflicts --confirm")
    elif conflicts:
        tips.append("ppmlx memory temporal-conflicts  # review multi-object slots")
    if dups:
        tips.append("ppmlx memory dedup-scan")
    if noisy:
        tips.append("ppmlx memory prune --confirm")
    if active_no_edge:
        tips.append("ppmlx memory rebuild --confirm")
    if failed_jobs:
        tips.append("ppmlx memory jobs --status failed")
    if not tips:
        tips.append("No maintenance actions required")
    return tips


def _compression_ratio(original_tokens: int, reduced_tokens: int) -> float:
    if reduced_tokens <= 0:
        return 0.0
    return round(original_tokens / reduced_tokens, 4)


def _round_or_none(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (percentile / 100)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _norm(value: str) -> str:
    return " ".join(str(value).lower().strip().split())
