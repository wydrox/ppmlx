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
            conn.commit()

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
                    json.dumps(metadata or {}, ensure_ascii=False),
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

        modifier = f"-{seconds:.3f} seconds"
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
                (f"stale claim requeued after {seconds:g}s", modifier),
            ).rowcount
            failed = conn.execute(
                f"""UPDATE memory_extraction_jobs
                    SET status = 'failed', error = ?, failed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                        invalid_at = COALESCE(invalid_at, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                    WHERE {stale_condition}
                      AND attempts >= max_attempts""",
                (f"stale claim exceeded max attempts after {seconds:g}s", modifier),
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

    def complete_extraction_job(self, job_id: str, *, result: dict[str, Any] | None = None) -> bool:
        self.init()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE memory_extraction_jobs
                   SET status = 'completed', result_json = ?, error = NULL,
                       completed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                       invalid_at = COALESCE(invalid_at, strftime('%Y-%m-%dT%H:%M:%f', 'now'))
                   WHERE job_id = ?""",
                (json.dumps(result or {}, ensure_ascii=False), job_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def fail_extraction_job(self, job_id: str, error: str, *, retry: bool = False) -> bool:
        self.init()
        status_expr = "CASE WHEN ? AND attempts < max_attempts THEN 'queued' ELSE 'failed' END"
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"""UPDATE memory_extraction_jobs
                   SET status = {status_expr}, error = ?, failed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                       invalid_at = CASE WHEN ? AND attempts < max_attempts THEN invalid_at
                                         ELSE COALESCE(invalid_at, strftime('%Y-%m-%dT%H:%M:%f', 'now')) END
                   WHERE job_id = ?""",
                (1 if retry else 0, error, 1 if retry else 0, job_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def store_atom(self, atom: dict[str, Any]) -> dict[str, Any]:
        self.init()
        atom_id = str(atom.get("atom_id") or self._atom_id(atom))
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
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
                    json.dumps(atom.get("metadata", {}), ensure_ascii=False),
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
            """SELECT atom_id, subject, object FROM memory_atoms
               WHERE type = ? AND predicate = ? AND scope = ?
                 AND atom_id != ? AND invalid_at IS NULL
                 AND (expired_at IS NULL OR expired_at > strftime('%Y-%m-%dT%H:%M:%f', 'now'))""",
            (atom["type"], atom["predicate"], scope, atom_id),
        ).fetchall()
        if not rows:
            return

        canonical_subject = _canonical_atom_subject(atom["subject"])
        object_norm = _norm(str(atom["object"]))
        superseded_ids = [
            row["atom_id"]
            for row in rows
            if _canonical_atom_subject(row["subject"]) == canonical_subject and _norm(str(row["object"])) != object_norm
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
            conn.commit()

    def mark_invalidated(self, candidate_ids: list[str], *, invalidated_by: str) -> None:
        if not candidate_ids:
            return
        self.init()
        with self._lock, self._connect() as conn:
            for candidate_id in candidate_ids:
                conn.execute(
                    """UPDATE memory_candidates
                       SET status = 'superseded', valid_to = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                       WHERE candidate_id = ? AND status = 'active'""",
                    (candidate_id,),
                )
                conn.execute(
                    """UPDATE memory_edges
                       SET status = 'superseded', valid_to = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                       WHERE source_candidate_id = ? AND status = 'active'""",
                    (candidate_id,),
                )
            conn.commit()

    def upsert_memory_edge(self, candidate: dict[str, Any]) -> None:
        self.init()
        subject_projection = canonicalize_graph_entity(candidate["subject"])
        object_projection = canonicalize_graph_entity(candidate["object"])
        if subject_projection is None or object_projection is None:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM memory_edges WHERE source_candidate_id = ?", (candidate["candidate_id"],))
                for raw_name, projection in ((candidate["subject"], subject_projection), (candidate["object"], object_projection)):
                    if projection is None:
                        continue
                    entity_id = self._entity_id(projection, "concept")
                    self._upsert_entity_conn(conn, entity_id, projection, "concept")
                    self._upsert_canonical_alias_conn(
                        conn,
                        entity_id=entity_id,
                        raw_name=raw_name,
                        canonical_name=projection,
                        entity_type="concept",
                        scope=str(candidate.get("scope") or "global"),
                        candidate_id=str(candidate.get("candidate_id") or ""),
                    )
                conn.commit()
            return

        from_entity_id = self._entity_id(subject_projection, "concept")
        to_entity_id = self._entity_id(object_projection, "concept")
        edge_id = self._edge_id(candidate["candidate_id"], candidate["predicate"])
        with self._lock, self._connect() as conn:
            self._upsert_entity_conn(conn, from_entity_id, subject_projection, "concept")
            self._upsert_entity_conn(conn, to_entity_id, object_projection, "concept")
            self._upsert_canonical_alias_conn(
                conn,
                entity_id=from_entity_id,
                raw_name=candidate["subject"],
                canonical_name=subject_projection,
                entity_type="concept",
                scope=str(candidate.get("scope") or "global"),
                candidate_id=str(candidate.get("candidate_id") or ""),
            )
            self._upsert_canonical_alias_conn(
                conn,
                entity_id=to_entity_id,
                raw_name=candidate["object"],
                canonical_name=object_projection,
                entity_type="concept",
                scope=str(candidate.get("scope") or "global"),
                candidate_id=str(candidate.get("candidate_id") or ""),
            )
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
                    None,
                ),
            )
            conn.commit()

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

    def find_active_slot(self, *, type: str, subject: str, predicate: str, scope: str) -> list[dict[str, Any]]:
        self.init()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM memory_candidates
                   WHERE type = ? AND subject = ? AND predicate = ? AND scope = ? AND status = 'active'
                   ORDER BY created_at DESC""",
                (type, subject, predicate, scope),
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
    ) -> list[dict[str, Any]]:
        self.init()
        terms = _search_terms(query)
        if not terms:
            return []
        if self._fts_available is not False:
            try:
                return self._search_fts(
                    terms, status=status, scope=scope, limit=limit,
                    app_id=app_id, project_id=project_id, session_id=session_id,
                )
            except sqlite3.Error:
                self._fts_available = False
        return self._search_like(
            terms, status=status, scope=scope, limit=limit,
            app_id=app_id, project_id=project_id, session_id=session_id,
        )

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

        for candidate in candidates:
            add(str(candidate.get("subject") or ""), role="subject", candidate=candidate)
            add(str(candidate.get("object") or ""), role="object", candidate=candidate)
        for edge in edges:
            add(str(edge.get("from_name") or ""), role="edge_from")
            add(str(edge.get("to_name") or ""), role="edge_to")

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
        return {
            "path": str(self.path),
            "events": events,
            "candidates": candidates,
            "entities": entities,
            "edges": edges,
            "atoms": atoms,
            "compactions": compactions,
            "extraction_jobs": extraction_jobs,
            "by_status": {row[0]: row[1] for row in by_status},
            "jobs_by_status": {row[0]: row[1] for row in jobs_by_status},
        }

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
    def _namespace_condition(
        *,
        scope: str | None,
        app_id: str | None,
        project_id: str | None,
        session_id: str | None,
    ) -> tuple[str | None, list[Any]]:
        if scope == "global":
            return None, []
        if scope == "project" and project_id:
            return "e.project_id = ?", [project_id]
        if scope == "session" and session_id:
            return "e.session_id = ?", [session_id]
        if scope:
            return None, []
        if not (app_id or project_id or session_id):
            return None, []

        clauses = ["c.scope = 'global'"]
        params: list[Any] = []
        if project_id:
            clauses.append("(c.scope = 'project' AND e.project_id = ?)")
            params.append(project_id)
        if session_id:
            clauses.append("(c.scope = 'session' AND e.session_id = ?)")
            params.append(session_id)
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
        parts = (
            atom.get("source_event_id") or "",
            atom.get("source_job_id") or "",
            atom.get("type") or "",
            atom.get("subject") or "",
            atom.get("predicate") or "",
            atom.get("object") or "",
            atom.get("scope") or "global",
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


def _search_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(term) >= 2][:12]


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
