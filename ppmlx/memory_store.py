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
        from_entity_id = self._entity_id(candidate["subject"], "concept")
        to_entity_id = self._entity_id(candidate["object"], "concept")
        edge_id = self._edge_id(candidate["candidate_id"], candidate["predicate"])
        with self._lock, self._connect() as conn:
            self._upsert_entity_conn(conn, from_entity_id, candidate["subject"], "concept")
            self._upsert_entity_conn(conn, to_entity_id, candidate["object"], "concept")
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
            compactions = conn.execute("SELECT COUNT(*) FROM memory_compactions").fetchone()[0]
        return {
            "path": str(self.path),
            "events": events,
            "candidates": candidates,
            "entities": entities,
            "edges": edges,
            "compactions": compactions,
            "by_status": {row[0]: row[1] for row in by_status},
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
    def _entity_id(name: str, entity_type: str) -> str:
        digest = sha1(f"{entity_type}:{_norm(name)}".encode()).hexdigest()[:16]
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
