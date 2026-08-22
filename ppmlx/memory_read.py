"""memory-read/v1 service: grants, sessions, disclosure labels, echo guard.

Implements the minimal memory-read/v1 slice per ADR 0006 and the Phase 7
wire-contract spec: grant credential verification, 15-minute read sessions,
server-side scope enforcement, disclosure-label filtering at read output,
provenance trust marking, and the re-capture feedback-loop dedup guard.

Secrets policy: raw bearer credentials are never stored (only a SHA-256
verifier), never logged, and never included in error messages.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MEMORY_READ_VERSION = "memory-read/v1"
SESSION_TTL_SECONDS = 15 * 60
CURSOR_TTL_SECONDS = 15 * 60
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
ALLOWED_TOOLS = ("memory_search", "memory_get_context", "memory_graph_walk", "memory_stats")
DISCLOSURE_LABELS = ("local_only", "remote_allowed", "secret")
DEFAULT_DISCLOSURE_LABEL = "local_only"
ECHO_SOURCE_TAG = "memory_read_echo"
_RECENT_READ_WINDOW = 512
_RECENT_READ_HASH_WINDOW = _RECENT_READ_WINDOW


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MemoryReadError(Exception):
    """Typed error carrying a stable code and HTTP status (ADR 0006)."""

    STATUS = {
        "permission_denied": 403,
        "credential_required": 401,
        "credential_invalid": 401,
        "credential_expired": 401,
        "credential_revoked": 401,
        "scope_denied": 403,
        "tool_denied": 403,
        "session_required": 401,
        "session_invalid": 401,
        "session_expired": 410,
        "cursor_expired": 410,
        "validation_error": 400,
        "cursor_invalid": 400,
        "version_unsupported": 400,
        "memory_unavailable": 503,
        "rate_limited": 429,
        "internal_error": 500,
    }

    def __init__(self, code: str, *, retryable: bool = False):
        self.code = code
        self.status = self.STATUS.get(code, 500)
        self.retryable = retryable
        super().__init__(code)


def credential_verifier(credential: str) -> str:
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Grant:
    grant_id: str
    harness_name: str
    harness_version: str
    instance_id: str
    allowed_scopes: list[dict[str, str]]
    allowed_tools: list[str]
    issued_at: str
    expires_at: str
    revoked_at: str | None = None
    remote_capable: bool = False
    verifier: str = ""  # SHA-256 of the raw credential; raw credential never stored.

    def to_public(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "harness": {
                "name": self.harness_name,
                "version": self.harness_version,
                "instance_id": self.instance_id,
            },
            "allowed_tools": list(self.allowed_tools),
            "allowed_scopes": [dict(s) for s in self.allowed_scopes],
            "expires_at": self.expires_at,
        }


@dataclass
class ReadSession:
    session_id: str
    grant_id: str
    created_at: float
    expires_at: float
    request_ids: dict[str, str] = field(default_factory=dict)  # request_id -> input fingerprint


class MemoryReadService:
    """Grant store + session manager for the memory-read/v1 contract.

    Grants live in a dedicated SQLite database (deployment mounts it in
    protected OS storage; tests use a tmp path). Only credential verifiers are
    persisted.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else _default_grants_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ReadSession] = {}
        self._recent_read_hashes: deque[str] = deque(maxlen=_RECENT_READ_HASH_WINDOW)
        self._recent_read_index: set[str] = set()
        self._init_db()

    # ------------------------------------------------------------------
    # Grants
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memory_grants (
                    grant_id TEXT PRIMARY KEY,
                    harness_name TEXT NOT NULL,
                    harness_version TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    allowed_scopes_json TEXT NOT NULL,
                    allowed_tools_json TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    remote_capable INTEGER NOT NULL DEFAULT 0,
                    credential_verifier TEXT NOT NULL
                )"""
            )
            conn.commit()

    def create_grant(
        self,
        *,
        harness_name: str,
        harness_version: str,
        instance_id: str,
        allowed_scopes: list[dict[str, str]],
        allowed_tools: list[str],
        lifetime_days: int = 30,
        remote_capable: bool = False,
    ) -> tuple[Grant, str]:
        """Create a grant; returns (grant, raw_credential). The raw credential
        is shown exactly once and never persisted or logged."""
        for tool in allowed_tools:
            if tool not in ALLOWED_TOOLS:
                raise MemoryReadError("validation_error")
        for s in allowed_scopes:
            if s.get("type") not in {"project", "app", "repository", "global"}:
                raise MemoryReadError("validation_error")
            if s["type"] == "global" and s.get("id") != "global":
                raise MemoryReadError("validation_error")
        credential = "mrc_" + secrets.token_urlsafe(32)
        now = _utcnow()
        grant = Grant(
            grant_id="mrg_" + uuid.uuid4().hex,
            harness_name=harness_name,
            harness_version=harness_version,
            instance_id=instance_id,
            allowed_scopes=allowed_scopes,
            allowed_tools=allowed_tools,
            issued_at=rfc3339(now),
            expires_at=rfc3339(now + timedelta(days=lifetime_days)),
            remote_capable=remote_capable,
            verifier=credential_verifier(credential),
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO memory_grants (
                    grant_id, harness_name, harness_version, instance_id,
                    allowed_scopes_json, allowed_tools_json, issued_at, expires_at,
                    revoked_at, remote_capable, credential_verifier
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    grant.grant_id, grant.harness_name, grant.harness_version,
                    grant.instance_id, json.dumps(grant.allowed_scopes),
                    json.dumps(grant.allowed_tools), grant.issued_at, grant.expires_at,
                    grant.revoked_at, int(grant.remote_capable), grant.verifier,
                ),
            )
            conn.commit()
        return grant, credential

    def revoke_grant(self, grant_id: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                "UPDATE memory_grants SET revoked_at = ? WHERE grant_id = ? AND revoked_at IS NULL",
                (rfc3339(_utcnow()), grant_id),
            )
            conn.commit()
        # Revocation kills live sessions immediately.
        self._sessions = {sid: s for sid, s in self._sessions.items() if s.grant_id != grant_id}
        return cur.rowcount > 0

    def _load_grant_by_verifier(self, verifier: str) -> Grant | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memory_grants WHERE credential_verifier = ?", (verifier,)
            ).fetchone()
        if row is None:
            return None
        return Grant(
            grant_id=row["grant_id"],
            harness_name=row["harness_name"],
            harness_version=row["harness_version"],
            instance_id=row["instance_id"],
            allowed_scopes=json.loads(row["allowed_scopes_json"]),
            allowed_tools=json.loads(row["allowed_tools_json"]),
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            remote_capable=bool(row["remote_capable"]),
            verifier=row["credential_verifier"],
        )

    # ------------------------------------------------------------------
    # Auth: handshake + per-request validation
    # ------------------------------------------------------------------
    def handshake(
        self,
        *,
        credential: str | None,
        version: str | None,
        is_loopback: bool,
    ) -> dict[str, Any]:
        if not credential:
            raise MemoryReadError("credential_required")
        if not version or version != MEMORY_READ_VERSION:
            raise MemoryReadError("version_unsupported")
        grant = self._load_grant_by_verifier(credential_verifier(credential))
        if grant is None:
            raise MemoryReadError("credential_invalid")
        if grant.revoked_at is not None:
            raise MemoryReadError("credential_revoked")
        if _utcnow() >= _parse_rfc3339(grant.expires_at):
            raise MemoryReadError("credential_expired")
        # local_only items are only ever served over a loopback connection.
        if not is_loopback:
            raise MemoryReadError("permission_denied")
        session = ReadSession(
            session_id="mrs_" + uuid.uuid4().hex,
            grant_id=grant.grant_id,
            created_at=time.time(),
            expires_at=time.time() + SESSION_TTL_SECONDS,
        )
        self._sessions[session.session_id] = session
        envelope = grant.to_public()
        envelope.update(
            {
                "version": MEMORY_READ_VERSION,
                "object": "memory_read_session",
                "read_session_id": session.session_id,
                "session_expires_at": rfc3339(
                    datetime.fromtimestamp(session.expires_at, tz=timezone.utc)
                ),
            }
        )
        return envelope

    def authenticate(
        self,
        *,
        credential: str | None,
        version: str | None,
        session_id: str | None,
    ) -> tuple[Grant, ReadSession]:
        """Validate bearer + session together; re-checks expiry/revocation."""
        if not credential:
            raise MemoryReadError("credential_required")
        if not version or version != MEMORY_READ_VERSION:
            raise MemoryReadError("version_unsupported")
        grant = self._load_grant_by_verifier(credential_verifier(credential))
        if grant is None:
            raise MemoryReadError("credential_invalid")
        if grant.revoked_at is not None:
            raise MemoryReadError("credential_revoked")
        if _utcnow() >= _parse_rfc3339(grant.expires_at):
            raise MemoryReadError("credential_expired")
        if not session_id:
            raise MemoryReadError("session_required")
        session = self._sessions.get(session_id)
        if session is None:
            raise MemoryReadError("session_invalid")
        if time.time() >= session.expires_at:
            self._sessions.pop(session_id, None)
            raise MemoryReadError("session_expired")
        if grant.grant_id != session.grant_id:
            raise MemoryReadError("credential_invalid")
        return grant, session

    def check_request_id(self, session: ReadSession, request_id: str, fingerprint: str) -> None:
        if not request_id:
            raise MemoryReadError("validation_error")
        seen = session.request_ids.get(request_id)
        if seen is not None and seen != fingerprint:
            raise MemoryReadError("validation_error")
        session.request_ids[request_id] = fingerprint

    # ------------------------------------------------------------------
    # Scope + tool enforcement (server-side; never trust caller scope)
    # ------------------------------------------------------------------
    def check_tool(self, grant: Grant, tool: str) -> None:
        if tool not in grant.allowed_tools:
            raise MemoryReadError("tool_denied")

    def resolve_scope(self, grant: Grant, scope: dict[str, Any] | None) -> dict[str, str]:
        if not isinstance(scope, dict) or set(scope) - {"type", "id"}:
            raise MemoryReadError("validation_error")
        stype = scope.get("type")
        sid = scope.get("id")
        if stype not in {"project", "app", "repository", "global"} or not isinstance(sid, str) or not sid:
            raise MemoryReadError("validation_error")
        if stype == "global" and sid != "global":
            raise MemoryReadError("validation_error")
        allowed = {(s.get("type"), s.get("id")) for s in grant.allowed_scopes}
        if (stype, sid) not in allowed:
            raise MemoryReadError("scope_denied")
        return {"type": stype, "id": sid}

    # ------------------------------------------------------------------
    # Disclosure labels + provenance at read output
    # ------------------------------------------------------------------
    def filter_item(self, row: dict[str, Any], grant: Grant) -> dict[str, Any] | None:
        """Apply disclosure-label filtering; return the wire item or None.

        Default label is local_only. secret items are never returned through
        the read contract; remote_allowed only for remote-capable grants.
        """
        try:
            metadata = row.get("metadata") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            if not metadata:
                metadata = json.loads(row.get("metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        label = metadata.get("disclosure", DEFAULT_DISCLOSURE_LABEL)
        if label not in DISCLOSURE_LABELS:
            label = DEFAULT_DISCLOSURE_LABEL
        if label == "secret":
            return None
        if label == "remote_allowed" and not grant.remote_capable:
            return None
        # local_only items pass because the endpoint is loopback-only.
        item = dict(row)
        item["disclosure"] = label
        item.setdefault("provenance", {})
        item["provenance"] = {
            **item["provenance"],
            "origin": item["provenance"].get("origin", "harness"),
            "origin_id": item["provenance"].get("origin_id", row.get("event_id")),
            "trust": "untrusted",
        }
        return item

    # ------------------------------------------------------------------
    # Feedback-loop guard: re-capture dedup
    # ------------------------------------------------------------------
    def note_read_outputs(self, texts: list[str]) -> None:
        """Record content hashes of read outputs so a re-capture of the same
        content is dropped by the ingest guard."""
        for text in texts:
            h = content_hash(text)
            if h not in self._recent_read_index:
                self._recent_read_index.add(h)
                self._recent_read_hashes.append(h)
                if len(self._recent_read_hashes) == _RECENT_READ_WINDOW:
                    old = self._recent_read_hashes.popleft()
                    self._recent_read_index.discard(old)

    def is_echo_event(self, event: dict[str, Any]) -> bool:
        metadata = event.get("metadata") or {}
        if not isinstance(metadata, dict):
            return False
        if metadata.get("source") == ECHO_SOURCE_TAG:
            return True
        text = event.get("response_text") or ""
        if not text:
            return False
        return content_hash(text) in self._recent_read_index

    # ------------------------------------------------------------------
    # Opaque cursors (HMAC-signed, bound to grant/tool/scope/params)
    # ------------------------------------------------------------------
    def _cursor_key(self) -> bytes:
        # Deterministic per-database signing key, stored alongside grants.
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_grants_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = conn.execute("SELECT value FROM memory_grants_meta WHERE key = 'cursor_key'").fetchone()
            if row is None:
                value = secrets.token_urlsafe(32)
                conn.execute(
                    "INSERT OR REPLACE INTO memory_grants_meta (key, value) VALUES ('cursor_key', ?)",
                    (value,),
                )
                conn.commit()
                return value.encode()
            return row[0].encode()

    def make_cursor(self, *, grant_id: str, tool: str, scope: dict[str, str], params: dict[str, Any], offset: int, expires_at: float) -> str:
        payload = json.dumps(
            {
                "g": grant_id, "t": tool, "s": f"{scope['type']}:{scope['id']}",
                "p": _stable_json(params), "o": offset, "e": int(expires_at),
            },
            sort_keys=True,
        ).encode()
        sig = hmac.new(self._cursor_key(), payload, hashlib.sha256).hexdigest()[:32]
        return f"mrc.{sig}.{offset}.{int(expires_at)}"

    def parse_cursor(self, token: str | None, *, grant_id: str, tool: str, scope: dict[str, str], params: dict[str, Any]) -> int:
        if token is None:
            return 0
        parts = token.split(".")
        if len(parts) != 4 or parts[0] != "mrc":
            raise MemoryReadError("cursor_invalid")
        _, sig, offset_s, expires_s = parts
        payload = json.dumps(
            {
                "g": grant_id, "t": tool, "s": f"{scope['type']}:{scope['id']}",
                "p": _stable_json(params), "o": int(offset_s), "e": int(expires_s),
            },
            sort_keys=True,
        ).encode()
        expected = hmac.new(self._cursor_key(), payload, hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, expected):
            raise MemoryReadError("cursor_invalid")
        if time.time() > int(expires_s):
            raise MemoryReadError("cursor_expired")
        return int(offset_s)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _parse_rfc3339(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _default_grants_db_path() -> Path:
    env = os.environ.get("PPMLX_MEMORY_GRANTS_DB")
    if env:
        return Path(env)
    return Path.home() / ".ppmlx" / "memory_grants.db"


_service: MemoryReadService | None = None


def get_service() -> MemoryReadService:
    """Lazily build the process-wide service (path from env, cached)."""
    global _service
    if _service is None:
        _service = MemoryReadService()
    return _service


def reset_service() -> None:
    global _service
    _service = None
