"""Tests for `ppmlx memory grant` CLI (create/list/revoke lifecycle)."""
import json
import os
import sqlite3
from datetime import datetime

import pytest
from typer.testing import CliRunner

from ppmlx.cli import app
from ppmlx.memory_read import (
    MEMORY_READ_VERSION,
    MemoryReadError,
    credential_verifier,
    reset_service,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def grants_db(tmp_path, monkeypatch):
    db = tmp_path / "memory_grants.db"
    monkeypatch.setenv("PPMLX_MEMORY_GRANTS_DB", str(db))
    reset_service()
    yield db
    reset_service()


def _invoke(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def _credentials(output: str) -> list[str]:
    return [ln for ln in output.splitlines() if ln.startswith("mrc_")]


def _db_rows(db):
    with sqlite3.connect(str(db)) as conn:
        return conn.execute("SELECT * FROM memory_grants").fetchall()


def test_grant_create_shows_credential_once(grants_db):
    result = _invoke(["memory", "grant", "create"])
    assert "shown ONLY now" in result.output
    creds = _credentials(result.output)
    assert len(creds) == 1
    credential = creds[0]
    # The raw credential is never persisted — only its SHA-256 verifier.
    with sqlite3.connect(str(grants_db)) as conn:
        verifiers = [
            r[0] for r in conn.execute("SELECT credential_verifier FROM memory_grants")
        ]
    assert len(verifiers) == 1
    assert verifiers[0] == credential_verifier(credential)
    assert credential not in "".join(str(r) for r in _db_rows(grants_db))


def test_grant_create_with_options(grants_db):
    result = _invoke([
        "memory", "grant", "create",
        "--project", "myproj", "--ttl-hours", "2", "--remote-capable",
    ])
    assert '"type": "project"' in result.output and "myproj" in result.output
    assert len(_credentials(result.output)) == 1
    with sqlite3.connect(str(grants_db)) as conn:
        row = conn.execute(
            "SELECT issued_at, expires_at, remote_capable, allowed_tools_json,"
            " allowed_scopes_json FROM memory_grants"
        ).fetchone()
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    delta = (
        datetime.strptime(row[1], fmt) - datetime.strptime(row[0], fmt)
    ).total_seconds()
    assert abs(delta - 2 * 3600) < 60
    assert row[2] == 1
    assert json.loads(row[3]) == ["memory_search", "memory_stats"]
    assert json.loads(row[4]) == [{"type": "project", "id": "myproj"}]


def test_grant_list_never_shows_credentials(grants_db):
    _invoke(["memory", "grant", "create"])
    result = _invoke(["memory", "grant", "list"])
    assert "mrg_" in result.output
    assert "mrc_" not in result.output
    listing = json.loads(_invoke(["memory", "grant", "list", "--json"]).output)
    assert len(listing) == 1
    assert "credential_verifier" not in listing[0]


def test_grant_revoke_lifecycle(grants_db):
    create = _invoke(["memory", "grant", "create"])
    grant_id = next(
        ln.split(":", 1)[1].strip()
        for ln in create.output.splitlines()
        if ln.startswith("Grant created:")
    )
    result = _invoke(["memory", "grant", "revoke", grant_id])
    assert f"Revoked: {grant_id}" in result.output
    listing = json.loads(_invoke(["memory", "grant", "list", "--json"]).output)
    assert listing[0]["revoked_at"] is not None
    fail = runner.invoke(app, ["memory", "grant", "revoke", grant_id])
    assert fail.exit_code == 1


def test_created_credential_works_until_revoked(grants_db):
    """The CLI-issued credential passes the documented handshake flow."""
    from ppmlx.memory_read import get_service

    create = _invoke(["memory", "grant", "create"])
    credential = _credentials(create.output)[0]
    service = get_service()
    envelope = service.handshake(
        credential=credential,
        version=MEMORY_READ_VERSION,
        is_loopback=True,
    )
    session_id = envelope["read_session_id"]
    grant, session = service.authenticate(
        credential=credential,
        version=MEMORY_READ_VERSION,
        session_id=session_id,
    )
    grant_id = next(
        ln.split(":", 1)[1].strip()
        for ln in create.output.splitlines()
        if ln.startswith("Grant created:")
    )
    _invoke(["memory", "grant", "revoke", grant_id])
    with pytest.raises(MemoryReadError) as exc:
        service.authenticate(
            credential=credential,
            version=MEMORY_READ_VERSION,
            session_id=session_id,
        )
    assert exc.value.code == "credential_revoked"
