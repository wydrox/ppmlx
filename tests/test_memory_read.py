"""Tests for the memory-read/v1 minimal slice (ADR 0006)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ppmlx.memory_read as mr
from ppmlx.memory_read import MemoryReadService, content_hash
from ppmlx.memory_store import MemoryStore


@pytest.fixture()
def service(tmp_path: Path, monkeypatch) -> MemoryReadService:
    monkeypatch.setenv("PPMLX_MEMORY_GRANTS_DB", str(tmp_path / "grants.db"))
    monkeypatch.setenv("PPMLX_MEMORY_DB", str(tmp_path / "memory.db"))
    mr.reset_service()
    return mr.get_service()


@pytest.fixture()
def client(service, tmp_path: Path, monkeypatch) -> TestClient:
    import ppmlx.memory_store as ms

    monkeypatch.setattr(ms, "_default_memory_db_path", lambda: tmp_path / "memory.db")
    from ppmlx.memory_read_routes import router

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def make_grant(service: MemoryReadService, *, scopes=None, tools=("memory_search", "memory_stats"), remote=False):
    return service.create_grant(
        harness_name="codex",
        harness_version="0.147.0",
        instance_id="hri_test",
        allowed_scopes=scopes or [{"type": "project", "id": "proj_a"}],
        allowed_tools=list(tools),
        remote_capable=remote,
    )


def auth_headers(credential: str, session_id: str | None):
    h = {
        "PPMLX-Memory-Version": "memory-read/v1",
        "Authorization": f"Bearer {credential}",
    }
    if session_id:
        h["PPMLX-Memory-Session"] = session_id
    return h


def seed_candidate(store_path: Path, text: str, project_id: str, disclosure: str | None = None, event_suffix: str = "") -> None:
    import threading

    store = MemoryStore(store_path)
    event_id = f"evt_{project_id}_{abs(hash(text + event_suffix)) % 10_000_000}"
    metadata: dict = {"project_id": project_id}
    if disclosure:
        metadata["disclosure"] = disclosure
    store.record_event({
        "event_id": event_id,
        "endpoint": "/v1/chat/completions",
        "app_id": None,
        "project_id": project_id,
        "session_id": None,
        "model_alias": "m",
        "model_repo": "r/m",
        "request": {},
        "response_text": text,
        "metadata": metadata,
    })
    store.store_candidate(
        {
            "candidate_id": f"cand_{event_id}",
            "event_id": event_id,
            "type": "fact",
            "subject": "s",
            "predicate": "uses",
            "object": text,
            "text": text,
            "scope": "project",
            "confidence": 0.9,
            "metadata": metadata,
        },
        {"status": "active", "reasons": [], "invalidates": []},
    )


# ---------------------------------------------------------------------------
# Handshake / auth
# ---------------------------------------------------------------------------

def test_unauthenticated_requests_are_rejected(client):
    for path in ("/v1/memory/read/handshake", "/v1/memory/read/search", "/v1/memory/read/stats"):
        resp = client.post(path, json={"version": "memory-read/v1", "request_id": "r1"})
        assert resp.status_code == 401, path
        body = resp.json()
        assert body["error"]["code"] == "credential_required"


def test_wrong_version_rejected(client):
    grant, cred = make_grant(mr.get_service())
    resp = client.post(
        "/v1/memory/read/handshake",
        headers={"PPMLX-Memory-Version": "memory-read/v0", "Authorization": f"Bearer {cred}"},
        json={"version": "memory-read/v1", "request_id": "r1"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "version_unsupported"


def test_handshake_returns_session_envelope(client, service):
    grant, cred = make_grant(service)
    resp = client.post(
        "/v1/memory/read/handshake",
        headers=auth_headers(cred, None),
        json={"version": "memory-read/v1", "request_id": "r1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "memory_read_session"
    assert body["read_session_id"].startswith("mrs_")
    assert body["grant_id"] == grant.grant_id
    assert body["harness"] == {"name": "codex", "version": "0.147.0", "instance_id": "hri_test"}
    assert "memory_search" in body["allowed_tools"]
    assert body["allowed_scopes"] == [{"type": "project", "id": "proj_a"}]
    assert body["session_expires_at"].endswith("Z")
    assert cred not in json.dumps(body)


def test_bad_credential_rejected(client, service):
    make_grant(service)
    resp = client.post(
        "/v1/memory/read/handshake",
        headers=auth_headers("mrc_wrong", None),
        json={"version": "memory-read/v1", "request_id": "r1"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "credential_invalid"


def test_session_ttl_expiry(client, service):
    grant, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    sid = env["read_session_id"]
    # Force expiry.
    service._sessions[sid].expires_at = time.time() - 1
    resp = client.post(
        "/v1/memory/read/stats",
        headers=auth_headers(cred, sid),
        json={"version": "memory-read/v1", "request_id": "r2", "scope": {"type": "project", "id": "proj_a"}, "parameters": {}, "limit": 20, "cursor": None},
    )
    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "session_expired"


def test_revocation_kills_live_sessions(client, service):
    grant, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    payload = {
        "version": "memory-read/v1", "request_id": "r3",
        "scope": {"type": "project", "id": "proj_a"}, "parameters": {}, "limit": 20, "cursor": None,
    }
    ok = client.post("/v1/memory/read/stats", headers=auth_headers(cred, env["read_session_id"]), json=payload)
    assert ok.status_code == 200
    assert service.revoke_grant(grant.grant_id)
    denied = client.post("/v1/memory/read/stats", headers=auth_headers(cred, env["read_session_id"]), json=payload)
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "credential_revoked"


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------

def test_scope_denied_for_ungranted_project(client, service):
    _, cred = make_grant(service)  # only proj_a granted
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred, env["read_session_id"]),
        json={
            "version": "memory-read/v1", "request_id": "r4",
            "scope": {"type": "project", "id": "proj_b"},
            "parameters": {"query": "kubernetes"}, "limit": 20, "cursor": None,
        },
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "scope_denied"
    # Scope name never leaks into the message.
    assert "proj_b" not in json.dumps(resp.json())


def test_results_scoped_to_granted_project_only(client, service, tmp_path):
    store_path = tmp_path / "memory.db"
    seed_candidate(store_path, "deploys kubernetes on Fridays", "proj_a")
    seed_candidate(store_path, "runs postgres in proj bee", "proj_b")
    _, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred, env["read_session_id"]),
        json={
            "version": "memory-read/v1", "request_id": "r5",
            "scope": {"type": "project", "id": "proj_a"},
            "parameters": {"query": "kubernetes"}, "limit": 20, "cursor": None,
        },
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    texts = [i["text"] for i in items]
    assert any("kubernetes" in t for t in texts)
    assert all("postgres" not in t for t in texts)


# ---------------------------------------------------------------------------
# Disclosure labels
# ---------------------------------------------------------------------------

def test_disclosure_filtering_secret_dropped_remote_gated(client, service, tmp_path):
    store_path = tmp_path / "memory.db"
    seed_candidate(store_path, "plain team fact about deploys", "proj_a", disclosure="local_only")
    seed_candidate(store_path, "secret api key material fact", "proj_a", disclosure="secret")
    seed_candidate(store_path, "shareable docs link fact", "proj_a", disclosure="remote_allowed")
    _, cred = make_grant(service, remote=False)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)

    def do_search(rid):
        return client.post(
            "/v1/memory/read/search",
            headers=auth_headers(cred, env["read_session_id"]),
            json={
                "version": "memory-read/v1", "request_id": rid,
                "scope": {"type": "project", "id": "proj_a"},
                "parameters": {"query": "fact"}, "limit": 50, "cursor": None,
            },
        )

    body = do_search("r6").json()
    texts = [i["text"] for i in body["items"]]
    assert any("team fact" in t for t in texts)
    assert not any("api key" in t for t in texts)          # secret never returned
    assert not any("docs link" in t for t in texts)         # remote_allowed gated off

    # Every returned item carries provenance.trust = untrusted and a label.
    for item in body["items"]:
        assert item["provenance"]["trust"] == "untrusted"

    # Remote-capable grant sees remote_allowed items, secrets stay hidden.
    _, cred_r = make_grant(service, scopes=[{"type": "project", "id": "proj_a"}], remote=True)
    env_r = service.handshake(credential=cred_r, version="memory-read/v1", is_loopback=True)
    body_r = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred_r, env_r["read_session_id"]),
        json={
            "version": "memory-read/v1", "request_id": "r7",
            "scope": {"type": "project", "id": "proj_a"},
            "parameters": {"query": "fact"}, "limit": 50, "cursor": None,
        },
    ).json()
    texts_r = [i["text"] for i in body_r["items"]]
    assert any("docs link" in t for t in texts_r)
    assert not any("api key" in t for t in texts_r)
    assert all(i["provenance"]["trust"] == "untrusted" for i in body_r["items"])


def test_default_disclosure_label_is_local_only(service):
    grant, _ = make_grant(service, remote=True)
    item = service.filter_item({"candidate_id": "c1", "text": "x", "event_id": "e1", "metadata_json": "{}"}, grant)
    assert item is not None
    assert item["disclosure"] == "local_only"


# ---------------------------------------------------------------------------
# Feedback-loop guard
# ---------------------------------------------------------------------------

def test_recapture_of_read_output_is_dropped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    text = "The deploy window is Friday 14:00 UTC."
    store.note_read_outputs([text])
    stored = store.record_event({
        "event_id": "evt_echo",
        "response_text": text,
        "metadata": {},
    })
    assert stored is False
    rows = store.query_events(limit=10)
    assert all(r["event_id"] != "evt_echo" for r in rows)


def test_echo_tagged_event_is_skipped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    stored = store.record_event({
        "event_id": "evt_echo2",
        "response_text": "something else entirely",
        "metadata": {"source": "memory_read_echo"},
    })
    assert stored is False


def test_normal_capture_still_stored_after_reads(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.note_read_outputs(["known read output"])
    assert store.record_event({"event_id": "e_ok", "response_text": "fresh user content", "metadata": {}}) is True


def test_search_notes_read_outputs_for_guard(client, service, tmp_path):
    store_path = tmp_path / "memory.db"
    seed_candidate(store_path, "unique zebra deployment policy", "proj_a")
    _, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred, env["read_session_id"]),
        json={
            "version": "memory-read/v1", "request_id": "r8",
            "scope": {"type": "project", "id": "proj_a"},
            "parameters": {"query": "zebra"}, "limit": 20, "cursor": None,
        },
    )
    assert resp.status_code == 200
    # The exact read output re-captured through record_event must be dropped.
    store = MemoryStore(store_path)
    assert store.record_event({
        "event_id": "evt_replay",
        "response_text": "unique zebra deployment policy",
        "metadata": {},
    }) is False


# ---------------------------------------------------------------------------
# Stats + misc contract behavior
# ---------------------------------------------------------------------------

def test_stats_endpoint_returns_stat_items(client, service, tmp_path):
    seed_candidate(tmp_path / "memory.db", "some fact for stats", "proj_a")
    _, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    resp = client.post(
        "/v1/memory/read/stats",
        headers=auth_headers(cred, env["read_session_id"]),
        json={"version": "memory-read/v1", "request_id": "r9", "scope": {"type": "project", "id": "proj_a"}, "parameters": {}, "limit": 20, "cursor": None},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "memory_stats"
    names = [i["name"] for i in body["items"]]
    assert names == sorted(names)
    assert all(set(i) >= {"type", "item_id", "name", "value", "unit"} for i in body["items"])


def test_tool_denied_and_limit_validation(client, service):
    grant, cred = make_grant(service, tools=("memory_stats",))
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    scope = {"type": "project", "id": "proj_a"}
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred, env["read_session_id"]),
        json={"version": "memory-read/v1", "request_id": "r10", "scope": scope,
              "parameters": {"query": "x"}, "limit": 20, "cursor": None},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "tool_denied"

    _, cred2 = make_grant(service)
    env2 = service.handshake(credential=cred2, version="memory-read/v1", is_loopback=True)
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred2, env2["read_session_id"]),
        json={"version": "memory-read/v1", "request_id": "r11", "scope": scope,
              "parameters": {"query": "x"}, "limit": 1000, "cursor": None},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_and_invalidated_by_tamper(client, service, tmp_path):
    store_path = tmp_path / "memory.db"
    for n in range(5):
        seed_candidate(store_path, f"pagination filler fact number {n}", "proj_a", event_suffix=str(n))
    _, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    scope = {"type": "project", "id": "proj_a"}
    params = {"query": "filler"}

    def do(rid, cursor=None):
        return client.post(
            "/v1/memory/read/search",
            headers=auth_headers(cred, env["read_session_id"]),
            json={"version": "memory-read/v1", "request_id": rid, "scope": scope,
                  "parameters": params, "limit": 2, "cursor": cursor},
        ).json()

    page1 = do("c1")
    assert len(page1["items"]) == 2 and page1["has_more"] and page1["next_cursor"]
    page2 = do("c2", page1["next_cursor"])
    ids1 = {i["item_id"] for i in page1["items"]}
    ids2 = {i["item_id"] for i in page2["items"]}
    assert not (ids1 & ids2)

    tampered = page1["next_cursor"][:-1] + ("0" if page1["next_cursor"][-1] != "0" else "1")
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred, env["read_session_id"]),
        json={"version": "memory-read/v1", "request_id": "c3", "scope": scope,
              "parameters": params, "limit": 2, "cursor": tampered},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "cursor_invalid"


def test_request_id_reuse_with_different_input_is_validation_error(client, service):
    _, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    headers = auth_headers(cred, env["read_session_id"])
    base = {"version": "memory-read/v1", "request_id": "dup", "scope": {"type": "project", "id": "proj_a"},
            "parameters": {"query": "alpha"}, "limit": 20, "cursor": None}
    ok = client.post("/v1/memory/read/search", headers=headers, json=base)
    assert ok.status_code == 200
    changed = dict(base, parameters={"query": "beta"})
    dup = client.post("/v1/memory/read/search", headers=headers, json=changed)
    assert dup.status_code == 400
    assert dup.json()["error"]["code"] == "validation_error"


def test_error_messages_leak_nothing(client, service):
    _, cred = make_grant(service)
    env = service.handshake(credential=cred, version="memory-read/v1", is_loopback=True)
    resp = client.post(
        "/v1/memory/read/search",
        headers=auth_headers(cred, env["read_session_id"]),
        json={"version": "memory-read/v1", "request_id": "z1",
              "scope": {"type": "project", "id": "other_proj"},
              "parameters": {"query": "super secret query terms"}, "limit": 20, "cursor": None},
    )
    raw = json.dumps(resp.json())
    assert "super secret query terms" not in raw
    assert cred not in raw
