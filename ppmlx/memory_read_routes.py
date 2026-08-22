"""HTTP routes for the memory-read/v1 wire contract (ADR 0006).

Mounted on the existing FastAPI app in ``server.py`` behind the same
loopback-only constraint as other strict local paths. Minimal slice:
``/handshake``, ``/search``, ``/stats``.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ppmlx.memory_read import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MEMORY_READ_VERSION,
    MemoryReadError,
    get_service,
)

router = APIRouter(prefix="/v1/memory/read", tags=["memory-read"])


def _error_response(request_id: str | None, err: MemoryReadError) -> JSONResponse:
    # Error messages carry no memory text, credentials, query, or scope names.
    return JSONResponse(
        status_code=err.status,
        content={
            "version": MEMORY_READ_VERSION,
            "object": "error",
            "request_id": request_id,
            "error": {"code": err.code, "message": _GENERIC_MESSAGES.get(err.code, "The request failed."), "retryable": err.retryable},
        },
    )


_GENERIC_MESSAGES = {
    "permission_denied": "Permission denied.",
    "credential_required": "A credential is required.",
    "credential_invalid": "The credential is not valid.",
    "credential_expired": "The credential has expired.",
    "credential_revoked": "The credential has been revoked.",
    "scope_denied": "The scope is not permitted.",
    "tool_denied": "The tool is not permitted.",
    "session_required": "A read session is required.",
    "session_invalid": "The read session is not valid.",
    "session_expired": "The read session has expired.",
    "cursor_expired": "The cursor has expired.",
    "validation_error": "The request is not valid.",
    "cursor_invalid": "The cursor is not valid.",
    "version_unsupported": "The contract version is not supported.",
    "memory_unavailable": "Memory is unavailable.",
    "rate_limited": "Too many requests.",
    "internal_error": "An internal error occurred.",
}


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client is not None else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost", "testclient"}


def _headers(request: Request) -> dict[str, str | None]:
    auth = request.headers.get("authorization") or ""
    credential = auth[len("Bearer "):] if auth.startswith("Bearer ") else None
    return {
        "version": request.headers.get("ppmlx-memory-version"),
        "credential": credential,
        "session_id": request.headers.get("ppmlx-memory-session"),
    }


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        raise MemoryReadError("validation_error")
    if not isinstance(data, dict):
        raise MemoryReadError("validation_error")
    return data


def _validate_envelope(data: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any], int, str | None]:
    if data.get("version") != MEMORY_READ_VERSION:
        raise MemoryReadError("version_unsupported")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise MemoryReadError("validation_error")
    limit = data.get("limit", DEFAULT_LIMIT)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > MAX_LIMIT:
        raise MemoryReadError("validation_error")
    scope = data.get("scope")
    params = data.get("parameters", {})
    if not isinstance(params, dict):
        raise MemoryReadError("validation_error")
    cursor = data.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise MemoryReadError("cursor_invalid")
    return request_id, scope if isinstance(scope, dict) else {}, params, limit, cursor


def _search_item(row: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    created_at = row.get("created_at")
    valid_from = row.get("valid_from")
    valid_to = row.get("valid_to")
    confidence = row.get("confidence")
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    return {
        "type": "memory",
        "item_id": row.get("candidate_id"),
        "text": row.get("text"),
        "scope": {"type": scope["type"], "id": scope["id"]},
        "provenance": {
            "origin": metadata.get("origin", "harness"),
            "origin_id": row.get("event_id"),
            "trust": "untrusted",
        },
        "sensitivity": metadata.get("disclosure", "local_only"),
        "observed_at": created_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "status": row.get("status"),
        "confidence": float(confidence) if confidence is not None else 0.0,
        "_row": row,
    }


@router.post("/handshake")
async def handshake(request: Request):
    try:
        h = _headers(request)
        body = await _body(request)
        service = get_service()
        envelope = service.handshake(
            credential=h["credential"],
            version=h["version"],
            is_loopback=_is_loopback(request),
        )
        return envelope
    except MemoryReadError as err:
        return _error_response(None, err)


@router.post("/search")
async def memory_search(request: Request):
    request_id: str | None = None
    try:
        h = _headers(request)
        body = await _body(request)
        request_id, raw_scope, params, limit, cursor_token = _validate_envelope(body)
        service = get_service()
        grant, session = service.authenticate(**h)
        service.check_tool(grant, "memory_search")
        scope = service.resolve_scope(grant, raw_scope)
        query = params.get("query")
        if not isinstance(query, str) or not (1 <= len(query) <= 2000):
            raise MemoryReadError("validation_error")
        canonical_params = {"query": query}
        fingerprint = hashlib.sha256(
            json.dumps({"t": "search", "s": scope, "p": canonical_params, "l": limit}, sort_keys=True).encode()
        ).hexdigest()
        service.check_request_id(session, request_id, fingerprint)

        offset = service.parse_cursor(
            cursor_token, grant_id=grant.grant_id, tool="memory_search",
            scope=scope, params=canonical_params,
        )

        from ppmlx.memory_store import MemoryStore

        store = MemoryStore()  # read path only; no capture methods exposed here
        kwargs: dict[str, Any] = {}
        if scope["type"] == "project":
            kwargs["project_id"] = scope["id"]
        elif scope["type"] == "app":
            kwargs["app_id"] = scope["id"]
        elif scope["type"] == "session":
            kwargs["session_id"] = scope["id"]

        fetch_limit = min(offset + limit + 1, 10_000)
        rows = store.search(query, status="active", limit=fetch_limit, **kwargs)
        items = []
        for row in rows:
            item = _search_item(row, scope)
            filtered = service.filter_item(item["_row"], grant)
            item.pop("_row", None)
            if filtered is None:
                continue  # disclosure filter dropped (e.g. secret): omit silently
            filtered.update({k: v for k, v in item.items() if k != "_row"})
            items.append(filtered)
        window = items[offset:offset + limit]
        has_more = len(items) > offset + limit
        next_cursor = None
        if has_more:
            from ppmlx.memory_read import CURSOR_TTL_SECONDS
            import time as _time

            next_cursor = service.make_cursor(
                grant_id=grant.grant_id, tool="memory_search", scope=scope,
                params=canonical_params, offset=offset + limit,
                expires_at=_time.time() + CURSOR_TTL_SECONDS,
            )
        texts = [str(i.get("text") or "") for i in items]
        service.note_read_outputs(texts)
        store.note_read_outputs(texts)  # feed the ingest-side echo guard
        return {
            "version": MEMORY_READ_VERSION,
            "object": "memory_read_result",
            "request_id": request_id,
            "tool": "memory_search",
            "scope": scope,
            "items": window,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }
    except MemoryReadError as err:
        return _error_response(request_id, err)
    except Exception:
        return _error_response(request_id, MemoryReadError("internal_error"))


@router.post("/stats")
async def memory_stats(request: Request):
    request_id: str | None = None
    try:
        h = _headers(request)
        body = await _body(request)
        request_id, raw_scope, params, limit, cursor_token = _validate_envelope(body)
        if params != {} or cursor_token is not None or limit != DEFAULT_LIMIT:
            raise MemoryReadError("validation_error")

        service = get_service()
        grant, session = service.authenticate(**h)
        service.check_tool(grant, "memory_stats")
        scope = service.resolve_scope(grant, raw_scope)
        service.check_request_id(session, request_id, "stats")

        from ppmlx.memory_store import MemoryStore

        stats = MemoryStore().stats()
        stat_defs = [
            ("atoms", stats.get("atoms", 0), "count"),
            ("candidates", stats.get("candidates", 0), "count"),
            ("compactions", stats.get("compactions", 0), "count"),
            ("edges", stats.get("edges", 0), "count"),
            ("entities", stats.get("entities", 0), "count"),
            ("events", stats.get("events", 0), "count"),
            ("extraction_jobs", stats.get("extraction_jobs", 0), "count"),
            ("inferred", stats.get("inferred", 0), "count"),
        ]
        items = [
            {
                "type": "stat",
                "item_id": f"stat_{name}",
                "name": name,
                "value": value,
                "unit": unit,
                "scope": scope,
                "sensitivity": "local_only",
                "provenance": {"origin": "service", "origin_id": None, "trust": "untrusted"},
            }
            for name, value, unit in sorted(stat_defs)
        ]
        return {
            "version": MEMORY_READ_VERSION,
            "object": "memory_read_result",
            "request_id": request_id,
            "tool": "memory_stats",
            "scope": scope,
            "items": items,
            "has_more": False,
            "next_cursor": None,
        }
    except MemoryReadError as err:
        return _error_response(request_id, err)
    except Exception:
        return _error_response(request_id, MemoryReadError("internal_error"))
