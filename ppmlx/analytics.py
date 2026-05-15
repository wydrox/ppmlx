from __future__ import annotations

import os
import platform
import threading
import uuid
from typing import Any

import httpx

from ppmlx import __version__
from ppmlx.config import get_ppmlx_dir


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


_SAFE_STRING_KEYS = {
    "command",
    "endpoint",
    "error_type",
    "error_stage",
    "memory_mode",
}


def _sanitize_value(key: str, value: Any) -> int | float | bool | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if key in _SAFE_STRING_KEYS and isinstance(value, str):
        safe = "".join(ch for ch in value if ch.isalnum() or ch in "_-/.:").strip()
        return safe[:80] or None
    return None


def _sanitize_data(data: dict[str, Any] | None) -> dict[str, int | float | bool | str]:
    if not data:
        return {}
    cleaned: dict[str, int | float | bool | str] = {}
    for key, value in data.items():
        safe = _sanitize_value(key, value)
        if safe is not None:
            cleaned[key] = safe
    return cleaned


def _anonymous_distinct_id() -> str:
    """Return a stable anonymous install id for coarse adoption counts."""
    path = get_ppmlx_dir() / ".analytics_id"
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except Exception:
        pass

    new_id = f"ppmlx-{uuid.uuid4().hex}"
    try:
        path.write_text(new_id)
    except Exception:
        pass
    return new_id


def _get_settings() -> tuple[bool, str, str]:
    try:
        from ppmlx.config import load_config

        cfg = load_config()
        analytics = getattr(cfg, "analytics", None)
    except Exception:
        return False, "", ""

    enabled = _truthy(getattr(analytics, "enabled", True), default=True)
    host = _string(getattr(analytics, "host", ""))
    project_api_key = _string(getattr(analytics, "project_api_key", ""))
    provider = _string(getattr(analytics, "provider", "posthog")) or "posthog"
    respect_dnt = _truthy(getattr(analytics, "respect_do_not_track", True), default=True)

    if provider != "posthog" or not enabled or not host or not project_api_key:
        return False, "", ""
    if respect_dnt and os.environ.get("DNT") == "1":
        return False, "", ""
    return True, host.rstrip("/"), project_api_key


def _payload(data: dict[str, Any] | None) -> dict[str, Any]:
    clean: dict[str, Any] = {
        "distinct_id": _anonymous_distinct_id(),
        "version": __version__,
        "python": ".".join(platform.python_version_tuple()[:2]),
        "platform": platform.system().lower(),
        "arch": platform.machine().lower(),
        # Keep events personless so PostHog does not create user profiles.
        "$process_person_profile": False,
    }
    clean.update(_sanitize_data(data))
    return clean


def _post_capture(host: str, project_api_key: str, event: str, properties: dict[str, Any]) -> bool:
    response = httpx.post(
        f"{host}/capture/",
        json={
            "api_key": project_api_key,
            "event": event,
            "properties": properties,
        },
        timeout=1.5,
    )
    return 200 <= response.status_code < 300


def track(event: str, data: dict[str, Any] | None = None, *, context: str = "cli") -> bool:
    enabled, host, project_api_key = _get_settings()
    if not enabled:
        return False

    try:
        properties = _payload({**(data or {}), "context_cli": context == "cli", "context_server": context == "server"})
        return _post_capture(host, project_api_key, event, properties)
    except Exception:
        return False


def track_async(event: str, data: dict[str, Any] | None = None, *, context: str = "cli") -> None:
    thread = threading.Thread(
        target=track,
        kwargs={"event": event, "data": data, "context": context},
        daemon=True,
        name=f"ppmlx-analytics-{event}",
    )
    thread.start()


def track_error(
    *,
    context: str,
    error_type: str,
    command: str | None = None,
    endpoint: str | None = None,
    status_code: int | None = None,
    exit_code: int | None = None,
    error_stage: str | None = None,
) -> None:
    data: dict[str, Any] = {"error_type": error_type}
    if command:
        data["command"] = command
    if endpoint:
        data["endpoint"] = endpoint
    if status_code is not None:
        data["status_code"] = status_code
    if exit_code is not None:
        data["exit_code"] = exit_code
    if error_stage:
        data["error_stage"] = error_stage
    track_async("cli_error" if context == "cli" else "api_error", data, context=context)
