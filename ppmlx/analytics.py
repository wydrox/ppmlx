from __future__ import annotations

import os
import platform
import threading
from typing import Any

from ppmlx import __version__

try:
    from posthog import Posthog as _Posthog
except ImportError:
    _Posthog = None  # type: ignore[assignment]

Posthog = _Posthog


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sanitize_value(value: Any) -> int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return None


def _sanitize_data(data: dict[str, Any] | None) -> dict[str, int | float | bool]:
    if not data:
        return {}
    cleaned: dict[str, int | float | bool] = {}
    for key, value in data.items():
        safe = _sanitize_value(value)
        if safe is not None:
            cleaned[key] = safe
    return cleaned


_CLIENT_LOCK = threading.Lock()
_CLIENT_CACHE: tuple[str, str, Any] | None = None


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
        "version": __version__,
        "python": ".".join(platform.python_version_tuple()[:2]),
        "platform": platform.system().lower(),
        "arch": platform.machine().lower(),
        # Keep events personless so the SDK does not create user profiles.
        "$process_person_profile": False,
    }
    clean.update(_sanitize_data(data))
    return clean


def _get_client(host: str, project_api_key: str) -> Any:
    if Posthog is None:
        return None

    global _CLIENT_CACHE
    cached = _CLIENT_CACHE
    if cached and cached[0] == host and cached[1] == project_api_key:
        return cached[2]

    with _CLIENT_LOCK:
        cached = _CLIENT_CACHE
        if cached and cached[0] == host and cached[1] == project_api_key:
            return cached[2]

        client = Posthog(
            project_api_key,
            host=host,
            sync_mode=True,
            timeout=1,
            disable_geoip=True,
        )
        _CLIENT_CACHE = (host, project_api_key, client)
        return client


def track(event: str, data: dict[str, Any] | None = None, *, context: str = "cli") -> bool:
    enabled, host, project_api_key = _get_settings()
    if not enabled:
        return False

    try:
        capture_id = _get_client(host, project_api_key).capture(
            event,
            properties=_payload(data),
        )
        return bool(capture_id)
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
