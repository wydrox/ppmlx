from __future__ import annotations

from types import SimpleNamespace


def _cfg(**overrides):
    defaults = {
        "enabled": True,
        "provider": "posthog",
        "host": "https://stats.example.com",
        "project_api_key": "phc_test_123",
        "respect_do_not_track": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(analytics=SimpleNamespace(**defaults))


def test_track_skips_when_sink_not_configured(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg(host="", project_api_key=""))
    assert track("serve_started") is False


def test_track_respects_do_not_track(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    called = {"value": False}

    def fake_post(*args, **kwargs):
        called["value"] = True
        raise AssertionError("_post_capture should not be called when DNT=1")

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._post_capture", fake_post)
    monkeypatch.setenv("DNT", "1")

    assert track("serve_started") is False
    assert called["value"] is False


def test_track_posts_minimal_posthog_event(monkeypatch, tmp_home):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    captured = {}

    def fake_post_capture(host, project_api_key, event, properties):
        captured["host"] = host
        captured["project_api_key"] = project_api_key
        captured["event"] = event
        captured["properties"] = properties
        return True

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._post_capture", fake_post_capture)
    monkeypatch.setattr("ppmlx.analytics._anonymous_distinct_id", lambda: "ppmlx-test-id")

    ok = track("serve_started", {"interactive": True, "ignored": "secret-string"}, context="server")

    assert ok is True
    assert captured["host"] == "https://stats.example.com"
    assert captured["project_api_key"] == "phc_test_123"
    assert captured["event"] == "serve_started"
    properties = captured["properties"]
    assert properties["distinct_id"] == "ppmlx-test-id"
    assert properties["interactive"] is True
    assert properties["context_server"] is True
    assert properties["$process_person_profile"] is False
    assert properties["platform"]
    assert properties["arch"]
    assert "ignored" not in properties


def test_track_error_sends_only_safe_error_fields(monkeypatch):
    from ppmlx.analytics import track_error

    captured = {}

    def fake_track_async(event, data=None, *, context="cli"):
        captured["event"] = event
        captured["data"] = data
        captured["context"] = context

    monkeypatch.setattr("ppmlx.analytics.track_async", fake_track_async)

    track_error(
        context="cli",
        command="pull",
        error_type="ModelNotFoundError",
        exit_code=1,
        error_stage="download",
    )

    assert captured == {
        "event": "cli_error",
        "context": "cli",
        "data": {
            "command": "pull",
            "error_type": "ModelNotFoundError",
            "exit_code": 1,
            "error_stage": "download",
        },
    }


def test_memory_observability_sends_safe_enum_but_not_namespace_ids(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    captured = {}

    def fake_post_capture(host, project_api_key, event, properties):
        captured.update(properties)
        return True

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._post_capture", fake_post_capture)
    monkeypatch.setattr("ppmlx.analytics._anonymous_distinct_id", lambda: "ppmlx-test-id")

    assert track(
        "memory_context_reduction",
        {
            "memory_mode": "compact",
            "original_tokens": 100000,
            "reduced_tokens": 9000,
            "compression_ratio": 11.1,
            "has_project_id": True,
            "project_id": "private-project-name",
            "session_id": "private-session-id",
        },
        context="server",
    )

    assert captured["memory_mode"] == "compact"
    assert captured["original_tokens"] == 100000
    assert captured["has_project_id"] is True
    assert "project_id" not in captured
    assert "session_id" not in captured


def test_string_properties_are_allowlisted(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    captured = {}

    def fake_post_capture(host, project_api_key, event, properties):
        captured.update(properties)
        return True

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._post_capture", fake_post_capture)
    monkeypatch.setattr("ppmlx.analytics._anonymous_distinct_id", lambda: "ppmlx-test-id")

    assert track(
        "cli_error",
        {
            "command": "pull",
            "error_type": "ModelNotFoundError",
            "unsafe_message": "could contain a path or prompt",
        },
    )

    assert captured["command"] == "pull"
    assert captured["error_type"] == "ModelNotFoundError"
    assert "unsafe_message" not in captured
