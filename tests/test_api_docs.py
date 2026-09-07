"""Tests for ppmlx.api_docs — Interactive API Playground & Documentation."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock ppmlx modules that server.py imports lazily (same pattern as test_server.py)
for mod in [
    "ppmlx.engine", "ppmlx.engine_vlm", "ppmlx.engine_embed",
    "ppmlx.models", "ppmlx.db", "ppmlx.config", "ppmlx.memory",
    "ppmlx.schema",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Save original module attributes so they can be restored by conftest's
# _restore_module_attrs fixture.  We only set mocks on modules that are
# already real (pre-imported by conftest) to avoid polluting imports in
# other test files that capture names at collection time.
_originals: dict[str, dict[str, object]] = {}
def _save_and_set(mod_name: str, attr: str, value: object) -> None:
    mod = sys.modules[mod_name]
    _originals.setdefault(mod_name, {})[attr] = getattr(mod, attr, None)
    setattr(mod, attr, value)

# Set up minimal mocks so the app can start
mock_engine = MagicMock()
mock_engine.list_loaded.return_value = []
mock_engine.generate.return_value = ("Hello!", None, 10, 5)
_save_and_set("ppmlx.engine", "get_engine", MagicMock(return_value=mock_engine))

mock_db = MagicMock()
mock_db.get_stats.return_value = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}
_save_and_set("ppmlx.db", "get_db", MagicMock(return_value=mock_db))

_save_and_set("ppmlx.memory", "get_system_ram_gb", MagicMock(return_value=16.0))

mock_config = MagicMock()
mock_config.logging.snapshot_interval_seconds = 60
_save_and_set("ppmlx.config", "load_config", MagicMock(return_value=mock_config))

_save_and_set("ppmlx.models", "resolve_alias", MagicMock(side_effect=lambda x: x))
_save_and_set("ppmlx.models", "list_local_models", MagicMock(return_value=[]))
_save_and_set("ppmlx.models", "all_aliases", MagicMock(return_value=[]))
_save_and_set("ppmlx.models", "is_vision_model", MagicMock(return_value=False))
_save_and_set("ppmlx.models", "is_embed_model", MagicMock(return_value=False))

import pytest
from fastapi.testclient import TestClient
from ppmlx.server import app

# Restore original module attributes immediately after importing the server
# app.  The mocks above are only needed for server.py's module-level init;
# leaving them in place would pollute later-collected test files (e.g.
# test_config.py) that capture names via ``from ppmlx.X import Y``.
for _mod_name, _attrs in _originals.items():
    _mod = sys.modules.get(_mod_name)
    if _mod is not None:
        for _attr, _val in _attrs.items():
            if _val is not None:
                setattr(_mod, _attr, _val)
del _mod_name, _attrs, _mod, _attr, _val


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Playground page ───────────────────────────────────────────────────


def test_playground_returns_200(client):
    """GET /playground should return an HTML page."""
    response = client.get("/playground")
    assert response.status_code == 200


def test_playground_is_html(client):
    """The playground page should have HTML content type."""
    response = client.get("/playground")
    assert "text/html" in response.headers.get("content-type", "")


def test_playground_contains_title(client):
    """The playground HTML should contain the ppmlx branding."""
    response = client.get("/playground")
    assert "ppmlx" in response.text
    assert "API Playground" in response.text


def test_playground_contains_endpoints(client):
    """The playground HTML should embed the endpoint list."""
    response = client.get("/playground")
    html = response.text
    assert "/v1/chat/completions" in html
    assert "/v1/models" in html
    assert "/health" in html
    assert "/v1/embeddings" in html


def test_playground_contains_code_tabs(client):
    """The playground should have code generation tabs for all languages."""
    response = client.get("/playground")
    html = response.text
    assert "curl" in html
    assert "Python" in html
    assert "JavaScript" in html
    assert "TypeScript" in html


def test_playground_contains_send_button(client):
    """The playground should have a send button."""
    response = client.get("/playground")
    assert "Send" in response.text


def test_playground_has_streaming_support(client):
    """The playground HTML should reference streaming functionality."""
    response = client.get("/playground")
    html = response.text
    assert "streaming" in html.lower() or "Streaming" in html
    assert "EventSource" in html or "getReader" in html


def test_playground_has_dark_theme(client):
    """The playground should use a dark color scheme."""
    response = client.get("/playground")
    html = response.text
    # Check for dark background color in CSS
    assert "#0d1117" in html or "#161b22" in html


def test_playground_has_copy_button(client):
    """The playground should have a copy-to-clipboard button."""
    response = client.get("/playground")
    assert "copyCode" in response.text or "copy-btn" in response.text


# ── Endpoint metadata API ────────────────────────────────────────────


def test_playground_endpoints_api(client):
    """GET /api/playground/endpoints should return endpoint metadata."""
    response = client.get("/api/playground/endpoints")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)
    assert len(data["endpoints"]) > 0


def test_playground_endpoints_have_required_fields(client):
    """Each endpoint entry should have method, path, and description."""
    response = client.get("/api/playground/endpoints")
    data = response.json()
    for ep in data["endpoints"]:
        assert "method" in ep
        assert "path" in ep
        assert "description" in ep


def test_playground_endpoints_includes_chat(client):
    """The endpoint list should include the chat completions endpoint."""
    response = client.get("/api/playground/endpoints")
    data = response.json()
    paths = [(e["method"], e["path"]) for e in data["endpoints"]]
    assert ("POST", "/v1/chat/completions") in paths


def test_playground_endpoints_includes_health(client):
    """The endpoint list should include the health endpoint."""
    response = client.get("/api/playground/endpoints")
    data = response.json()
    paths = [(e["method"], e["path"]) for e in data["endpoints"]]
    assert ("GET", "/health") in paths


def test_playground_endpoints_excludes_playground(client):
    """The endpoint metadata should not include playground routes themselves."""
    response = client.get("/api/playground/endpoints")
    data = response.json()
    paths = [e["path"] for e in data["endpoints"]]
    for p in paths:
        assert not p.startswith("/playground")
        assert not p.startswith("/api/playground")


def test_playground_endpoints_has_server_url(client):
    """The response should include the server URL."""
    response = client.get("/api/playground/endpoints")
    data = response.json()
    assert "server_url" in data
    assert "http" in data["server_url"]


# ── Internal functions ────────────────────────────────────────────────


def test_example_requests_defined():
    """Verify that example requests are defined for key endpoints."""
    from ppmlx.api_docs import _EXAMPLE_REQUESTS
    assert "POST /v1/chat/completions" in _EXAMPLE_REQUESTS
    assert "POST /v1/embeddings" in _EXAMPLE_REQUESTS
    assert "GET /v1/models" in _EXAMPLE_REQUESTS
    assert "GET /health" in _EXAMPLE_REQUESTS


def test_endpoint_docs_defined():
    """Verify that endpoint documentation is defined for key endpoints."""
    from ppmlx.api_docs import _ENDPOINT_DOCS
    assert "POST /v1/chat/completions" in _ENDPOINT_DOCS
    assert "POST /v1/embeddings" in _ENDPOINT_DOCS
    assert "GET /health" in _ENDPOINT_DOCS


def test_discover_endpoints():
    """The endpoint discovery function should find the app routes."""
    from ppmlx.api_docs import _discover_endpoints
    endpoints = _discover_endpoints(app)
    assert isinstance(endpoints, list)
    assert len(endpoints) > 0
    methods_paths = [(e["method"], e["path"]) for e in endpoints]
    assert ("GET", "/health") in methods_paths
    assert ("POST", "/v1/chat/completions") in methods_paths


def test_discover_endpoints_excludes_playground():
    """Discovery should skip playground routes."""
    from ppmlx.api_docs import _discover_endpoints
    endpoints = _discover_endpoints(app)
    for ep in endpoints:
        assert not ep["path"].startswith("/playground")
        assert not ep["path"].startswith("/api/playground")


def test_playground_html_is_valid():
    """The HTML generator should return a non-empty string with doctype."""
    from ppmlx.api_docs import _get_playground_html
    html = _get_playground_html()
    assert isinstance(html, str)
    assert len(html) > 1000  # should be a substantial page
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_playground_html_contains_example_data():
    """The HTML should embed the example request data as JSON."""
    from ppmlx.api_docs import _get_playground_html
    html = _get_playground_html()
    assert "llama3" in html
    assert "nomic-embed" in html
    assert "Hello" in html
