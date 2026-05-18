"""Tests for the local memory graph view."""
from __future__ import annotations

from ppmlx.graph_view import render_graph_html


def test_render_graph_html_uses_pinned_antv_g6() -> None:
    html = render_graph_html({"status": "active", "limit": "120"})

    assert "https://cdn.jsdelivr.net/npm/@antv/g6@4.8.24/dist/g6.min.js" in html
    assert "new G6.Graph" in html
    assert "container: 'graph'" in html
    assert '<div id="graph-wrap"' in html
    assert "type: 'force'" in html
    assert "drag-canvas" in html
    assert "zoom-canvas" in html
    assert "drag-node" in html
    assert "Unable to load AntV G6 from the CDN" in html


def test_render_graph_html_does_not_include_manual_svg_ring_layout() -> None:
    html = render_graph_html()

    assert '<svg id="graph"' not in html
    assert "createElementNS" not in html
    assert "setAttribute('viewBox'" not in html
    assert "Math.cos" not in html
    assert "Math.sin" not in html


def test_render_graph_html_includes_curated_and_raw_ui_modes() -> None:
    html = render_graph_html({"mode": "raw"})

    assert '<label>Mode <select id="mode" aria-label="UI mode">' in html
    assert '<option value="curated">curated</option>' in html
    assert '<option value="raw">raw/debug</option>' in html
    assert "mode: <b>curated</b>" in html
    assert "mode: <b>raw/debug</b>" in html
    assert "['query','project_id','session_id','app_id','status','limit','mode']" in html
    assert "params.set(id, $(id).value || '')" in html


def test_render_graph_html_includes_polluted_data_diagnosis_text() -> None:
    html = render_graph_html()

    assert "Polluted data / chaos diagnosis" in html
    assert "Potential polluted-data / chaos signals detected." in html
    assert "No obvious polluted-data signals" in html
    assert "suspicious facts" in html
    assert "weak labels" in html
    assert "Details & evidence" in html


def test_render_graph_html_includes_live_refresh_and_new_memory_animation() -> None:
    html = render_graph_html()

    assert 'id="live"' in html
    assert "Live 1s" in html
    assert "setInterval(() => { if (!document.hidden) load({ auto: true }); }, 1000)" in html
    assert "snapshotFingerprint" in html
    assert "new-memory" in html
    assert "@keyframes pulseNew" in html
    assert "animateNewGraphItems" in html
