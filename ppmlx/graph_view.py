"""Local read-only web view for the ppmlx temporal memory graph."""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ppmlx.memory_store import MemoryStore


def serve_graph_view(
    store: MemoryStore,
    *,
    host: str = "127.0.0.1",
    port: int = 6777,
    status: str | None = "active",
    query: str | None = None,
    app_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    limit: int = 120,
    open_browser: bool = True,
    on_start: Callable[[str], None] | None = None,
) -> str:
    """Serve the local graph viewer until interrupted and return the URL."""
    defaults = {
        "status": status or "active",
        "query": query or "",
        "app_id": app_id or "",
        "project_id": project_id or "",
        "session_id": session_id or "",
        "limit": str(limit),
        "mode": "curated",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_text(render_graph_html(defaults), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/graph":
                params = _query_params(parsed.query, defaults)
                snapshot = store.graph_snapshot(
                    status=params["status"],
                    query=params["query"],
                    app_id=params["app_id"] or None,
                    project_id=params["project_id"] or None,
                    session_id=params["session_id"] or None,
                    limit=int(params["limit"]),
                )
                self._send_json(snapshot)
                return
            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            return

        def _send_json(self, data: dict[str, Any]) -> None:
            raw = json.dumps(data, ensure_ascii=False, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_text(self, text: str, *, content_type: str) -> None:
            raw = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = httpd.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    if on_start:
        on_start(url)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return url


def render_graph_html(defaults: dict[str, str] | None = None) -> str:
    defaults_json = json.dumps(defaults or {}, ensure_ascii=False)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ppmlx graph</title>
<style>
:root { color-scheme: dark; --bg:#0b0f14; --panel:#111923; --muted:#8493a7; --text:#e6edf3; --accent:#7dd3fc; --edge:#334155; --hot:#fbbf24; --danger:#fb7185; --ok:#86efac; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
header { display:flex; gap:16px; align-items:center; padding:14px 18px; border-bottom:1px solid #1f2937; background:#0d131b; position:sticky; top:0; z-index:2; }
h1 { font-size:18px; margin:0; letter-spacing:.02em; }
.badge { color:#08111a; background:var(--accent); border-radius:999px; padding:2px 8px; font-weight:700; }
.controls { display:grid; grid-template-columns:repeat(8, minmax(90px, 1fr)); gap:8px; padding:12px 18px; border-bottom:1px solid #1f2937; background:#0d131b; }
label { color:var(--muted); font-size:12px; display:flex; flex-direction:column; gap:4px; }
input, select, button { background:#0f1720; color:var(--text); border:1px solid #263244; border-radius:8px; padding:8px; }
button { cursor:pointer; background:#132033; }
main { display:grid; grid-template-columns:minmax(420px, 1fr) 420px; min-height:calc(100vh - 116px); }
#graph-wrap { position:relative; width:100%; height:calc(100vh - 116px); background:radial-gradient(circle at center, #101923 0, #0b0f14 70%); }
#graph { width:100%; height:100%; }
.graph-message { position:absolute; inset:16px auto auto 16px; max-width:min(520px, calc(100% - 32px)); background:#0d141ee6; border:1px solid #334155; border-radius:10px; color:var(--muted); padding:12px 14px; z-index:1; }
.graph-message.error { color:#fecaca; border-color:#ef4444; background:#190d12e6; }
aside { border-left:1px solid #1f2937; background:var(--panel); overflow:auto; height:calc(100vh - 116px); }
section { padding:14px 16px; border-bottom:1px solid #223044; }
h2 { font-size:13px; margin:0 0 8px; color:var(--accent); text-transform:uppercase; letter-spacing:.08em; }
.statgrid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.stat { background:#0d141e; border:1px solid #1f2b3d; border-radius:10px; padding:9px; }
.stat b { display:block; font-size:18px; }
.item { border:1px solid #1f2b3d; border-radius:10px; padding:10px; margin:8px 0; background:#0d141e; cursor:pointer; }
.item:hover { border-color:#3b82f6; }
.muted { color:var(--muted); }
.mode-note, .diagnosis { color:var(--muted); font-size:13px; }
.diagnosis ul { margin:8px 0 0; padding-left:18px; }
.diagnosis li { margin:4px 0; }
.pill { display:inline-block; border:1px solid #334155; border-radius:999px; padding:2px 8px; margin:2px 4px 2px 0; color:#cbd5e1; background:#0b1119; font-size:12px; }
.pill.warn { border-color:#92400e; color:#fde68a; }
.pill.danger { border-color:#9f1239; color:#fecdd3; }
.pill.ok { border-color:#166534; color:#bbf7d0; }
.detail-block { margin:0 0 10px; }
.detail-block b { color:#dbeafe; }
pre { white-space:pre-wrap; word-break:break-word; background:#0b1119; border:1px solid #223044; border-radius:10px; padding:10px; max-height:360px; overflow:auto; }
@media (max-width: 960px) { .controls { grid-template-columns:1fr 1fr; } main { grid-template-columns:1fr; } aside { height:auto; border-left:0; border-top:1px solid #1f2937; } #graph-wrap { height:60vh; } }
</style>
</head>
<body>
<header><h1>ppmlx graph</h1><span class="badge">local read-only</span><span class="muted" id="path"></span></header>
<div class="controls">
<label>Search <input id="query" placeholder="fact, entity, source quote" /></label>
<label>Project <input id="project_id" placeholder="project_id" /></label>
<label>Session <input id="session_id" placeholder="session_id" /></label>
<label>App <input id="app_id" placeholder="app_id" /></label>
<label>Status <select id="status"><option>active</option><option>all</option><option>disputed</option><option>rejected</option><option>superseded</option><option>forgotten</option></select></label>
<label>Mode <select id="mode" aria-label="UI mode"><option value="curated">curated</option><option value="raw">raw/debug</option></select></label>
<label>Limit <input id="limit" type="number" min="1" max="500" value="120" /></label>
<button id="refresh">Refresh</button>
</div>
<main>
<div id="graph-wrap" role="img" aria-label="temporal memory graph"><div id="graph-message" class="graph-message" hidden></div><div id="graph"></div></div>
<aside>
<section><h2>Stats</h2><div class="statgrid" id="stats"></div><p class="mode-note" id="mode-note"></p></section>
<section><h2>Polluted data / chaos diagnosis</h2><div class="diagnosis" id="diagnosis">Load a snapshot to diagnose graph noise.</div></section>
<section><h2>Details & evidence</h2><pre id="details">Click a node, edge, or fact to inspect details and evidence.</pre></section>
<section><h2>Facts</h2><div id="facts"></div></section>
<section><h2>Timeline</h2><div id="events"></div></section>
</aside>
</main>
<script src="https://cdn.jsdelivr.net/npm/@antv/g6@4.8.24/dist/g6.min.js" crossorigin="anonymous" onerror="window.__ppmlxG6LoadError = true"></script>
<script>
const defaults = __DEFAULTS_JSON__;
const $ = id => document.getElementById(id);
let graph = null;
for (const [k,v] of Object.entries(defaults)) if ($(k) && v) $(k).value = v;
$('refresh').onclick = load;
$('mode').addEventListener('change', load);
for (const id of ['query','project_id','session_id','app_id','status','limit']) $(id).addEventListener('keydown', e => { if (e.key === 'Enter') load(); });
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function modeLabel() { return $('mode').value === 'raw' ? 'raw/debug' : 'curated'; }
async function load() {
  const params = new URLSearchParams();
  for (const id of ['query','project_id','session_id','app_id','status','limit','mode']) params.set(id, $(id).value || '');
  const res = await fetch('/api/graph?' + params.toString());
  const data = await res.json();
  render(data);
}
function render(data) {
  const view = applyUiMode(data);
  const stats = data.stats || {};
  $('path').textContent = data.path || '';
  $('stats').innerHTML = [
    ['nodes', view.nodes.length], ['edges', view.edges.length], ['facts', view.candidates.length], ['events', view.events.length],
    ['db candidates', stats.candidates ?? '—'], ['db edges', stats.edges ?? '—']
  ].map(([k,v]) => `<div class="stat"><span class="muted">${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
  $('mode-note').innerHTML = modeLabel() === 'curated'
    ? 'UI mode: <b>curated</b> hides obvious rejected/forgotten/low-confidence noise in this browser only. /api/graph is unchanged.'
    : 'UI mode: <b>raw/debug</b> shows the snapshot as returned by /api/graph, with noise indicators left visible.';
  renderDiagnosis(data, view);
  renderGraph(view);
  $('facts').innerHTML = view.candidates.map(c => `<div class="item" data-id="${esc(c.candidate_id)}"><b>${esc(c.type)} · ${esc(c.subject)} → ${esc(c.predicate)} → ${esc(c.object)}</b><div>${esc(c.text)}</div><div class="muted">${esc(c.status)} · confidence ${esc(c.confidence)} · ${esc(c.project_id || '')}/${esc(c.session_id || '')}</div>${noisePills(c).join('')}</div>`).join('') || '<p class="muted">No facts for current filters.</p>';
  for (const el of document.querySelectorAll('#facts .item')) el.onclick = () => show(view.candidates.find(c => c.candidate_id === el.dataset.id));
  $('events').innerHTML = view.events.map(e => `<div class="item"><b>${esc(e.timestamp)}</b><div>${esc(e.endpoint || '')}</div><div class="muted">${esc(e.event_id)} · ${esc(e.project_id || '')}/${esc(e.session_id || '')}</div></div>`).join('') || '<p class="muted">No events.</p>';
}
function badText(value) {
  const text = String(value ?? '').trim();
  if (!text) return true;
  if (new RegExp('^(unknown|none|null|n/a|undefined|tbd)$', 'i').test(text)) return true;
  if (text.length > 120) return true;
  if (text.startsWith('{') || text.startsWith('[') || /```|<[^>]+>/.test(text)) return true;
  return false;
}
function candidateIssues(c) {
  const issues = [];
  const confidence = Number(c.confidence ?? 1);
  if (['rejected','forgotten'].includes(String(c.status || '').toLowerCase())) issues.push('non-curated status');
  if (!Number.isNaN(confidence) && confidence < 0.45) issues.push('low confidence');
  if (badText(c.subject) || badText(c.object) || badText(c.predicate)) issues.push('weak entity text');
  if (!String(c.source_quote || c.text || '').trim()) issues.push('missing evidence');
  return issues;
}
function nodeIssues(n) {
  const issues = [];
  if (badText(n.name || n.label || n.id)) issues.push('weak label');
  if (Number(n.degree || 0) === 0 && Number(n.candidate_count || 0) > 1) issues.push('unlinked repeat');
  if (Number(n.candidate_count || 0) > 8 && Number(n.degree || 0) <= 1) issues.push('possible blob entity');
  return issues;
}
function noisePills(obj) {
  const issues = obj && Object.prototype.hasOwnProperty.call(obj, 'candidate_id') ? candidateIssues(obj) : nodeIssues(obj || {});
  if (!issues.length) return ['<span class="pill ok">curated-looking</span>'];
  return issues.map(issue => `<span class="pill warn">${esc(issue)}</span>`);
}
function applyUiMode(data) {
  const source = {
    ...data,
    nodes: data.nodes || [],
    edges: data.edges || [],
    candidates: data.candidates || [],
    events: data.events || [],
  };
  if ($('mode').value === 'raw') return source;
  const candidates = source.candidates.filter(c => !candidateIssues(c).some(issue => ['non-curated status','low confidence','weak entity text'].includes(issue)));
  const keptCandidateIds = new Set(candidates.map(c => String(c.candidate_id || '')));
  const edges = source.edges.filter(e => !e.source_candidate_id || keptCandidateIds.has(String(e.source_candidate_id)));
  const connectedNodeIds = new Set(edges.flatMap(e => [e.source || e.from_entity_id, e.target || e.to_entity_id]).filter(Boolean).map(String));
  const candidateNames = new Set(candidates.flatMap(c => [c.subject, c.object]).filter(Boolean).map(v => String(v).trim().toLowerCase()));
  const nodes = source.nodes.filter(n => connectedNodeIds.has(String(n.id)) || candidateNames.has(String(n.name || n.label || '').trim().toLowerCase()));
  return { ...source, nodes, edges, candidates };
}
function renderDiagnosis(raw, view) {
  const rawCandidates = raw.candidates || [];
  const rawNodes = raw.nodes || [];
  const candidateNoise = rawCandidates.filter(c => candidateIssues(c).length);
  const nodeNoise = rawNodes.filter(n => nodeIssues(n).length);
  const hiddenFacts = rawCandidates.length - view.candidates.length;
  const hiddenNodes = rawNodes.length - view.nodes.length;
  const stats = raw.stats || {};
  const totalCandidates = Number(stats.candidates ?? rawCandidates.length) || 0;
  const edgeTotal = Number(stats.edges ?? (raw.edges || []).length) || 0;
  const density = rawNodes.length ? ((raw.edges || []).length / rawNodes.length).toFixed(2) : '0.00';
  const severity = candidateNoise.length || nodeNoise.length || totalCandidates > rawCandidates.length * 3 ? 'warn' : 'ok';
  const diagnosis = severity === 'ok' ? 'No obvious polluted-data signals in this filtered snapshot.' : 'Potential polluted-data / chaos signals detected.';
  $('diagnosis').innerHTML = `
    <div><span class="pill ${severity}">${diagnosis}</span><span class="pill">mode: ${esc(modeLabel())}</span></div>
    <ul>
      <li>Snapshot: ${esc(rawCandidates.length)} facts, ${esc(rawNodes.length)} nodes, ${esc((raw.edges || []).length)} edges; graph density ${esc(density)} edges/node.</li>
      <li>Database stats: ${esc(totalCandidates)} candidates and ${esc(edgeTotal)} edges before current UI filters.</li>
      <li>Heuristics: ${esc(candidateNoise.length)} suspicious facts and ${esc(nodeNoise.length)} suspicious nodes (low confidence, rejected/forgotten status, weak labels, missing evidence, or blob-like entities).</li>
      <li>Curated UI hidden right now: ${esc(Math.max(0, hiddenFacts))} facts and ${esc(Math.max(0, hiddenNodes))} nodes. Switch to raw/debug to inspect them.</li>
    </ul>`;
}
function evidenceSummary(obj) {
  if (!obj) return 'No selection.';
  if (Object.prototype.hasOwnProperty.call(obj, 'candidate_id')) {
    return `Fact candidate\nType: ${obj.type || '—'}\nClaim: ${obj.subject || '—'} → ${obj.predicate || '—'} → ${obj.object || '—'}\nStatus: ${obj.status || '—'} · confidence ${obj.confidence ?? '—'}\nEvidence: ${obj.source_quote || obj.text || 'No source quote recorded.'}\nProject/session: ${obj.project_id || '—'} / ${obj.session_id || '—'}\n\nRaw JSON:\n${JSON.stringify(obj, null, 2)}`;
  }
  if (Object.prototype.hasOwnProperty.call(obj, 'edge_id') || Object.prototype.hasOwnProperty.call(obj, 'relation')) {
    return `Graph edge\nRelation: ${obj.from_name || obj.source || '—'} → ${obj.relation || obj.label || '—'} → ${obj.to_name || obj.target || '—'}\nSource candidate: ${obj.source_candidate_id || '—'}\nEvidence: ${obj.source_quote || obj.candidate_text || 'No source quote recorded.'}\nProject/session: ${obj.project_id || '—'} / ${obj.session_id || '—'}\n\nRaw JSON:\n${JSON.stringify(obj, null, 2)}`;
  }
  return `Graph node\nLabel: ${obj.name || obj.label || obj.id || '—'}\nRoles: ${(obj.roles || []).join(', ') || '—'}\nDegree: ${obj.degree ?? '—'} · candidate count ${obj.candidate_count ?? '—'}\nNoise indicators: ${nodeIssues(obj).join(', ') || 'none'}\n\nRaw JSON:\n${JSON.stringify(obj, null, 2)}`;
}
function renderGraph(data) {
  const container = $('graph');
  const message = $('graph-message');
  if (graph) { graph.destroy(); graph = null; }
  container.innerHTML = '';
  message.hidden = true;
  message.className = 'graph-message';
  const nodes = data.nodes || [], edges = data.edges || [];
  if (window.__ppmlxG6LoadError || typeof G6 === 'undefined') {
    message.textContent = 'Unable to load AntV G6 from the CDN. Check your network connection or content-blocking settings, then refresh.';
    message.className = 'graph-message error';
    message.hidden = false;
    return;
  }
  if (!nodes.length) {
    message.textContent = 'No graph data for current filters.';
    message.hidden = false;
    return;
  }
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;
  const graphData = {
    nodes: nodes.map(n => ({
      ...n,
      label: n.label || n.name || n.id,
      size: n.size || Math.min(48, Math.max(18, 20 + Number(n.degree || n.candidate_count || 0) * 4)),
      style: { fill: Number(n.degree || 0) > 1 ? '#92400e' : '#1d4ed8', stroke: Number(n.degree || 0) > 1 ? '#fbbf24' : '#93c5fd', lineWidth: 1.5 },
      labelCfg: { style: { fill: '#dbeafe', fontSize: 11, stroke: '#0b0f14', lineWidth: 3 } },
    })),
    edges: edges.map(e => ({
      ...e,
      id: e.id || e.edge_id || `${e.source || e.from_entity_id}-${e.target || e.to_entity_id}-${e.relation || ''}`,
      source: e.source || e.from_entity_id,
      target: e.target || e.to_entity_id,
      label: e.label || e.relation || '',
      style: { stroke: '#334155', lineWidth: 1.2, opacity: 0.85 },
      labelCfg: { autoRotate: true, style: { fill: '#93a4b8', fontSize: 10, stroke: '#0b0f14', lineWidth: 3 } },
    })),
  };
  graph = new G6.Graph({
    container: 'graph',
    width,
    height,
    fitView: true,
    fitViewPadding: 36,
    modes: { default: ['drag-canvas', 'zoom-canvas', 'drag-node'] },
    layout: { type: 'force', preventOverlap: true, linkDistance: 150, nodeStrength: -70, edgeStrength: 0.2 },
    defaultNode: { type: 'circle' },
    defaultEdge: { type: 'line' },
  });
  graph.data(graphData);
  graph.render();
  graph.on('node:click', ev => show(ev.item.getModel()));
  graph.on('edge:click', ev => show(ev.item.getModel()));
}
function show(obj) { $('details').textContent = evidenceSummary(obj); }
load();
</script>
</body>
</html>"""
    return html.replace("__DEFAULTS_JSON__", defaults_json)


def _query_params(raw_query: str, defaults: dict[str, str]) -> dict[str, str]:
    parsed = parse_qs(raw_query)
    out = dict(defaults)
    for key in ("status", "query", "app_id", "project_id", "session_id", "limit"):
        if key in parsed:
            out[key] = parsed[key][0]
    try:
        limit = max(1, min(int(out.get("limit") or 120), 500))
    except ValueError:
        limit = 120
    out["limit"] = str(limit)
    if not out.get("status"):
        out["status"] = "active"
    return out
