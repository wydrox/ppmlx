"""Interactive API playground and documentation for ppmlx.

Serves a self-contained HTML page at ``/playground`` with:
- Endpoint selector with auto-generated documentation
- Request editor with JSON syntax highlighting
- Response viewer with streaming SSE support
- Code generator (Python, curl, JavaScript, TypeScript)
- Dark theme matching the terminal aesthetic
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(tags=["playground"])

# ── Example requests for each endpoint ────────────────────────────────

_EXAMPLE_REQUESTS: dict[str, dict[str, Any] | None] = {
    "POST /v1/chat/completions": {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": "Hello! What can you help me with?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    },
    "POST /v1/chat/completions (streaming)": {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": "Write a haiku about coding."}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    },
    "POST /v1/completions": {
        "model": "llama3",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.7,
    },
    "POST /v1/embeddings": {
        "model": "nomic-embed",
        "input": "Hello world",
    },
    "POST /v1/responses": {
        "model": "llama3",
        "input": "Explain the Pythagorean theorem briefly.",
        "stream": False,
    },
    "GET /v1/models": None,
    "GET /health": None,
    "GET /metrics": None,
}

_ENDPOINT_DOCS: dict[str, dict[str, str]] = {
    "POST /v1/chat/completions": {
        "description": "OpenAI-compatible chat completions. Supports streaming via SSE.",
        "parameters": json.dumps({
            "model": "string (required) - Model name or alias",
            "messages": "array (required) - List of {role, content} message objects",
            "temperature": "float (0-2, default 0.7) - Sampling temperature",
            "top_p": "float (0-1, default 1.0) - Nucleus sampling",
            "max_tokens": "int (optional) - Maximum tokens to generate",
            "stream": "bool (default false) - Enable SSE streaming",
            "stop": "string|array (optional) - Stop sequences",
            "seed": "int (optional) - Random seed for reproducibility",
        }, indent=2),
    },
    "POST /v1/chat/completions (streaming)": {
        "description": "Same as chat completions but with stream=true. Returns Server-Sent Events.",
        "parameters": "Same parameters as chat completions, with stream set to true.",
    },
    "POST /v1/completions": {
        "description": "Legacy text completions endpoint.",
        "parameters": json.dumps({
            "model": "string (required) - Model name or alias",
            "prompt": "string (required) - The prompt to complete",
            "max_tokens": "int (optional) - Maximum tokens to generate",
            "temperature": "float (0-2, default 0.7) - Sampling temperature",
        }, indent=2),
    },
    "POST /v1/embeddings": {
        "description": "Generate embeddings for input text. Requires an embedding model.",
        "parameters": json.dumps({
            "model": "string (required) - Embedding model name",
            "input": "string|array (required) - Text(s) to embed",
            "encoding_format": "string (float|base64, default float)",
        }, indent=2),
    },
    "POST /v1/responses": {
        "description": "OpenAI Responses API (used by Codex and newer OpenAI tools).",
        "parameters": json.dumps({
            "model": "string (required) - Model name or alias",
            "input": "string|array (required) - User input or message list",
            "instructions": "string (optional) - System prompt",
            "stream": "bool (default false) - Enable SSE streaming",
            "max_output_tokens": "int (optional) - Maximum tokens to generate",
            "temperature": "float (0-2, default 0.7) - Sampling temperature",
        }, indent=2),
    },
    "GET /v1/models": {
        "description": "List all available models (local and aliases).",
        "parameters": "No parameters required.",
    },
    "GET /health": {
        "description": "Health check. Returns server status, version, loaded models, and system info.",
        "parameters": "No parameters required.",
    },
    "GET /metrics": {
        "description": "Usage metrics. Returns request counts, average duration, and loaded models.",
        "parameters": "No parameters required.",
    },
}


_PLAYGROUND_HTML_CACHE: str | None = None


def _get_playground_html() -> str:
    """Return the full self-contained HTML for the playground page.

    The result is cached after the first call since all embedded data is static.
    """
    global _PLAYGROUND_HTML_CACHE
    if _PLAYGROUND_HTML_CACHE is not None:
        return _PLAYGROUND_HTML_CACHE

    endpoints_json = json.dumps(list(_EXAMPLE_REQUESTS.keys()))
    examples_json = json.dumps(_EXAMPLE_REQUESTS)
    docs_json = json.dumps(_ENDPOINT_DOCS)

    html = (
        r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ppmlx API Playground</title>
<style>
:root {
  --bg: #0d1117;
  --bg-card: #161b22;
  --bg-input: #0d1117;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --accent-hover: #79c0ff;
  --green: #3fb950;
  --red: #f85149;
  --orange: #d29922;
  --font-mono: "SF Mono", "Fira Code", "JetBrains Mono", "Cascadia Code", monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  line-height: 1.5;
  min-height: 100vh;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
}

.header h1 {
  font-size: 18px;
  font-weight: 600;
  color: var(--text);
}

.header h1 span { color: var(--accent); }

.header .version {
  font-size: 12px;
  color: var(--text-muted);
  background: var(--bg);
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid var(--border);
}

.container {
  display: grid;
  grid-template-columns: 280px 1fr 1fr;
  gap: 0;
  height: calc(100vh - 57px);
}

/* Sidebar */
.sidebar {
  background: var(--bg-card);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 12px 0;
}

.sidebar-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  padding: 8px 16px 4px;
}

.endpoint-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 13px;
  border-left: 2px solid transparent;
  transition: all 0.15s;
}

.endpoint-item:hover {
  background: rgba(88, 166, 255, 0.08);
}

.endpoint-item.active {
  background: rgba(88, 166, 255, 0.12);
  border-left-color: var(--accent);
}

.method-badge {
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
  padding: 2px 6px;
  border-radius: 3px;
  min-width: 36px;
  text-align: center;
}

.method-GET { background: rgba(63, 185, 80, 0.15); color: var(--green); }
.method-POST { background: rgba(88, 166, 255, 0.15); color: var(--accent); }

.endpoint-path {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text);
}

/* Panels */
.panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  min-height: 48px;
}

.panel-header h2 {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.request-panel { border-right: 1px solid var(--border); }

/* Editor */
textarea.editor {
  width: 100%;
  min-height: 200px;
  background: var(--bg-input);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.6;
  resize: vertical;
  outline: none;
  tab-size: 2;
}

textarea.editor:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.btn:hover { border-color: var(--text-muted); }

.btn-primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

.btn-primary:hover {
  background: var(--accent-hover);
  border-color: var(--accent-hover);
}

/* Response area */
.response-meta {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.meta-badge {
  font-size: 12px;
  font-family: var(--font-mono);
  padding: 3px 8px;
  border-radius: 4px;
  background: var(--bg-card);
  border: 1px solid var(--border);
}

.meta-badge.status-2xx { color: var(--green); border-color: rgba(63, 185, 80, 0.3); }
.meta-badge.status-4xx { color: var(--orange); border-color: rgba(210, 153, 34, 0.3); }
.meta-badge.status-5xx { color: var(--red); border-color: rgba(248, 81, 73, 0.3); }

pre.response-body {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  min-height: 80px;
}

/* Docs section */
.docs-section {
  margin-bottom: 16px;
  padding: 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
}

.docs-section h3 {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent);
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.docs-section p, .docs-section pre {
  font-size: 13px;
  color: var(--text-muted);
}

.docs-section pre {
  font-family: var(--font-mono);
  font-size: 12px;
  white-space: pre-wrap;
  margin-top: 4px;
}

/* Code snippets */
.code-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-top: 16px;
}

.code-tab {
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
  background: none;
  border-top: none;
  border-left: none;
  border-right: none;
}

.code-tab:hover { color: var(--text); }
.code-tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.code-block-container { position: relative; }

.code-block {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 6px 6px;
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  font-size: 11px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.15s;
}

.copy-btn:hover { color: var(--text); border-color: var(--text-muted); }
.copy-btn.copied { color: var(--green); border-color: var(--green); }

/* Streaming indicator */
.streaming-indicator {
  display: none;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--green);
}

.streaming-indicator.active { display: inline-flex; }

.streaming-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* Responsive */
@media (max-width: 1024px) {
  .container {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr 1fr;
  }
  .sidebar {
    display: flex;
    overflow-x: auto;
    border-right: none;
    border-bottom: 1px solid var(--border);
    padding: 8px;
    gap: 4px;
  }
  .sidebar-title { display: none; }
  .endpoint-item {
    white-space: nowrap;
    border-left: none;
    border-bottom: 2px solid transparent;
    padding: 6px 12px;
    border-radius: 4px;
  }
  .endpoint-item.active {
    border-left-color: transparent;
    border-bottom-color: var(--accent);
  }
}

/* Loading spinner */
.spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: var(--text-muted);
  font-size: 14px;
  text-align: center;
  gap: 8px;
}
</style>
</head>
<body>

<div class="header">
  <h1><span>ppmlx</span> API Playground</h1>
  <span class="version" id="server-version">loading...</span>
</div>

<div class="container">
  <!-- Sidebar: endpoint list -->
  <div class="sidebar">
    <div class="sidebar-title">Endpoints</div>
    <div id="endpoint-list"></div>
  </div>

  <!-- Request panel -->
  <div class="panel request-panel">
    <div class="panel-header">
      <h2>Request</h2>
      <div style="display:flex; gap:8px; align-items:center;">
        <span class="streaming-indicator" id="streaming-indicator">
          <span class="streaming-dot"></span> Streaming
        </span>
        <button class="btn btn-primary" id="send-btn" onclick="sendRequest()">Send</button>
      </div>
    </div>
    <div class="panel-body">
      <div id="docs-area"></div>
      <div id="editor-area" style="margin-top:12px;">
        <textarea class="editor" id="request-editor" spellcheck="false" placeholder="Select an endpoint from the sidebar..."></textarea>
      </div>
      <!-- Code snippets -->
      <div id="code-snippets-area">
        <div class="code-tabs" id="code-tabs">
          <button class="code-tab active" data-lang="curl" onclick="switchCodeTab('curl')">curl</button>
          <button class="code-tab" data-lang="python" onclick="switchCodeTab('python')">Python</button>
          <button class="code-tab" data-lang="javascript" onclick="switchCodeTab('javascript')">JavaScript</button>
          <button class="code-tab" data-lang="typescript" onclick="switchCodeTab('typescript')">TypeScript</button>
        </div>
        <div class="code-block-container">
          <pre class="code-block" id="code-block">Select an endpoint to see code snippets</pre>
          <button class="copy-btn" id="copy-code-btn" onclick="copyCode()">Copy</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Response panel -->
  <div class="panel">
    <div class="panel-header">
      <h2>Response</h2>
      <div id="response-timing" style="font-size:12px; color:var(--text-muted); font-family:var(--font-mono);"></div>
    </div>
    <div class="panel-body">
      <div class="response-meta" id="response-meta" style="display:none;"></div>
      <pre class="response-body" id="response-body"><span class="empty-state">Response will appear here after sending a request.</span></pre>
    </div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────
const ENDPOINTS = """
        + endpoints_json
        + r""";
const EXAMPLES = """
        + examples_json
        + r""";
const DOCS = """
        + docs_json
        + r""";

let currentEndpoint = ENDPOINTS[0];
let currentCodeLang = "curl";
let abortController = null;
const baseUrl = window.location.origin;

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  renderEndpointList();
  selectEndpoint(ENDPOINTS[0]);
  fetchVersion();
});

function fetchVersion() {
  fetch(baseUrl + "/health")
    .then(r => r.json())
    .then(d => {
      document.getElementById("server-version").textContent = "v" + (d.version || "?");
    })
    .catch(() => {
      document.getElementById("server-version").textContent = "offline";
    });
}

// ── Endpoint list ────────────────────────────────────────────────────
function renderEndpointList() {
  const container = document.getElementById("endpoint-list");
  container.innerHTML = "";
  for (const ep of ENDPOINTS) {
    const parts = ep.split(" ");
    const method = parts[0];
    const path = parts.slice(1).join(" ");
    const el = document.createElement("div");
    el.className = "endpoint-item";
    el.dataset.endpoint = ep;
    el.innerHTML =
      '<span class="method-badge method-' + method + '">' + method + "</span>" +
      '<span class="endpoint-path">' + escapeHtml(path) + "</span>";
    el.addEventListener("click", () => selectEndpoint(ep));
    container.appendChild(el);
  }
}

function selectEndpoint(ep) {
  currentEndpoint = ep;
  // Update active state
  document.querySelectorAll(".endpoint-item").forEach(el => {
    el.classList.toggle("active", el.dataset.endpoint === ep);
  });

  // Update editor
  const example = EXAMPLES[ep];
  const editor = document.getElementById("request-editor");
  if (example !== null && example !== undefined) {
    editor.value = JSON.stringify(example, null, 2);
    editor.style.display = "block";
  } else {
    editor.value = "";
    editor.style.display = "none";
  }

  // Update docs
  renderDocs(ep);

  // Update code snippets
  updateCodeSnippets();
}

// ── Documentation ────────────────────────────────────────────────────
function renderDocs(ep) {
  const area = document.getElementById("docs-area");
  const doc = DOCS[ep];
  if (!doc) { area.innerHTML = ""; return; }
  let html = '<div class="docs-section"><h3>Description</h3><p>' +
    escapeHtml(doc.description || "") + "</p></div>";
  if (doc.parameters) {
    html += '<div class="docs-section"><h3>Parameters</h3><pre>' +
      escapeHtml(doc.parameters) + "</pre></div>";
  }
  area.innerHTML = html;
}

// ── Code generation ──────────────────────────────────────────────────
function getRequestInfo() {
  const parts = currentEndpoint.split(" ");
  const method = parts[0];
  let path = parts.slice(1).join(" ");
  // Normalize streaming variant
  if (path.includes("(streaming)")) {
    path = "/v1/chat/completions";
  }
  const editor = document.getElementById("request-editor");
  let body = null;
  if (method === "POST" && editor.value.trim()) {
    try { body = JSON.parse(editor.value); } catch (e) { body = editor.value; }
  }
  return { method, path, body };
}

function generateCurl(info) {
  const url = baseUrl + info.path;
  if (info.method === "GET") {
    return "curl " + url;
  }
  const bodyStr = typeof info.body === "object"
    ? JSON.stringify(info.body, null, 2)
    : String(info.body || "{}");
  let cmd = "curl " + url + " \\\n  -H \"Content-Type: application/json\" \\\n  -d '" +
    bodyStr.replace(/'/g, "'\\''") + "'";
  if (info.body && info.body.stream) {
    cmd += " \\\n  --no-buffer";
  }
  return cmd;
}

function generatePython(info) {
  const url = baseUrl + info.path;
  if (info.method === "GET") {
    return 'import requests\n\nresponse = requests.get("' + url + '")\nprint(response.json())';
  }
  const bodyStr = typeof info.body === "object"
    ? JSON.stringify(info.body, null, 4).replace(/null/g, "None").replace(/true/g, "True").replace(/false/g, "False")
    : "{}";

  if (info.body && info.body.stream) {
    return 'from openai import OpenAI\n\nclient = OpenAI(base_url="' + baseUrl +
      '/v1", api_key="local")\n\nstream = client.chat.completions.create(\n    model="' +
      (info.body.model || "llama3") + '",\n    messages=' +
      JSON.stringify(info.body.messages || [], null, 4).replace(/null/g, "None").replace(/true/g, "True").replace(/false/g, "False") +
      ',\n    stream=True,\n)\n\nfor chunk in stream:\n    if chunk.choices[0].delta.content:\n        print(chunk.choices[0].delta.content, end="")';
  }

  if (info.path === "/v1/chat/completions") {
    return 'from openai import OpenAI\n\nclient = OpenAI(base_url="' + baseUrl +
      '/v1", api_key="local")\n\nresponse = client.chat.completions.create(\n    model="' +
      (info.body.model || "llama3") + '",\n    messages=' +
      JSON.stringify(info.body.messages || [], null, 4).replace(/null/g, "None").replace(/true/g, "True").replace(/false/g, "False") +
      ',\n)\n\nprint(response.choices[0].message.content)';
  }

  return 'import requests\n\nresponse = requests.post(\n    "' + url +
    '",\n    json=' + bodyStr + ',\n)\nprint(response.json())';
}

function generateJavaScript(info) {
  const url = baseUrl + info.path;
  if (info.method === "GET") {
    return 'const response = await fetch("' + url + '");\nconst data = await response.json();\nconsole.log(data);';
  }
  const bodyStr = JSON.stringify(info.body, null, 2);

  if (info.body && info.body.stream) {
    return 'const response = await fetch("' + url + '", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(' +
      bodyStr + '),\n});\n\nconst reader = response.body.getReader();\nconst decoder = new TextDecoder();\n\nwhile (true) {\n  const { done, value } = await reader.read();\n  if (done) break;\n  const text = decoder.decode(value);\n  for (const line of text.split("\\n")) {\n    if (line.startsWith("data: ") && line !== "data: [DONE]") {\n      const chunk = JSON.parse(line.slice(6));\n      const content = chunk.choices?.[0]?.delta?.content;\n      if (content) process.stdout.write(content);\n    }\n  }\n}';
  }

  return 'const response = await fetch("' + url +
    '", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(' +
    bodyStr + '),\n});\n\nconst data = await response.json();\nconsole.log(data);';
}

function generateTypeScript(info) {
  const url = baseUrl + info.path;
  if (info.method === "GET") {
    return 'const response: Response = await fetch("' + url + '");\nconst data: Record<string, unknown> = await response.json();\nconsole.log(data);';
  }
  const bodyStr = JSON.stringify(info.body, null, 2);

  if (info.body && info.body.stream) {
    return 'const response: Response = await fetch("' + url + '", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(' +
      bodyStr + '),\n});\n\nconst reader: ReadableStreamDefaultReader<Uint8Array> = response.body!.getReader();\nconst decoder = new TextDecoder();\n\nwhile (true) {\n  const { done, value } = await reader.read();\n  if (done) break;\n  const text: string = decoder.decode(value);\n  for (const line of text.split("\\n")) {\n    if (line.startsWith("data: ") && line !== "data: [DONE]") {\n      const chunk = JSON.parse(line.slice(6));\n      const content: string | undefined = chunk.choices?.[0]?.delta?.content;\n      if (content) process.stdout.write(content);\n    }\n  }\n}';
  }

  return 'interface ApiResponse {\n  [key: string]: unknown;\n}\n\nconst response: Response = await fetch("' + url +
    '", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(' +
    bodyStr + '),\n});\n\nconst data: ApiResponse = await response.json();\nconsole.log(data);';
}

function updateCodeSnippets() {
  const info = getRequestInfo();
  const generators = {
    curl: generateCurl,
    python: generatePython,
    javascript: generateJavaScript,
    typescript: generateTypeScript,
  };
  const code = generators[currentCodeLang](info);
  document.getElementById("code-block").textContent = code;
}

function switchCodeTab(lang) {
  currentCodeLang = lang;
  document.querySelectorAll(".code-tab").forEach(el => {
    el.classList.toggle("active", el.dataset.lang === lang);
  });
  updateCodeSnippets();
}

// ── Send request ─────────────────────────────────────────────────────
async function sendRequest() {
  const info = getRequestInfo();
  const url = baseUrl + info.path;
  const responseBody = document.getElementById("response-body");
  const responseMeta = document.getElementById("response-meta");
  const responseTiming = document.getElementById("response-timing");
  const streamingIndicator = document.getElementById("streaming-indicator");
  const sendBtn = document.getElementById("send-btn");

  // Cancel any in-flight request
  if (abortController) { abortController.abort(); }
  abortController = new AbortController();

  // Reset UI
  responseBody.textContent = "";
  responseMeta.style.display = "none";
  responseMeta.innerHTML = "";
  responseTiming.textContent = "";
  streamingIndicator.classList.remove("active");
  sendBtn.innerHTML = '<span class="spinner"></span> Sending';
  sendBtn.disabled = true;

  const startTime = performance.now();

  try {
    const isStreaming = info.body && info.body.stream === true;
    const fetchOpts = {
      method: info.method,
      signal: abortController.signal,
    };
    if (info.method === "POST" && info.body) {
      fetchOpts.headers = { "Content-Type": "application/json" };
      fetchOpts.body = JSON.stringify(info.body);
    }

    const response = await fetch(url, fetchOpts);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    // Show metadata
    responseMeta.style.display = "flex";
    const statusClass = response.status < 300 ? "status-2xx"
      : response.status < 500 ? "status-4xx" : "status-5xx";
    responseMeta.innerHTML =
      '<span class="meta-badge ' + statusClass + '">' + response.status + " " + response.statusText + "</span>" +
      '<span class="meta-badge">' + elapsed + "s</span>" +
      '<span class="meta-badge">' + (response.headers.get("content-type") || "unknown") + "</span>";

    if (isStreaming && response.ok) {
      // Stream SSE chunks
      streamingIndicator.classList.add("active");
      sendBtn.innerHTML = "Stop";
      sendBtn.disabled = false;
      sendBtn.onclick = () => {
        if (abortController) abortController.abort();
        streamingIndicator.classList.remove("active");
        sendBtn.innerHTML = "Send";
        sendBtn.onclick = sendRequest;
      };

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";
      let chunkCount = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const payload = line.slice(6);
            if (payload === "[DONE]") {
              accumulated += "\n\n--- Stream complete (" + chunkCount + " chunks) ---";
            } else {
              chunkCount++;
              try {
                const chunk = JSON.parse(payload);
                const content = chunk.choices && chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content;
                if (content) {
                  accumulated += content;
                } else {
                  // Show the full chunk for non-content deltas (e.g., tool calls, finish)
                  accumulated += "\n[chunk " + chunkCount + "] " + JSON.stringify(chunk, null, 2) + "\n";
                }
              } catch (e) {
                accumulated += payload + "\n";
              }
            }
          }
        }
        responseBody.textContent = accumulated;
        responseBody.scrollTop = responseBody.scrollHeight;
      }

      const totalElapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      responseTiming.textContent = totalElapsed + "s total";
      streamingIndicator.classList.remove("active");
    } else {
      // Non-streaming response
      const text = await response.text();
      try {
        const parsed = JSON.parse(text);
        responseBody.textContent = JSON.stringify(parsed, null, 2);
      } catch (e) {
        responseBody.textContent = text;
      }
      responseTiming.textContent = elapsed + "s";
    }
  } catch (err) {
    if (err.name === "AbortError") {
      responseBody.textContent = "Request cancelled.";
    } else {
      responseBody.textContent = "Error: " + err.message;
      responseMeta.style.display = "flex";
      responseMeta.innerHTML = '<span class="meta-badge status-5xx">Network Error</span>';
    }
  } finally {
    sendBtn.innerHTML = "Send";
    sendBtn.disabled = false;
    sendBtn.onclick = sendRequest;
  }
}

// ── Copy code ────────────────────────────────────────────────────────
function copyCode() {
  const code = document.getElementById("code-block").textContent;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById("copy-code-btn");
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = "Copy";
      btn.classList.remove("copied");
    }, 1500);
  });
}

// ── Update code on editor change ─────────────────────────────────────
document.getElementById("request-editor").addEventListener("input", updateCodeSnippets);

// ── Keyboard shortcut ────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    sendRequest();
  }
});

// ── Utils ────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}
</script>
</body>
</html>"""
    )
    _PLAYGROUND_HTML_CACHE = html
    return html


# ── Endpoint metadata API ────────────────────────────────────────────


def _discover_endpoints(app) -> list[dict]:
    """Introspect FastAPI app routes and return endpoint metadata."""
    endpoints = []
    for route in app.routes:
        if not hasattr(route, "methods"):
            continue
        path = getattr(route, "path", "")
        # Skip internal / playground routes
        if path.startswith("/playground") or path.startswith("/api/playground"):
            continue
        methods = sorted(route.methods - {"HEAD", "OPTIONS"})
        for method in methods:
            doc = getattr(route, "endpoint", None)
            description = ""
            if doc and doc.__doc__:
                description = doc.__doc__.strip().split("\n")[0]
            key = f"{method} {path}"
            example = _EXAMPLE_REQUESTS.get(key)
            docs = _ENDPOINT_DOCS.get(key, {})
            endpoints.append({
                "method": method,
                "path": path,
                "description": docs.get("description", description),
                "example_request": example,
            })
    return endpoints


@router.get("/playground", response_class=HTMLResponse)
async def playground():
    """Serve the interactive API playground."""
    return HTMLResponse(content=_get_playground_html())


@router.get("/api/playground/endpoints")
async def playground_endpoints(request: Request):
    """Return endpoint metadata for the playground."""
    endpoints = _discover_endpoints(request.app)
    return JSONResponse({
        "endpoints": endpoints,
        "server_url": str(request.base_url).rstrip("/"),
    })
