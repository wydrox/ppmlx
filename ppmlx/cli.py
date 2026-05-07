from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
try:
    import setproctitle as _setproctitle_mod
except ImportError:
    _setproctitle_mod = None  # type: ignore[assignment]

app = typer.Typer(
    name="ppmlx",
    help="Run LLMs locally on Apple Silicon via MLX",
    no_args_is_help=True,
)
console = Console()

_VALID_QUANTIZE_BITS = frozenset({2, 3, 4, 6, 8})


@dataclass
class _LaunchItem:
    key: str    # "run" | "claude" | "codex" | "opencode" | "pi"
    label: str
    desc: str
    cmd: str    # executable to check with shutil.which; "" = always installed


@dataclass
class _PickerRow:
    alias: str
    size_gb: float | None
    downloaded: bool
    section_header: str | None  # non-None → non-selectable section label
    params_b: float | None = None
    is_loaded: bool = False
    is_favorite: bool = False


_LAUNCH_ITEMS: list[_LaunchItem] = [
    _LaunchItem("run",      "Run a model",        "Start an interactive chat with a model",       ""),
    _LaunchItem("serve",    "Start API server",   "OpenAI-compatible server on :6767",            ""),
    _LaunchItem("claude",   "Launch Claude Code", "Agentic coding across large codebases",        "claude"),
    _LaunchItem("codex",    "Launch Codex",       "OpenAI's open-source coding agent",            "codex"),
    _LaunchItem("opencode", "Launch Opencode",    "Anomaly's open-source coding agent",           "opencode"),
    _LaunchItem("openwebui","Launch Open WebUI",  "Web-based chat UI with multimodal support",    "open-webui"),
    _LaunchItem("pi",       "Launch Pi",          "Minimal AI agent toolkit with plugin support", "pi"),
]


def _track_usage(event: str, data: dict | None = None, *, context: str = "cli") -> None:
    try:
        from ppmlx.analytics import track_async

        track_async(event, data, context=context)
    except Exception:
        pass


# ── Unified model record & table builder ─────────────────────────────

@dataclass
class ModelRecord:
    alias: str
    repo_id: str
    is_favorite: bool = False
    is_downloaded: bool = False
    is_loaded: bool = False
    size_gb: float | None = None
    local_path: Path | None = None
    params_b: float | None = None
    model_type: str | None = None   # "dense" | "sparse"
    lab: str | None = None
    modalities: str | None = None
    downloads: int | None = None
    released: str | None = None
    source: str = "registry"        # "built-in" | "custom" | "registry" | "local-only"


def _build_model_records(
    *,
    filter_downloaded: bool | None = None,
    filter_favorites: bool = False,
    filter_text: str | None = None,
    filter_lab: str | None = None,
    filter_modality: str | None = None,
    filter_type: str | None = None,
    sort_by: str = "alias",
    limit: int | None = None,
    exclude_embed: bool = True,
) -> list[ModelRecord]:
    """Build a unified list of model records from all sources."""
    from ppmlx.models import (
        all_aliases, DEFAULT_ALIASES, load_user_aliases,
        list_local_models, load_favorites,
    )

    try:
        from ppmlx.registry import registry_entries
        reg = registry_entries()
    except Exception:
        reg = {}

    aliases = all_aliases()
    user_aliases = load_user_aliases()
    local_models = list_local_models()
    favorites_list = load_favorites()
    fav_set = set(favorites_list)

    local_by_repo: dict[str, dict] = {}
    for m in local_models:
        local_by_repo[m["repo_id"]] = m

    # Source priority: custom > built-in > registry
    _SOURCE_PRIO = {"custom": 0, "built-in": 1, "registry": 2, "local-only": 3}

    # Group by repo_id, keeping only the best alias per repo.
    by_repo: dict[str, ModelRecord] = {}
    for alias, repo_id in aliases.items():
        lm = local_by_repo.get(repo_id)
        reg_entry = reg.get(alias, {})

        source = "registry"
        if alias in user_aliases:
            source = "custom"
        elif alias in DEFAULT_ALIASES:
            source = "built-in"

        candidate = ModelRecord(
            alias=alias,
            repo_id=repo_id,
            is_favorite=alias in fav_set,
            is_downloaded=lm is not None,
            size_gb=lm["size_gb"] if lm else reg_entry.get("size_gb"),
            local_path=lm["path"] if lm else None,
            params_b=reg_entry.get("params_b") or None,
            model_type=reg_entry.get("type"),
            lab=reg_entry.get("lab"),
            modalities=reg_entry.get("modalities"),
            downloads=reg_entry.get("downloads"),
            released=reg_entry.get("created"),
            source=source,
        )

        existing = by_repo.get(repo_id)
        if existing is None:
            by_repo[repo_id] = candidate
        else:
            # Pick best alias: higher source priority wins, then shorter name.
            e_prio = _SOURCE_PRIO.get(existing.source, 9)
            c_prio = _SOURCE_PRIO.get(candidate.source, 9)
            if (c_prio, len(alias)) < (e_prio, len(existing.alias)):
                # Carry over fields the new candidate might lack.
                if candidate.params_b is None:
                    candidate.params_b = existing.params_b
                if candidate.size_gb is None:
                    candidate.size_gb = existing.size_gb
                if candidate.lab is None:
                    candidate.lab = existing.lab
                if not candidate.is_favorite and existing.is_favorite:
                    candidate.is_favorite = True
                by_repo[repo_id] = candidate
            else:
                # Keep existing but merge any missing metadata from candidate.
                if existing.params_b is None:
                    existing.params_b = candidate.params_b
                if existing.size_gb is None:
                    existing.size_gb = candidate.size_gb
                if existing.lab is None:
                    existing.lab = candidate.lab
                if not existing.is_favorite and candidate.is_favorite:
                    existing.is_favorite = True

    records = dict(by_repo)

    # Local-only models (downloaded but not in any alias map)
    alias_repos = set(aliases.values())
    for m in local_models:
        if m["repo_id"] not in alias_repos:
            alias = m.get("alias") or m["repo_id"]
            if m["repo_id"] not in records:
                records[m["repo_id"]] = ModelRecord(
                    alias=alias,
                    repo_id=m["repo_id"],
                    is_favorite=alias in fav_set,
                    is_downloaded=True,
                    size_gb=m["size_gb"],
                    local_path=m["path"],
                    source="local-only",
                )

    result = list(records.values())

    if exclude_embed:
        result = [r for r in result if not r.alias.startswith("embed:")]
    if filter_downloaded is True:
        result = [r for r in result if r.is_downloaded]
    elif filter_downloaded is False:
        result = [r for r in result if not r.is_downloaded]
    if filter_favorites:
        result = [r for r in result if r.is_favorite]
    if filter_text:
        q = filter_text.lower()
        result = [r for r in result if q in f"{r.alias} {r.repo_id} {r.lab or ''}".lower()]
    if filter_lab:
        q = filter_lab.lower()
        result = [r for r in result if q in (r.lab or "").lower()]
    if filter_modality:
        q = filter_modality.lower()
        result = [r for r in result if q in (r.modalities or "").lower()]
    if filter_type:
        q = filter_type.lower()
        result = [r for r in result if (r.model_type or "").lower() == q]

    sort_keys = {
        "alias":     lambda r: r.alias,
        "downloads": lambda r: -(r.downloads or 0),
        "size":      lambda r: -(r.size_gb or 0),
        "params":    lambda r: -(r.params_b or 0),
        "created":   lambda r: r.released or "",
        "name":      lambda r: r.alias,
    }
    result.sort(key=lambda r: (not r.is_favorite, sort_keys.get(sort_by, sort_keys["alias"])(r)))

    if limit:
        result = result[:limit]
    return result


# ── Picker helpers ───────────────────────────────────────────────────

def _group_by_lab(records: list[ModelRecord]) -> list[tuple[str, list[ModelRecord]]]:
    """Group records by lab, sorted alphabetically. Models without lab go under 'Other'."""
    groups: dict[str, list[ModelRecord]] = {}
    for r in records:
        lab = r.lab or "Other"
        groups.setdefault(lab, []).append(r)
    for models in groups.values():
        models.sort(key=lambda r: r.alias)
    return sorted(groups.items(), key=lambda x: (x[0] == "Other", x[0]))


def _build_picker_rows(*, local_only: bool = False) -> list[_PickerRow]:
    records = _build_model_records()
    rows: list[_PickerRow] = []

    fav_records = [r for r in records if r.is_favorite and (not local_only or r.is_downloaded)]
    dl_records = [r for r in records if r.is_downloaded and not r.is_favorite]
    avail_records = [r for r in records if not r.is_downloaded and not r.is_favorite]

    def _row(r: ModelRecord) -> _PickerRow:
        return _PickerRow(
            alias=r.alias, size_gb=r.size_gb, downloaded=r.is_downloaded,
            section_header=None, params_b=r.params_b,
            is_loaded=r.is_loaded, is_favorite=r.is_favorite,
        )

    if fav_records:
        rows.append(_PickerRow("", None, True, "★ Favorites"))
        for r in fav_records:
            rows.append(_row(r))

    if dl_records:
        rows.append(_PickerRow("", None, True, "Downloaded"))
        for r in sorted(dl_records, key=lambda r: r.alias):
            rows.append(_row(r))

    if avail_records and not local_only:
        rows.append(_PickerRow("", None, False, "Available"))
        for r in sorted(avail_records, key=lambda r: r.alias):
            rows.append(_row(r))

    return rows


def _visible_rows(rows: list[_PickerRow], ft: str) -> list[_PickerRow]:
    if not ft:
        return rows
    result: list[_PickerRow] = []
    pending_header: _PickerRow | None = None
    for row in rows:
        if row.section_header is not None:
            pending_header = row
            continue
        if ft.lower() in row.alias.lower():
            if pending_header is not None:
                result.append(pending_header)
                pending_header = None
            result.append(row)
    return result


def _launch_tui(
    *, preset_action: str | None = None, command_str: str = "ppmlx launch",
) -> tuple[str | None, str | None]:
    """Textual launch menu. Returns (action_key, model_alias) or (None, None)."""
    from ppmlx.tui import launch_menu
    return launch_menu(preset_action=preset_action, command_str=command_str)


def _start_server_bg(model: str, host: str, port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "ppmlx.cli", "serve", "--host", host, "--port", str(port)]
    if model:
        cmd += ["--model", model]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _wait_server_ready(host: str, port: int, proc: subprocess.Popen, timeout: int = 30) -> bool:
    import httpx
    import time
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _port_in_use(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _flush_port(host: str, port: int) -> None:
    """Kill ppmlx processes listening on the given port and wait until it's free.

    Only kills processes whose command line contains 'ppmlx' to avoid
    accidentally terminating unrelated services.
    """
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip()
    if not pids:
        return
    for pid_str in pids.splitlines():
        pid_str = pid_str.strip()
        if not pid_str:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        # Verify the process belongs to ppmlx before killing
        try:
            ps_result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=5,
            )
            cmd_line = ps_result.stdout.strip()
            if "ppmlx" not in cmd_line:
                console.print(
                    f"[yellow]Skipping PID {pid} on port {port} "
                    f"(not a ppmlx process)[/yellow]"
                )
                continue
        except Exception:
            # If we can't verify, skip rather than kill blindly
            continue
        try:
            os.kill(pid, 9)
            console.print(f"[yellow]Killed ppmlx process {pid} on port {port}[/yellow]")
        except (ProcessLookupError, PermissionError):
            pass
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _port_in_use(host, port):
            return
        time.sleep(0.3)
    console.print(f"[red]Port {port} still in use after killing processes.[/red]")


def _launch_coding_tool(action: str, model: str, host: str, port: int) -> None:
    if _port_in_use(host, port):
        console.print(f"[red]Port {port} is already in use.[/red]")
        console.print(f"[dim]Try: ppmlx launch -a {action} -m {model} --port {port + 1}[/dim]")
        raise typer.Exit(1)

    proc = _start_server_bg(model, host, port)
    console.print(f"[dim]Starting ppmlx server on {host}:{port}...[/dim]")

    try:
        ready = _wait_server_ready(host, port, proc)
    except KeyboardInterrupt:
        proc.terminate()
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(1)

    if not ready:
        stderr_output = ""
        if proc.stderr:
            stderr_output = proc.stderr.read().decode(errors="replace").strip()
        proc.terminate()
        console.print("[red]Server failed to start within 30 seconds.[/red]")
        if stderr_output:
            console.print(f"[dim]{stderr_output[-500:]}[/dim]")
        raise typer.Exit(1)

    base_url = f"http://{host}:{port}/v1"
    env = os.environ.copy()

    if action == "claude":
        base = f"http://{host}:{port}"
        cmd = ["claude", "--model", model]
        env["ANTHROPIC_BASE_URL"] = base
        env["ANTHROPIC_API_KEY"] = "local"
    elif action == "codex":
        cmd = [
            "codex", "--model", model,
            "-c", 'model_provider="ppmlx"',
            "-c", 'model_providers.ppmlx.name="ppmlx"',
            "-c", f'model_providers.ppmlx.base_url="{base_url}"',
            "-c", 'model_providers.ppmlx.env_key="OPENAI_API_KEY"',
            "-c", 'model_providers.ppmlx.wire_api="responses"',
        ]
        env["OPENAI_API_KEY"] = "local"
    elif action == "opencode":
        cmd = ["opencode"]
        env["OPENAI_API_KEY"] = "local"
        env["OPENAI_BASE_URL"] = base_url
    elif action == "pi":
        models_file = Path.home() / ".pi" / "agent" / "models.json"
        models_file.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(models_file.read_text()) if models_file.exists() else {}
        if isinstance(existing, dict) and "providers" in existing:
            existing["providers"]["ppmlx"] = {
                "api": "openai-completions",
                "apiKey": "local",
                "baseUrl": base_url,
                "models": [{
                    "_launch": True,
                    "contextWindow": 262144,
                    "id": model,
                    "input": ["text"],
                    "reasoning": True,
                }],
            }
        else:
            if isinstance(existing, list):
                entries = [e for e in existing if isinstance(e, dict) and e.get("id") != "ppmlx"]
            else:
                entries = []
            entries.append({
                "id": "ppmlx",
                "name": f"ppmlx ({model})",
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": "local",
            })
            existing = entries
        models_file.write_text(json.dumps(existing, indent=2))
        cmd = ["pi", "--model", f"ppmlx/{model}"]
    elif action == "openwebui":
        import time, webbrowser
        owui_env = env.copy()
        owui_env["OPENAI_API_BASE_URLS"] = base_url
        owui_env["OPENAI_API_KEYS"] = "local"
        owui_env["OLLAMA_BASE_URLS"] = ""
        owui_env["ENABLE_OLLAMA_API"] = "false"
        owui_env["DEFAULT_MODELS"] = model
        owui_port = 8080
        owui_proc = subprocess.Popen(
            ["open-webui", "serve", "--port", str(owui_port)],
            env=owui_env,
        )
        owui_url = f"http://localhost:{owui_port}"
        console.print(f"[dim]Waiting for Open WebUI on {owui_url}...[/dim]")
        import httpx
        deadline = time.monotonic() + 60
        ready = False
        while time.monotonic() < deadline:
            if owui_proc.poll() is not None:
                break
            try:
                if httpx.get(owui_url, timeout=1.0).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if ready:
            webbrowser.open(owui_url)
            console.print(f"[green]Open WebUI ready at {owui_url}[/green]")
            try:
                owui_proc.wait()
            except KeyboardInterrupt:
                pass
        else:
            console.print("[red]Open WebUI failed to start within 60 seconds.[/red]")
        owui_proc.terminate()
        try:
            owui_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owui_proc.kill()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    else:
        proc.terminate()
        return

    try:
        subprocess.run(cmd, env=env)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _pick_model_tui(*, command_str: str = "ppmlx", local_only: bool = False, allow_none: bool = False) -> str | None:
    """Textual model picker. Returns alias, None (lazy-load), or raises typer.Exit on cancel."""
    from ppmlx.tui import pick_model

    rows = _build_picker_rows(local_only=local_only)
    if not any(r.section_header is None for r in rows):
        if local_only:
            console.print("[yellow]No local models found. Run: ppmlx pull <model>[/yellow]")
        else:
            console.print("[yellow]No models available.[/yellow]")
        raise typer.Exit(1)

    selected = pick_model(local_only=local_only, command_str=command_str, allow_none=allow_none)
    if selected is None and not allow_none:
        raise typer.Exit()
    return selected


def _pick_model(local_only: bool = False, multi: bool = False) -> str | list[str]:
    """Model picker. Returns alias (multi=False) or list of aliases (multi=True).

    Raises typer.Exit if cancelled or nothing selected.
    """
    if not multi:
        result = _pick_model_tui(local_only=local_only)
        if result is None:
            raise typer.Exit()
        return result

    from ppmlx.tui import pick_models
    selected = pick_models(local_only=local_only)
    if not selected:
        raise typer.Exit()
    return selected


def _version_callback(value: bool):
    if value:
        from ppmlx import __version__
        console.print(f"ppmlx {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    )
):
    """ppmlx: Run LLMs on Apple Silicon via MLX."""
    from ppmlx.config import check_first_run; check_first_run()


@app.command()
def launch(
    action: Optional[str] = typer.Argument(None, help="Action: run, serve, claude, codex, opencode, openwebui, pi"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name or alias"),
    host: Optional[str] = typer.Option(None, help="Bind host"),
    port: Optional[int] = typer.Option(None, help="Bind port (default: 6767)"),
    no_cors: bool = typer.Option(False, "--no-cors", help="Disable CORS"),
    flush: bool = typer.Option(False, "--flush", "-f", help="Kill any process using the port before starting"),
):
    """Select an action and model, then launch.

    Without arguments, opens an interactive TUI picker.
    With ACTION and MODEL, launches directly (non-interactive).
    """
    from ppmlx.config import load_config

    valid_actions = {item.key for item in _LAUNCH_ITEMS}
    _track_usage("launch_invoked", {"interactive": not bool(action and model)})

    overrides = {k: v for k, v in [("host", host), ("port", port)] if v}
    cfg = load_config(cli_overrides=overrides)
    effective_host = host or cfg.server.host
    effective_port = port or cfg.server.port

    if flush:
        _flush_port(effective_host, effective_port)

    if action and model:
        if action not in valid_actions:
            console.print(f"[red]Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}[/red]")
            raise typer.Exit(1)
    elif action and not model:
        if action in valid_actions:
            action, model = _launch_tui(
                preset_action=action,
                command_str=f"ppmlx launch {action}",
            )
        else:
            action, model = _launch_tui()
    else:
        action, model = _launch_tui()

    if not action:
        raise typer.Exit()

    if not model:
        console.print("[yellow]No model selected. Press \u2192 in the menu to pick one.[/yellow]")
        raise typer.Exit(1)

    if action == "run":
        run(model=model, system=None, max_kv_size=None, temperature=None, max_tokens=None)
    elif action == "serve":
        serve(model=model, host=effective_host, port=effective_port, interactive=False, no_cors=no_cors)
    else:
        _launch_coding_tool(action, model, effective_host, effective_port)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host"),
    port: Optional[int] = typer.Option(None, help="Bind port (default: 6767)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Pre-load a model on startup"),
    embed_model: Optional[str] = typer.Option(None, "--embed-model", help="Pre-load an embedding model"),
    no_cors: bool = typer.Option(False, "--no-cors", help="Disable CORS"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactively select a model to serve"),
):
    """Start the OpenAI-compatible API server."""
    import uvicorn
    from ppmlx.config import load_config
    from ppmlx import __version__

    overrides = {}
    if host: overrides["host"] = host
    if port: overrides["port"] = port
    if no_cors: overrides["cors"] = False
    cfg = load_config(cli_overrides=overrides)
    _track_usage(
        "serve_started",
        {
            "interactive": interactive,
            "preload_model": bool(model),
            "embed_model": bool(embed_model),
            "cors": cfg.server.cors,
        },
        context="server",
    )

    effective_host = host or cfg.server.host
    effective_port = port or cfg.server.port

    # Interactive model selection
    if interactive and model is None:
        model = _pick_model_tui(
            command_str="ppmlx serve --interactive",
            local_only=True,
            allow_none=True,
        )

    if effective_host != "127.0.0.1" and effective_host != "localhost":
        console.print(
            "[bold yellow]Warning:[/bold yellow] Server bound to "
            f"[bold]{effective_host}[/bold] — accessible from your network.\n"
            "         ppmlx has no authentication. Use a reverse proxy for production.",
        )

    console.print(Panel(
        f"[bold green]ppmlx server v{__version__}[/bold green]\n"
        f"   Listening on [link]http://{effective_host}:{effective_port}[/link]\n"
        f"   Endpoints:\n"
        f"     POST /v1/chat/completions\n"
        f"     POST /v1/completions\n"
        f"     POST /v1/embeddings\n"
        f"     GET  /v1/models\n"
        f"     GET  /health\n"
        f"     GET  /metrics\n"
        f"   SQLite log: ~/.ppmlx/ppmlx.db",
        title="ppmlx",
        border_style="green",
    ))

    # IDE connection hint
    selected_model = model or "(any — set model in your IDE)"
    console.print(Panel(
        f"[bold]API base:[/bold]  http://{effective_host}:{effective_port}/v1\n"
        f"[bold]Model:[/bold]     {selected_model}\n"
        f"[bold]API key:[/bold]   (not required — use any string)\n\n"
        f"[dim]Cursor[/dim]   → Settings › AI › OpenAI-compatible\n"
        f"[dim]Continue[/dim] → config.json: provider 'openai', apiBase above\n"
        f"[dim]Aider[/dim]    → --openai-api-base http://{effective_host}:{effective_port}/v1",
        title="Connect your IDE",
        border_style="blue",
    ))

    if _setproctitle_mod:
        _setproctitle_mod.setproctitle(f"ppmlx: server ({effective_host}:{effective_port})")

    uvicorn.run(
        "ppmlx.server:app",
        host=effective_host,
        port=effective_port,
        log_level="info",
        reload=False,
    )


@app.command()
def run(
    model: Optional[str] = typer.Argument(None, help="Model name or alias"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_kv_size: Optional[int] = typer.Option(None, "--max-kv-size", help="Max KV cache tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
):
    """Start an interactive chat REPL with a model."""
    _track_usage("repl_started", {"interactive_model_pick": model is None})
    if not model:
        model = _pick_model()
    from ppmlx.models import get_model_path, download_model, resolve_alias, ModelNotFoundError
    from ppmlx.engine import get_engine
    from ppmlx.memory import check_memory_warning

    try:
        repo_id = resolve_alias(model)
    except ModelNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    local_path = get_model_path(repo_id)
    if not local_path:
        console.print(f"[yellow]Model not found locally. Downloading {model}...[/yellow]")
        try:
            local_path = download_model(model)
        except KeyboardInterrupt:
            console.print("\n[yellow]Download cancelled.[/yellow]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            raise typer.Exit(1)

    warning = check_memory_warning(local_path)
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")

    engine = get_engine()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    # Session state
    history_enabled = True
    wordwrap = True
    verbose = False
    format_json = False
    think = False

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.filters import emacs_insert_mode

    _kb = KeyBindings()

    @_kb.add("c-left")   # Ctrl+Left  — jump word left
    @_kb.add("escape", "b")  # Alt/Option+B
    def _word_left(event):
        event.current_buffer.cursor_left(count=len(
            event.current_buffer.document.get_word_before_cursor(WORD=True) or " "
        ))

    @_kb.add("c-right")  # Ctrl+Right — jump word right
    @_kb.add("escape", "f")  # Alt/Option+F
    def _word_right(event):
        event.current_buffer.cursor_right(count=len(
            event.current_buffer.document.get_word_after_cursor(WORD=True) or " "
        ))

    @_kb.add("s-left")   # Shift+Left  — extend selection left (move + select)
    def _sel_left(event):
        buf = event.current_buffer
        if buf.selection_state is None:
            buf.start_selection()
        buf.cursor_left()

    @_kb.add("s-right")  # Shift+Right — extend selection right
    def _sel_right(event):
        buf = event.current_buffer
        if buf.selection_state is None:
            buf.start_selection()
        buf.cursor_right()

    @_kb.add("c-a")      # Ctrl+A — beginning of line (also Cmd+Left in most macOS terminals)
    def _bol(event):
        event.current_buffer.cursor_position = 0

    @_kb.add("c-e")      # Ctrl+E — end of line (also Cmd+Right)
    def _eol(event):
        event.current_buffer.cursor_position = len(event.current_buffer.text)

    @_kb.add("c-k")      # Ctrl+K — delete to end of line
    def _kill_eol(event):
        buf = event.current_buffer
        buf.delete(count=len(buf.document.get_text_after_cursor()))

    @_kb.add("c-u")      # Ctrl+U — delete to beginning of line
    def _kill_bol(event):
        buf = event.current_buffer
        deleted = buf.cursor_position
        buf.cursor_position = 0
        buf.delete(count=deleted)

    _session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        key_bindings=_kb,
        mouse_support=False,
        enable_open_in_editor=False,
    )
    _prompt = ANSI("\033[1;34mYou\033[0m: ")

    console.print(f"[green]Chatting with [bold]{model}[/bold]. Type /help or /? for commands, /bye to exit.[/green]")

    while True:
        try:
            user_input = _session.prompt(_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/bye", "/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        elif user_input in ("/help", "/?"):
            console.print("[bold]REPL commands:[/bold]")
            console.print("  [cyan]/set parameter <key> <val>[/cyan]  Set a parameter (temperature, max_tokens)")
            console.print("  [cyan]/set system <string>[/cyan]        Set system message")
            console.print("  [cyan]/set history[/cyan]                Enable history")
            console.print("  [cyan]/set nohistory[/cyan]              Disable history")
            console.print("  [cyan]/set wordwrap[/cyan]               Enable word wrap")
            console.print("  [cyan]/set nowordwrap[/cyan]             Disable word wrap")
            console.print("  [cyan]/set format json[/cyan]            Enable JSON mode")
            console.print("  [cyan]/set noformat[/cyan]               Disable formatting")
            console.print("  [cyan]/set verbose[/cyan]                Show LLM stats after each reply")
            console.print("  [cyan]/set quiet[/cyan]                  Disable LLM stats")
            console.print("  [cyan]/set think[/cyan]                  Show thinking blocks")
            console.print("  [cyan]/set nothink[/cyan]                Hide thinking blocks")
            console.print("  [cyan]/show info[/cyan]                  Show model details")
            console.print("  [cyan]/show license[/cyan]               Show model license")
            console.print("  [cyan]/show modelfile[/cyan]             Show model config")
            console.print("  [cyan]/show parameters[/cyan]            Show generation parameters")
            console.print("  [cyan]/show system[/cyan]                Show system message")
            console.print("  [cyan]/show template[/cyan]              Show chat template")
            console.print("  [cyan]/clear[/cyan]                      Clear conversation history")
            console.print("  [cyan]/model <name>[/cyan]               Switch to a different model")
            console.print("  [cyan]/bye[/cyan]                        Exit")

        elif user_input.startswith("/set"):
            arg = user_input[4:].strip()
            parts = arg.split(None, 2)
            sub = parts[0] if parts else ""

            def _set_help() -> None:
                console.print("[bold]/set options:[/bold]")
                console.print("  [cyan]system <string>[/cyan]         Set system message")
                console.print("  [cyan]parameter <key> <value>[/cyan] Set a generation parameter")
                console.print("  [cyan]history[/cyan]                 Enable history")
                console.print("  [cyan]nohistory[/cyan]               Disable history")
                console.print("  [cyan]wordwrap[/cyan]                Enable word wrap")
                console.print("  [cyan]nowordwrap[/cyan]              Disable word wrap")
                console.print("  [cyan]format json[/cyan]             Enable JSON mode")
                console.print("  [cyan]noformat[/cyan]                Disable formatting")
                console.print("  [cyan]verbose[/cyan]                 Show LLM stats after each reply")
                console.print("  [cyan]quiet[/cyan]                   Disable LLM stats")
                console.print("  [cyan]think[/cyan]                   Show thinking blocks")
                console.print("  [cyan]nothink[/cyan]                 Hide thinking blocks")

            def _set_parameter_help() -> None:
                console.print("[bold]/set parameter options:[/bold]")
                console.print("  [cyan]temperature <float>[/cyan]   Sampling temperature (e.g. 0.7)")
                console.print("  [cyan]max_tokens <int>[/cyan]      Max tokens to generate (e.g. 2048)")
                console.print("  [cyan]num_predict <int>[/cyan]     Alias for max_tokens")

            if not sub or sub == "?":
                _set_help()
            elif sub == "system":
                if len(parts) < 2 or parts[1] == "?":
                    console.print("[bold]/set system[/bold] — set the system message sent before the conversation.")
                    console.print("  Usage: [cyan]/set system <your prompt here>[/cyan]")
                else:
                    new_system = arg[len("system"):].strip()
                    messages = [m for m in messages if m["role"] != "system"]
                    if new_system:
                        messages.insert(0, {"role": "system", "content": new_system})
                    console.print("[dim]System message updated.[/dim]")
            elif sub == "history":
                history_enabled = True
                console.print("[dim]History enabled.[/dim]")
            elif sub == "nohistory":
                history_enabled = False
                console.print("[dim]History disabled.[/dim]")
            elif sub == "wordwrap":
                wordwrap = True
                console.print("[dim]Word wrap enabled.[/dim]")
            elif sub == "nowordwrap":
                wordwrap = False
                console.print("[dim]Word wrap disabled.[/dim]")
            elif sub == "format":
                if len(parts) < 2 or parts[1] == "?":
                    console.print("[bold]/set format[/bold] — enable a response format.")
                    console.print("  [cyan]/set format json[/cyan]  Respond only with valid JSON")
                elif parts[1] == "json":
                    format_json = True
                    console.print("[dim]JSON mode enabled.[/dim]")
                else:
                    console.print(f"[red]Unknown format: {parts[1]}. Supported: json[/red]")
            elif sub == "noformat":
                format_json = False
                console.print("[dim]Formatting disabled.[/dim]")
            elif sub == "verbose":
                verbose = True
                console.print("[dim]Verbose mode enabled.[/dim]")
            elif sub == "quiet":
                verbose = False
                console.print("[dim]Quiet mode enabled.[/dim]")
            elif sub == "think":
                think = True
                console.print("[dim]Thinking enabled.[/dim]")
            elif sub == "nothink":
                think = False
                console.print("[dim]Thinking disabled.[/dim]")
            elif sub == "parameter":
                if len(parts) < 3 or parts[1] == "?":
                    _set_parameter_help()
                else:
                    key, val = parts[1], parts[2]
                    if key == "temperature":
                        try:
                            temperature = float(val)
                            console.print(f"[dim]temperature = {temperature}[/dim]")
                        except ValueError:
                            console.print(f"[red]Invalid value: {val}[/red]")
                    elif key in ("max_tokens", "num_predict"):
                        try:
                            max_tokens = int(val)
                            console.print(f"[dim]max_tokens = {max_tokens}[/dim]")
                        except ValueError:
                            console.print(f"[red]Invalid value: {val}[/red]")
                    else:
                        console.print(f"[red]Unknown parameter: {key}[/red]")
                        _set_parameter_help()
            else:
                console.print(f"[red]Unknown /set option: {sub}[/red]")
                _set_help()

        elif user_input.startswith("/show"):
            sub = user_input[5:].strip()
            sys_msgs = [m for m in messages if m["role"] == "system"]
            sys_prompt = sys_msgs[0]["content"] if sys_msgs else "(none)"
            t = temperature if temperature is not None else 0.7
            mt = max_tokens if max_tokens is not None else 2048

            if sub in ("", "info"):
                console.print(f"  [bold]model[/bold]       {model}  ({repo_id})")
                console.print(f"  [bold]path[/bold]        {local_path or '(not cached)'}")
                console.print(f"  [bold]system[/bold]      {sys_prompt[:80]}")
                console.print(f"  [bold]temperature[/bold] {t}")
                console.print(f"  [bold]max_tokens[/bold]  {mt}")
                console.print(f"  [bold]history[/bold]     {'on' if history_enabled else 'off'}")
                console.print(f"  [bold]wordwrap[/bold]    {'on' if wordwrap else 'off'}")
                console.print(f"  [bold]verbose[/bold]     {'on' if verbose else 'off'}")
                console.print(f"  [bold]think[/bold]       {'on' if think else 'off'}")
                console.print(f"  [bold]json[/bold]        {'on' if format_json else 'off'}")

            elif sub == "system":
                console.print(sys_prompt)

            elif sub in ("parameters", "params"):
                console.print(f"  temperature  {t}")
                console.print(f"  max_tokens   {mt}")

            elif sub == "license":
                if local_path:
                    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "license.txt"):
                        lic = Path(local_path) / name
                        if lic.exists():
                            console.print(lic.read_text())
                            break
                    else:
                        console.print("[dim]No LICENSE file found in model directory.[/dim]")
                else:
                    console.print("[dim]Model not downloaded yet.[/dim]")

            elif sub == "modelfile":
                console.print(f"  FROM {repo_id}")
                if sys_prompt != "(none)":
                    console.print(f"  SYSTEM {sys_prompt}")
                console.print(f"  PARAMETER temperature {t}")
                console.print(f"  PARAMETER num_predict {mt}")
                if local_path:
                    cfg = Path(local_path) / "config.json"
                    if cfg.exists():
                        try:
                            data = json.loads(cfg.read_text())
                            for k in ("model_type", "architectures", "quantization_config"):
                                if k in data:
                                    console.print(f"  # {k}: {data[k]}")
                        except Exception:
                            pass

            elif sub == "template":
                if local_path:
                    tc = Path(local_path) / "tokenizer_config.json"
                    if tc.exists():
                        try:
                            data = json.loads(tc.read_text())
                            tmpl = data.get("chat_template")
                            if tmpl:
                                console.print(tmpl)
                            else:
                                console.print("[dim]No chat_template in tokenizer_config.json.[/dim]")
                        except Exception as exc:
                            console.print(f"[red]Could not read tokenizer_config.json: {exc}[/red]")
                    else:
                        console.print("[dim]tokenizer_config.json not found.[/dim]")
                else:
                    console.print("[dim]Model not downloaded yet.[/dim]")

            elif sub == "?":
                console.print("[bold]/show options:[/bold]")
                console.print("  [cyan]info[/cyan]        Show model details and session state")
                console.print("  [cyan]license[/cyan]     Show model license")
                console.print("  [cyan]modelfile[/cyan]   Show model config (FROM, SYSTEM, PARAMETER)")
                console.print("  [cyan]parameters[/cyan]  Show generation parameters")
                console.print("  [cyan]system[/cyan]      Show system message")
                console.print("  [cyan]template[/cyan]    Show chat template")
            else:
                console.print(f"[red]Unknown /show option: {sub}[/red]")
                console.print("[bold]/show options:[/bold]")
                console.print("  [cyan]info[/cyan]        Show model details and session state")
                console.print("  [cyan]license[/cyan]     Show model license")
                console.print("  [cyan]modelfile[/cyan]   Show model config (FROM, SYSTEM, PARAMETER)")
                console.print("  [cyan]parameters[/cyan]  Show generation parameters")
                console.print("  [cyan]system[/cyan]      Show system message")
                console.print("  [cyan]template[/cyan]    Show chat template")

        elif user_input == "/clear":
            sys_msgs = [m for m in messages if m["role"] == "system"]
            messages = sys_msgs
            console.print("[dim]Conversation cleared.[/dim]")

        elif user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            try:
                repo_id = resolve_alias(new_model)
                local_path = get_model_path(repo_id)
                model = new_model
                console.print(f"[dim]Switched to {model}[/dim]")
            except ModelNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")

        else:
            # Parse image references from input: [/path/to/img.jpg] or bare paths
            import re as _re
            _IMG_EXTS = r"\.(?:jpg|jpeg|png|gif|webp|bmp)"
            _bracket = _re.compile(r"\[([^\[\]]+?" + _IMG_EXTS + r")\]", _re.IGNORECASE)
            _bare    = _re.compile(r"(?<!\S)((?:/|~/)[\S]+" + _IMG_EXTS + r")(?!\S)", _re.IGNORECASE)

            image_paths = [m.group(1) for m in _bracket.finditer(user_input)]
            clean_input = _bracket.sub("", user_input)
            for m in _bare.finditer(clean_input):
                image_paths.append(m.group(1))
            clean_input = _bare.sub("", clean_input).strip()
            text_input  = clean_input or user_input  # fallback if only path was typed

            # Build user content: structured if images present, plain string otherwise
            if image_paths:
                user_content: object = [{"type": "text", "text": text_input}] + [
                    {"type": "image_url", "image_url": {"url": p}} for p in image_paths
                ]
            else:
                user_content = user_input

            if history_enabled:
                messages.append({"role": "user", "content": user_content})

            # Build messages to send (inject JSON instruction if needed)
            send_msgs = list(messages) if history_enabled else []
            if not history_enabled:
                sys_msgs = [m for m in messages if m["role"] == "system"]
                send_msgs = sys_msgs + [{"role": "user", "content": user_content}]
            if format_json:
                if send_msgs and send_msgs[0]["role"] == "system":
                    send_msgs[0] = {
                        "role": "system",
                        "content": send_msgs[0]["content"] + "\n\nRespond only with valid JSON.",
                    }
                else:
                    send_msgs.insert(0, {"role": "system", "content": "Respond only with valid JSON."})

            console.print("[bold green]Assistant:[/bold green] ", end="")
            full_response = ""
            try:
                if image_paths:
                    # Vision path — no streaming support in mlx-vlm
                    from ppmlx.engine_vlm import get_vision_engine
                    import time as _time
                    t0 = _time.monotonic()
                    text, prompt_toks, completion_toks = get_vision_engine().generate(
                        repo_id, send_msgs,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 2048,
                    )
                    elapsed = _time.monotonic() - t0
                    console.print(text, no_wrap=not wordwrap)
                    full_response = text
                    if verbose:
                        tps = completion_toks / elapsed if elapsed > 0 else 0
                        console.print(
                            f"[dim]prompt {prompt_toks} tokens  "
                            f"completion {completion_toks} tokens  "
                            f"{tps:.1f} tok/s  {elapsed:.2f}s[/dim]"
                        )
                elif verbose:
                    import time as _time
                    t0 = _time.monotonic()
                    text, reasoning, prompt_toks, completion_toks = engine.generate(
                        repo_id, send_msgs,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 2048,
                        strip_thinking=not think,
                        enable_thinking=think,
                    )
                    elapsed = _time.monotonic() - t0
                    if think and reasoning:
                        console.print()
                        console.print(f"[dim italic]{reasoning}[/dim italic]")
                        console.print()
                    console.print(text, no_wrap=not wordwrap)
                    full_response = text
                    tps = completion_toks / elapsed if elapsed > 0 else 0
                    console.print(
                        f"[dim]prompt {prompt_toks} tokens  "
                        f"completion {completion_toks} tokens  "
                        f"{tps:.1f} tok/s  {elapsed:.2f}s[/dim]"
                    )
                else:
                    # Streaming with think-tag handling
                    in_think = False
                    for chunk in engine.stream_generate(
                        repo_id, send_msgs,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 2048,
                        enable_thinking=think,
                    ):
                        if "<think>" in chunk and not in_think:
                            before, _, after = chunk.partition("<think>")
                            if before:
                                console.print(before, end="")
                                full_response += before
                            in_think = True
                            chunk = after
                        if "</think>" in chunk and in_think:
                            inside, _, after = chunk.partition("</think>")
                            if think and inside:
                                console.print(f"[dim italic]{inside}[/dim italic]", end="")
                            in_think = False
                            chunk = after
                        if chunk:
                            if in_think:
                                if think:
                                    console.print(chunk, end="", style="dim italic")
                            else:
                                console.print(chunk, end="")
                                full_response += chunk
                    console.print()
            except KeyboardInterrupt:
                console.print("\n[dim]Generation interrupted.[/dim]")
                if history_enabled:
                    messages.pop()
                continue
            except Exception as exc:
                console.print(f"\n[red]Error: {exc}[/red]")
                if history_enabled:
                    messages.pop()
                continue

            if history_enabled:
                messages.append({"role": "assistant", "content": full_response})
            continue

        continue


def _do_pull(
    model: str,
    token: Optional[str],
    *,
    do_quantize: bool = False,
    bits: int = 4,
    keep_original: bool = False,
) -> bool:
    """Download a single model and print result. Returns True on success."""
    from ppmlx.models import download_model, resolve_alias, ModelNotFoundError
    from ppmlx.memory import check_memory_warning

    try:
        repo_id = resolve_alias(model)
    except ModelNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return False

    console.print(f"[blue]Pulling [bold]{model}[/bold] ({repo_id})[/blue]")
    try:
        local_path = download_model(model, token=token)
        console.print(f"[green]✓ Downloaded to {local_path}[/green]")
        _track_usage("model_pulled", {"used_token": bool(token)})
        warning = check_memory_warning(local_path)
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Download cancelled.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Pull failed: {e}[/red]")
        return False

    if do_quantize:
        from ppmlx.quantize import (
            quantize as run_quantize,
            QuantizeConfig,
            QuantizationError,
        )

        console.print(
            f"[blue]Quantizing [bold]{model}[/bold] to {bits}-bit...[/blue]"
        )
        cfg = QuantizeConfig(bits=bits, hf_token=token)
        try:
            quantized_path = run_quantize(
                model,
                cfg,
                progress_callback=lambda msg: console.print(
                    f"  [dim]{msg}[/dim]"
                ),
                local_path=local_path,
            )
            console.print(
                f"[green]✓ Quantized model saved to {quantized_path}[/green]"
            )
        except QuantizationError as e:
            console.print(f"[red]Quantization failed: {e}[/red]")
            return False

        if not keep_original:
            console.print(
                f"[dim]Removing original download at {local_path}...[/dim]"
            )
            shutil.rmtree(local_path, ignore_errors=True)
            console.print("[dim]Original removed.[/dim]")

    return True


@app.command()
def pull(
    model: Optional[str] = typer.Argument(None, help="Model alias or HuggingFace repo ID (omit for interactive selector)"),
    token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token"),
    do_quantize: bool = typer.Option(False, "--quantize", "-q", help="Quantize the model after downloading"),
    bits: int = typer.Option(4, "--bits", help="Quantization bit depth (2, 3, 4, 6, or 8)"),
    keep_original: bool = typer.Option(False, "--keep-original", help="Keep the full-precision download after quantization"),
):
    """Download a model from HuggingFace Hub (interactive multiselect when no model given)."""
    if do_quantize and bits not in _VALID_QUANTIZE_BITS:
        console.print(f"[red]Invalid --bits value: {bits}. Must be one of {sorted(_VALID_QUANTIZE_BITS)}.[/red]")
        raise typer.Exit(1)

    if model is None:
        from ppmlx.tui import pick_models

        selected = pick_models(local_only=False)
        if not selected:
            console.print("[dim]Nothing selected.[/dim]")
            return

        for m in selected:
            _do_pull(m, token, do_quantize=do_quantize, bits=bits, keep_original=keep_original)
        return

    if not _do_pull(model, token, do_quantize=do_quantize, bits=bits, keep_original=keep_original):
        raise typer.Exit(1)


@app.command(name="list")
def list_models(
    all_models: bool = typer.Option(False, "--all", "-a", help="Show all models (local + registry)"),
):
    """List models. Shows downloaded models by default, --all includes registry."""
    _track_usage("list_models", {"all_models": all_models})

    rows = _build_picker_rows(local_only=not all_models)
    if not any(r.section_header is None for r in rows):
        console.print("[dim]No models downloaded yet. Run: ppmlx pull <model>[/dim]")
        return

    from ppmlx.tui import browse_models
    title = "Models" if all_models else "Local Models"
    browse_models(rows, title=title, command_str="ppmlx list")


@app.command()
def rm(
    model: Optional[str] = typer.Argument(None, help="Model alias or name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a locally downloaded model."""
    _track_usage("remove_model", {"force": force, "interactive_model_pick": model is None})
    from ppmlx.models import remove_model, get_model_path, resolve_alias

    models_to_remove: list[str] = []
    if not model:
        selected = _pick_model(local_only=True, multi=True)
        models_to_remove = selected  # type: ignore[assignment]
    else:
        models_to_remove = [model]

    for m in models_to_remove:
        try:
            repo_id = resolve_alias(m)
        except Exception:
            repo_id = m

        path = get_model_path(repo_id)
        if not path:
            console.print(f"[yellow]Model '{m}' not found locally.[/yellow]")
            continue

        if not force:
            confirm = typer.confirm(f"Remove '{m}' from {path}?")
            if not confirm:
                continue

        if remove_model(m):
            console.print(f"[green]Removed {m}[/green]")
        else:
            console.print(f"[red]Failed to remove {m}[/red]")


@app.command()
def ps():
    """Show currently loaded models and memory usage."""
    _track_usage("ps_checked")
    import httpx
    from ppmlx.config import load_config

    cfg = load_config()
    url = f"http://{cfg.server.host}:{cfg.server.port}/health"

    try:
        response = httpx.get(url, timeout=3.0)
        data = response.json()
        loaded = data.get("loaded_models", [])
        uptime = data.get("uptime_seconds", 0)

        if not loaded:
            console.print("[dim]No models currently loaded. Start server: ppmlx serve[/dim]")
            return

        loaded_set = set(loaded)
        records = _build_model_records()
        # Build picker rows for loaded models only
        rows: list[_PickerRow] = []
        for r in records:
            if r.alias in loaded_set or r.repo_id in loaded_set:
                r.is_loaded = True
                rows.append(_PickerRow(
                    alias=r.alias, size_gb=r.size_gb, downloaded=r.is_downloaded,
                    section_header=None, params_b=r.params_b,
                    is_loaded=True, is_favorite=r.is_favorite,
                ))
                loaded_set.discard(r.alias)
                loaded_set.discard(r.repo_id)
        # Models not in registry
        for name in loaded_set:
            rows.append(_PickerRow(
                alias=name, size_gb=None, downloaded=False,
                section_header=None, is_loaded=True,
            ))

        from ppmlx.tui import browse_models
        browse_models(
            rows, title="Loaded Models", command_str="ppmlx ps",
            footer_extra=f"Server uptime: {uptime}s",
        )
    except Exception:
        console.print("[yellow]Server not running. Start it with: ppmlx serve[/yellow]")


@app.command()
def quantize(
    model: Optional[str] = typer.Argument(None, help="HuggingFace repo ID or alias"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (2,3,4,6,8)"),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    upload: Optional[str] = typer.Option(None, "--upload-repo", help="HF repo to upload to"),
    token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token"),
):
    """Convert and quantize a HuggingFace model to MLX format."""
    _track_usage(
        "quantize_started",
        {"bits": bits, "group_size": group_size, "upload": bool(upload)},
    )
    if not model:
        model = _pick_model()
    from ppmlx.quantize import quantize as do_quantize, QuantizeConfig

    cfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        output_path=Path(output) if output else None,
        upload_repo=upload,
        hf_token=token,
    )

    try:
        path = do_quantize(model, cfg, progress_callback=lambda msg: console.print(f"[blue]{msg}[/blue]"))
        console.print(f"[green]Quantized model saved to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Quantization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="config")
def config_cmd(
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="Set HuggingFace token"),
    thinking: Optional[bool] = typer.Option(
        None,
        "--thinking/--no-thinking",
        help="Enable or disable thinking mode for reasoning models.",
    ),
    reasoning_budget: Optional[int] = typer.Option(
        None,
        "--reasoning-budget",
        help="Default max tokens for thinking phase (0 = unlimited).",
    ),
    effort_base: Optional[int] = typer.Option(
        None,
        "--effort-base",
        help="Base tokens for effort mapping (low=base, medium=base*4, high=base*32).",
    ),
    max_tools_tokens: Optional[int] = typer.Option(
        None,
        "--max-tools-tokens",
        help="Max tokens for tool definitions (0 = unlimited).",
    ),
    analytics: Optional[bool] = typer.Option(
        None,
        "--analytics/--no-analytics",
        help="Enable or disable anonymous usage analytics.",
    ),
):
    """View or interactively set ppmlx configuration (HF token, defaults, etc.)."""
    import tomllib
    import tomli_w  # type: ignore[import]
    from ppmlx.config import get_ppmlx_dir

    cfg_path = get_ppmlx_dir() / "config.toml"

    # Load existing config
    try:
        with open(cfg_path, "rb") as f:
            data: dict = tomllib.load(f)
    except Exception:
        data = {}

    # Non-interactive: apply any flags passed via CLI
    has_flag = any(v is not None for v in [hf_token, thinking, reasoning_budget, effort_base, max_tools_tokens, analytics])
    if has_flag:
        if hf_token is not None:
            data.setdefault("auth", {})["hf_token"] = hf_token
        if thinking is not None:
            data.setdefault("thinking", {})["enabled"] = thinking
        if reasoning_budget is not None:
            data.setdefault("thinking", {})["default_reasoning_budget"] = reasoning_budget
        if effort_base is not None:
            data.setdefault("thinking", {})["effort_base"] = effort_base
        if max_tools_tokens is not None:
            data.setdefault("server", {})["max_tools_tokens"] = max_tools_tokens
        if analytics is not None:
            data.setdefault("analytics", {})["enabled"] = analytics
        try:
            cfg_path.write_bytes(tomli_w.dumps(data).encode())
        except Exception as exc:
            console.print(f"[red]Failed to write config: {exc}[/red]")
            raise typer.Exit(1)
        msgs = []
        if hf_token is not None:
            msgs.append("HuggingFace token saved")
        if thinking is not None:
            msgs.append(f"Thinking {'enabled' if thinking else 'disabled'}")
        if reasoning_budget is not None:
            msgs.append(f"Reasoning budget set to {reasoning_budget} tokens")
        if effort_base is not None:
            msgs.append(f"Effort base set to {effort_base} (low={effort_base}, med={effort_base*4}, high={effort_base*32})")
        if max_tools_tokens is not None:
            msgs.append(f"Max tools tokens set to {'unlimited' if max_tools_tokens == 0 else max_tools_tokens}")
        if analytics is not None:
            msgs.append(f"Analytics {'enabled' if analytics else 'disabled'}")
        console.print(f"[green]{' | '.join(msgs)}.[/green]")
        console.print(f"[dim]{cfg_path}[/dim]")
        return

    # Interactive TUI config
    from ppmlx.tui import config_menu
    config_menu()


_NO_DB_MSG = "[yellow]No database found. Start the server with `ppmlx serve` to begin logging.[/yellow]"
_NO_REQUESTS_MSG = "[yellow]No requests logged yet. Start the server with `ppmlx serve` to begin logging.[/yellow]"


def _open_log_db():
    """Return the log Database, or print a message and raise typer.Exit if DB doesn't exist."""
    from ppmlx.db import get_db
    from ppmlx.config import get_ppmlx_dir

    db_path = get_ppmlx_dir() / "ppmlx.db"
    if not db_path.exists():
        console.print(_NO_DB_MSG)
        raise typer.Exit()
    return get_db(db_path)


@app.command()
def logs(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of requests to show"),
    model: str = typer.Option(None, "--model", "-m", help="Filter by model alias"),
    since: float = typer.Option(None, "--since", "-s", help="Hours to look back"),
    errors: bool = typer.Option(False, "--errors", "-e", help="Show only errors"),
    slow: float = typer.Option(None, "--slow", help="Min duration in ms"),
    thinking: bool = typer.Option(False, "--thinking", "-t", help="Show only thinking-enabled requests"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Query and display request history from the log database."""
    from rich.table import Table

    db = _open_log_db()
    rows = db.query_requests(
        limit=limit, model=model, since_hours=since,
        errors_only=errors, min_duration_ms=slow,
    )

    if thinking:
        rows = [r for r in rows if r.get("thinking_enabled")]

    if not rows:
        console.print(_NO_REQUESTS_MSG)
        raise typer.Exit()

    if json_output:
        console.print(json.dumps(rows, indent=2, default=str))
        raise typer.Exit()

    has_reasoning = any(r.get("reasoning_tokens") for r in rows)

    table = Table(title="Request History")
    table.add_column("Timestamp", style="dim")
    table.add_column("Model")
    table.add_column("Duration (ms)", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Status")
    if has_reasoning:
        table.add_column("Reasoning", justify="right")

    for r in rows:
        ts = str(r.get("timestamp", ""))[:16]
        model_alias = r.get("model_alias", "")

        dur = r.get("total_duration_ms")
        if dur is not None:
            dur_val = float(dur)
            if dur_val < 1000:
                dur_str = f"[green]{dur_val:.0f}[/green]"
            elif dur_val < 5000:
                dur_str = f"[yellow]{dur_val:.0f}[/yellow]"
            else:
                dur_str = f"[red]{dur_val:.0f}[/red]"
        else:
            dur_str = "-"

        tps = r.get("tokens_per_second")
        tps_str = f"{float(tps):.1f}" if tps is not None else "-"

        ttft = r.get("time_to_first_token_ms")
        ttft_str = f"{float(ttft):.0f}" if ttft is not None else "-"

        prompt_t = r.get("prompt_tokens", 0) or 0
        comp_t = r.get("completion_tokens", 0) or 0
        tok_str = f"{prompt_t}/{comp_t}"

        status = r.get("status", "ok")
        status_str = "[green]ok[/green]" if status == "ok" else f"[red]{status}[/red]"

        row_cells = [ts, model_alias, dur_str, tps_str, ttft_str, tok_str, status_str]
        if has_reasoning:
            rt = r.get("reasoning_tokens")
            row_cells.append(str(rt) if rt else "-")
        table.add_row(*row_cells)

    console.print(table)

    durations = [float(r["total_duration_ms"]) for r in rows if r.get("total_duration_ms") is not None]
    tps_vals = [float(r["tokens_per_second"]) for r in rows if r.get("tokens_per_second") is not None]
    avg_dur = f"{sum(durations) / len(durations):.0f}ms" if durations else "N/A"
    avg_tps = f"{sum(tps_vals) / len(tps_vals):.1f}" if tps_vals else "N/A"
    console.print(f"\nShowing {len(rows)} requests | Avg duration: {avg_dur} | Avg tok/s: {avg_tps}")


@app.command()
def stats(
    since: float = typer.Option(24, "--since", "-s", help="Hours to look back"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Display aggregated statistics from the log database."""
    from rich.table import Table

    db = _open_log_db()
    s = db.get_stats(since_hours=since)

    if s["total_requests"] == 0:
        console.print(_NO_REQUESTS_MSG)
        raise typer.Exit()

    if json_output:
        console.print(json.dumps(s, indent=2, default=str))
        raise typer.Exit()

    avg_dur = f"{s['avg_duration_ms']:.0f}ms" if s.get("avg_duration_ms") is not None else "N/A"

    console.print(Panel(
        f"Total requests: [bold]{s['total_requests']}[/bold]  |  Avg duration: [bold]{avg_dur}[/bold]",
        title=f"ppmlx Stats (last {since}h)",
    ))

    if s.get("by_model"):
        table = Table(title="Per-Model Breakdown")
        table.add_column("Model")
        table.add_column("Requests", justify="right")
        table.add_column("Avg Tok/s", justify="right")
        table.add_column("Avg TTFT (ms)", justify="right")
        table.add_column("Errors", justify="right")

        for m in s["by_model"]:
            tps = f"{m['avg_tps']:.1f}" if m.get("avg_tps") is not None else "-"
            ttft = f"{m['avg_ttft']:.0f}" if m.get("avg_ttft") is not None else "-"
            errs = str(m.get("errors", 0))
            table.add_row(m["model"], str(m["count"]), tps, ttft, errs)

        console.print(table)

    if s.get("thinking"):
        t = s["thinking"]
        console.print(Panel(
            f"Thinking requests: [bold]{t.get('count', 0)}[/bold]  |  "
            f"% thinking: [bold]{t.get('pct', 0):.1f}%[/bold]  |  "
            f"Avg reasoning tokens: [bold]{t.get('avg_reasoning_tokens', 'N/A')}[/bold]",
            title="Thinking Stats",
        ))


@app.command()
def bench(
    model: str = typer.Argument(..., help="Model name or alias to benchmark"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of iterations per scenario"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save JSON results to this path"),
    scenarios: Optional[str] = typer.Option(None, "--scenarios", "-s", help="Comma-separated scenario names (simple,complex,long_context)"),
    compare: Optional[str] = typer.Option(None, "--compare", "-c", help="Compare against a baseline JSON file"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(6767, "--port", "-p", help="Server port"),
    no_auto_server: bool = typer.Option(False, "--no-auto-server", help="Do not auto-start the server"),
):
    """Run standardized benchmarks against a model."""
    from ppmlx.bench import (
        BenchmarkRunner,
        SCENARIOS,
        print_results,
        print_comparison,
        save_results,
        load_results,
    )

    base_url = f"http://{host}:{port}"
    scenario_list = [s.strip() for s in scenarios.split(",")] if scenarios else None

    # Check if server is already running
    server_proc = None
    if not no_auto_server:
        import httpx

        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            resp.raise_for_status()
            console.print(f"[green]Server already running at {base_url}[/green]")
        except Exception:
            console.print(f"[yellow]Starting ppmlx server on {base_url}...[/yellow]")
            server_proc = subprocess.Popen(
                [sys.executable, "-m", "ppmlx.cli", "serve", "--host", host, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for server to become healthy
            healthy = False
            for _ in range(30):
                time.sleep(1)
                try:
                    resp = httpx.get(f"{base_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        healthy = True
                        break
                except Exception:
                    continue
            if not healthy:
                console.print("[red]Server failed to start within 30 seconds.[/red]")
                if server_proc:
                    server_proc.terminate()
                raise typer.Exit(1)
            console.print(f"[green]Server started at {base_url}[/green]")

    try:
        try:
            runner = BenchmarkRunner(
                model=model,
                base_url=base_url,
                runs=runs,
                scenarios=scenario_list,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        result = runner.run()
        print_results(result, console)

        # Save results
        if output:
            out_path = save_results(result, Path(output))
            console.print(f"\n[green]Results saved to {out_path}[/green]")

        # Compare against baseline
        if compare:
            compare_path = Path(compare)
            if not compare_path.exists():
                console.print(f"[red]Baseline file not found: {compare}[/red]")
                raise typer.Exit(1)
            baseline = load_results(compare_path)
            console.print()
            print_comparison(result, baseline, console)

    finally:
        if server_proc:
            console.print("[dim]Stopping auto-started server...[/dim]")
            server_proc.terminate()
            server_proc.wait(timeout=5)


if __name__ == "__main__":
    app()
