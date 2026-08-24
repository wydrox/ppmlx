"""Interactive TUI for route/alias management (ADR 0005 route policies).

All mutations go through ``~/.ppmlx/routes.toml`` (or ``PPMLX_ROUTE_POLICY`` /
``[server] route_policy`` when configured). Writes are atomic with a
timestamped backup (reusing :mod:`ppmlx.onboard` helpers) and every document
is validated through :func:`ppmlx.router.policy_from_dict` before it touches
disk. API keys are never read or displayed here.

The interactive surface (:func:`run_route_tui`) is a full-screen
prompt_toolkit application following the conventions of
:mod:`ppmlx.tui._multi_picker`: ``Layout``/``HSplit``/``Window`` with
``FormattedTextControl`` renderers, a plain-dict cursor state, shared styling
from :mod:`ppmlx.tui._style`, and inline modal forms instead of sequential
prompts. All scriptable helpers (``route_list``, ``route_alias_add``, ...)
remain importable for CLI use.
"""
from __future__ import annotations

import asyncio
import difflib
import os
import tomllib
import tomli_w  # noqa: F401  (re-exported convenience for callers)
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ppmlx.onboard import _backup, routes_toml_path

__all__ = [
    "PROVIDER_KINDS",
    "AliasRow",
    "alias_count",
    "load_route_document",
    "render_alias_table",
    "route_alias_add",
    "route_alias_remove",
    "route_alias_rename",
    "route_list",
    "route_policy_path",
    "route_set",
    "route_test",
    "run_route_tui",
]

console = Console()

PROVIDER_KINDS = (
    # kind, description shown in the picker
    ("openai", "OpenAI-compatible API (api.openai.com or custom base_url)"),
    ("anthropic", "Anthropic Messages API (ANTHROPIC_API_KEY)"),
    (
        "anthropic-subscription",
        "Anthropic via subscription tunnel (locked: no custom base_url)",
    ),
)

# Well-known provider ids with default base URLs. A route candidate may use
# any of these as its ``provider`` id and the correct endpoint + env key are
# picked automatically (no base_url needed in routes.toml).
KNOWN_PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "grok": {"base_url": "https://api.x.ai/v1", "env_key": "XAI_API_KEY"},
    "xai": {"base_url": "https://api.x.ai/v1", "env_key": "XAI_API_KEY"},
    "kimi": {"base_url": "https://api.kimi.com/coding/v1", "env_key": "KIMI_API_KEY"},
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
    },
}

_LOCAL_PROVIDER_IDS = frozenset({"mlx", "local"})
_DEFAULT_DOC: dict = {
    "routes": {
        "version": "1",
        "aliases": {},
        "entries": [],
    }
}


# ---------------------------------------------------------------------------
# Policy document plumbing
# ---------------------------------------------------------------------------


def route_policy_path(*, home: Path | None = None) -> Path:
    """Resolve the active route-policy path.

    Precedence: ``PPMLX_ROUTE_POLICY`` env var, then ``home/.ppmlx/routes.toml``
    (mirrors ``ppmlx.server._route_policy_path``).
    """
    env_path = os.environ.get("PPMLX_ROUTE_POLICY") or ""
    if env_path:
        return Path(env_path).expanduser()
    return routes_toml_path(home)


def load_route_document(path: Path) -> dict:
    """Load the raw route-policy mapping (empty skeleton when absent)."""
    if not path.exists():
        return _skeleton_doc()
    with open(path, "rb") as handle:
        doc = tomllib.load(handle)
    doc.setdefault("routes", {})
    return doc


def _skeleton_doc() -> dict:
    import copy

    return copy.deepcopy(_DEFAULT_DOC)


def _validate(doc: dict) -> None:
    """Validate a raw document against the router's policy rules."""
    from ppmlx.router import policy_from_dict

    policy_from_dict(doc)


def save_route_document(
    doc: dict,
    path: Path,
    *,
    dry_run: bool = False,
) -> Path | None:
    """Atomically persist ``doc`` as TOML with a timestamped backup.

    Returns the backup path when an existing file was replaced, else None.
    Raises ValueError when the document violates router policy rules.
    """

    _validate(doc)
    payload = tomli_w.dumps(doc)
    backup: Path | None = None
    if path.exists():
        backup = _backup(path)
    if dry_run:
        return backup
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)
    return backup


def _diff_text(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )


def _toml_text(path: Path) -> str:

    if not path.exists():
        return ""
    return path.read_text()


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AliasRow:
    alias: str
    provider_id: str
    model_id: str
    base_url: str | None = None


def route_list(*, path: Path | None = None) -> list[AliasRow]:
    """Parse current aliases into rows: alias -> provider/model/base_url."""
    path = path or route_policy_path()
    if not path.exists():
        return []
    try:
        doc = load_route_document(path)
        routes = doc.get("routes", {}) or {}
        raw_aliases = routes.get("aliases", {}) or {}
    except Exception:
        return []
    entries = {
        entry.get("key"): entry
        for entry in routes.get("entries", []) or []
        if isinstance(entry, dict)
    }
    rows: list[AliasRow] = []
    for alias, target in sorted(raw_aliases.items()):
        if (
            not isinstance(target, list)
            or len(target) != 2
            or any(not isinstance(part, str) or not part for part in target)
        ):
            continue
        provider_id, model_id = target[0], target[1]
        base_url = None
        entry = entries.get(f"openai-chat:{alias}")
        if isinstance(entry, dict):
            candidates = entry.get("candidates", [])
            if candidates and isinstance(candidates[0], dict):
                base_url = candidates[0].get("base_url")
                # Entry candidates carry the authoritative model/kind.
                model_id = candidates[0].get("model", model_id)
                provider_id = candidates[0].get("provider", provider_id)
        rows.append(
            AliasRow(
                alias=alias,
                provider_id=provider_id,
                model_id=model_id,
                base_url=base_url,
            )
        )
    return rows


def alias_count(*, path: Path | None = None) -> int:
    return len(route_list(path=path))


def render_alias_table(rows: list[AliasRow]) -> Table:
    table = Table(title="Route aliases", expand=False)
    table.add_column("Alias", style="cyan bold")
    table.add_column("Provider", style="magenta")
    table.add_column("Model")
    table.add_column("Base URL", style="dim")
    table.add_column("Status")
    for row in rows:
        status = "[green]ok[/green]"
        note = ""
        if row.provider_id in _LOCAL_PROVIDER_IDS:
            status, note = "[yellow]local[/yellow]", ""
        elif row.provider_id == "openai":
            note = "OPENAI_API_KEY"
        elif row.provider_id.startswith("anthropic"):
            note = "ANTHROPIC_API_KEY"
        table.add_row(
            row.alias,
            row.provider_id,
            row.model_id,
            row.base_url or "",
            f"{status} [dim]{note}[/dim]" if note else status,
        )
    return table


# ---------------------------------------------------------------------------
# Mutations (scriptable core used by both CLI and the wizard)
# ---------------------------------------------------------------------------


def route_alias_add(
    alias: str,
    provider_id: str,
    model_id: str,
    *,
    base_url: str | None = None,
    path: Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Add one alias (plus its openai-chat route entry). Returns backup path."""
    if not alias or ":" in alias or "/" in alias:
        raise ValueError("Alias name must be non-empty and contain no ':' or '/'")
    if not model_id:
        raise ValueError("Model id must be non-empty")
    path = path or route_policy_path()
    doc = load_route_document(path) if path.exists() else _skeleton_doc()
    routes = doc.setdefault("routes", {})
    routes.setdefault("version", "1")
    aliases = routes.setdefault("aliases", {})
    if alias in aliases:
        raise ValueError(f"Alias {alias!r} already exists (use rename or remove)")
    aliases[alias] = [provider_id, model_id]

    candidate: dict = {"provider": provider_id, "model": model_id}
    if base_url:
        candidate["base_url"] = base_url
    if provider_id.startswith("anthropic-subscription"):
        candidate["provider_kind"] = "anthropic"
    elif provider_id.startswith("anthropic"):
        candidate["provider_kind"] = "anthropic"
    else:
        candidate.setdefault("provider_kind", "openai")
    entries = routes.setdefault("entries", [])
    key = f"openai-chat:{alias}"
    if any(isinstance(e, dict) and e.get("key") == key for e in entries):
        raise ValueError(f"A route entry for {key!r} already exists")
    entries.append(
        {"key": key, "candidates": [candidate], "fallback_errors": []}
    )
    return save_route_document(doc, path, dry_run=dry_run)


def route_alias_remove(
    alias: str,
    *,
    path: Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Remove one alias and its matching route entry."""
    path = path or route_policy_path()
    doc = load_route_document(path)
    routes = doc.get("routes", {})
    aliases = routes.get("aliases", {})
    if alias not in aliases:
        raise ValueError(f"Alias {alias!r} is not defined")
    del aliases[alias]
    key = f"openai-chat:{alias}"
    routes["entries"] = [
        e
        for e in routes.get("entries", [])
        if not (isinstance(e, dict) and e.get("key") == key)
    ]
    return save_route_document(doc, path, dry_run=dry_run)


def route_alias_rename(
    old: str,
    new: str,
    *,
    path: Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Rename an alias in place (single atomic write)."""
    if not new or ":" in new or "/" in new:
        raise ValueError("New alias name must be non-empty with no ':' or '/'")
    path = path or route_policy_path()
    doc = load_route_document(path)
    routes = doc.get("routes", {})
    aliases = routes.get("aliases", {})
    if old not in aliases:
        raise ValueError(f"Alias {old!r} is not defined")
    if new in aliases:
        raise ValueError(f"Alias {new!r} already exists")
    aliases[new] = aliases.pop(old)
    old_key, new_key = f"openai-chat:{old}", f"openai-chat:{new}"
    for entry in routes.get("entries", []):
        if isinstance(entry, dict) and entry.get("key") == old_key:
            entry["key"] = new_key
    return save_route_document(doc, path, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Interactive pieces (non-TUI wizard kept for `ppmlx route set`)
# ---------------------------------------------------------------------------


def _ask_provider_kind() -> tuple[str, str]:
    """Arrow-key provider-kind selection with graceful prompt fallback."""
    try:
        import questionary

        choice = questionary.select(
            "Provider kind:",
            choices=[
                questionary.Choice(f"{label} — {desc}", value=label)
                for label, desc in PROVIDER_KINDS
            ],
        ).ask()
        if choice is None:
            raise KeyboardInterrupt
        return choice, ""
    except ImportError:
        from rich.prompt import Prompt

        labels = [label for label, _ in PROVIDER_KINDS]
        raw = Prompt.ask(
            f"Provider kind ({'/'.join(labels)})",
            choices=labels,
            default="openai",
        )
        desc = next(desc for label, desc in PROVIDER_KINDS if label == raw)
        return raw, "" if raw != "anthropic-subscription" else desc


def _print_diff(path: Path, doc: dict) -> str:

    before = _toml_text(path)
    after = tomli_w.dumps(doc)
    diff = _diff_text(before, after)
    colored = "\n".join(
        (
            f"[green]{line}[/green]"
            if line.startswith("+") and not line.startswith("+++")
            else f"[red]{line}[/red]"
            if line.startswith("-") and not line.startswith("---")
            else f"[dim]{line}[/dim]"
        )
        for line in diff.splitlines()
    )
    console.print(Panel(colored or "[dim]no textual change[/dim]", title="Diff"))
    return after


def route_set(
    alias: str,
    *,
    path: Path | None = None,
    dry_run: bool = False,
) -> AliasRow:
    """Wizard: define/replace ``alias`` with live preview + diff before write."""
    from rich.prompt import Confirm, Prompt

    if not alias:
        alias = Prompt.ask("Alias name").strip()
    path = path or route_policy_path()

    while True:
        provider_kind, _note = _ask_provider_kind()
        base_url: str | None = None
        if provider_kind == "anthropic-subscription":
            console.print(
                "[yellow]anthropic-subscription is locked: it always uses the "
                "built-in subscription tunnel; base_url is ignored.[/yellow]"
            )
        elif provider_kind in KNOWN_PROVIDERS:
            # Known provider: default base URL comes from the registry, so a
            # bare alias like `model-ciężki = ["grok", "grok-4.6"]` just works.
            base_url = KNOWN_PROVIDERS[provider_kind]["base_url"]
            console.print(
                f"[dim]{provider_kind}: using {base_url} "
                f"(key: {KNOWN_PROVIDERS[provider_kind]['env_key']})[/dim]"
            )
        else:
            default_base = (
                "https://api.anthropic.com/v1"
                if provider_kind.startswith("anthropic")
                else "https://api.openai.com/v1"
            )
            raw = Prompt.ask(
                "Custom base URL (blank = default)", default=""
            ).strip()
            if raw:
                base_url = raw
            else:
                base_url = None
            console.print(f"[dim]Default base URL: {default_base}[/dim]")
        model_id = Prompt.ask("Model id (e.g. gpt-5-mini)").strip()
        if model_id:
            break
        console.print("[red]Model id is required.[/red]")

    # Optional fallback candidates.
    fallbacks: list[tuple[str, str]] = []
    if Confirm.ask("Add fallback candidate(s)?", default=False):
        while True:
            fp = Prompt.ask("Fallback provider id").strip()
            fm = Prompt.ask("Fallback model id").strip()
            if fp and fm:
                fallbacks.append((fp, fm))
            if not Confirm.ask("Add another fallback?", default=False):
                break

    # Build the prospective document without touching disk.
    probe = (
        load_route_document(path)
        if path.exists()
        else _skeleton_doc()
    )
    if alias in (probe.get("routes", {}).get("aliases", {}) or {}):
        if not Confirm.ask(
            f"Alias {alias!r} exists — replace it?", default=False
        ):
            console.print("[yellow]Aborted — nothing written.[/yellow]")
            raise SystemExit(1)
        route_alias_remove(alias, path=path, dry_run=True)  # validation only
    doc = (
        load_route_document(path)
        if path.exists()
        else _skeleton_doc()
    )
    routes = doc.setdefault("routes", {})
    routes.setdefault("version", "1")
    routes.setdefault("aliases", {})[alias] = [provider_kind, model_id]
    candidates: list[dict] = []
    primary: dict = {"provider": provider_kind, "model": model_id}
    if base_url and provider_kind != "anthropic-subscription":
        primary["base_url"] = base_url
    if provider_kind.startswith("anthropic"):
        primary["provider_kind"] = "anthropic"
    else:
        primary.setdefault("provider_kind", "openai")
    candidates.append(primary)
    for fp, fm in fallbacks:
        fb: dict = {"provider": fp, "model": fm}
        if fp.startswith("anthropic"):
            fb["provider_kind"] = "anthropic"
        else:
            fb.setdefault("provider_kind", "openai")
        candidates.append(fb)
    key = f"openai-chat:{alias}"
    entries = [
        e
        for e in routes.setdefault("entries", [])
        if not (isinstance(e, dict) and e.get("key") == key)
    ]
    entries.append({"key": key, "candidates": candidates, "fallback_errors": []})
    routes["entries"] = entries

    _print_diff(path, doc)
    try:
        _validate(doc)  # raises ValueError against router.load_policy rules
    except ValueError as exc:
        console.print(f"[red]Route policy validation failed: {exc}[/red]")
        console.print(
            "[yellow]Fix the highlighted field in the policy file and retry.[/yellow]"
        )
        raise SystemExit(1) from None
    if not Confirm.ask("Write this route policy?", default=True):
        console.print("[yellow]Aborted — nothing written.[/yellow]")
        raise SystemExit(1)
    backup = save_route_document(doc, path, dry_run=dry_run)
    if backup is not None:
        console.print(f"[dim]Backup written: {backup}[/dim]")
    console.print(f"[green]Saved {path}[/green]")
    return AliasRow(
        alias=alias,
        provider_id=provider_kind,
        model_id=model_id,
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Live test request
# ---------------------------------------------------------------------------


def _build_service():
    """Build the real RoutingService (same wiring as ppmlx serve)."""
    from ppmlx.server import _get_remote_routing_service

    return _get_remote_routing_service()


def route_test(
    alias: str,
    *,
    prompt: str = "Say 'pong' and nothing else.",
) -> int:
    """Send a real test request through the routing service.

    Prints which candidate served the response. Returns 0 on success, 1 on a
    typed routing/provider failure. Never prints API keys.
    """
    service = _build_service()
    if service is None:
        console.print(
            "[red]No route policy configured.[/red] Run [bold]ppmlx onboard[/bold] "
            "with route setup, or set PPMLX_ROUTE_POLICY."
        )
        return 1
    if not service.is_remote_model(alias):
        console.print(
            f"[red]Alias {alias!r} is not defined in the active route policy."
            f"[/red] Configured: {', '.join(sorted(service.policy.aliases)) or 'none'}"
        )
        return 1

    # Prime keychain-stored credentials into the environment (same as serve).
    from ppmlx.routing_service import prime_provider_credentials

    provider_ids = tuple(
        {c.provider_id for entry in service.policy.entries.values() for c in entry.candidates}
    )
    primed = prime_provider_credentials(provider_ids)
    missing = sorted(set(provider_ids) - set(primed))
    if missing:
        console.print(
            f"[yellow]No stored credential for: {', '.join(missing)}. "
            f"Run [bold]ppmlx auth add <provider>[/bold] first.[/yellow]"
        )

    body = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
    }
    from ppmlx.protocols.base import DecodeContext
    from ppmlx.protocols.openai_chat import OpenAIChatAdapter
    from ppmlx.routing_service import RoutingServiceError
    from ppmlx.router import RouteInput

    import time
    import uuid
    from ppmlx import __version__

    request_id = "req_tuitest_" + uuid.uuid4().hex[:8]
    try:
        decoded = OpenAIChatAdapter().decode_request(
            body,
            context=DecodeContext(request_id=request_id, kind="initial"),
        )
        route_input = RouteInput(
            public_model=alias,
            harness="openai-chat",
            harness_version=__version__,
            protocol="openai-chat",
            required=_test_required_capabilities(),
            policy_version=service.policy.version,
            health_snapshot_id="auto",
            request_id=request_id,
            session_id="sess-route-tui",
        )
        started = time.time()
        result = service.execute(route_input, decoded.request)
    except RoutingServiceError as error:
        console.print(f"[red]Routing failed:[/red] {error.code}")
        return 1
    except Exception as error:  # secret-free surface only
        console.print(
            f"[red]Test failed:[/red] {type(error).__name__}: {_one_line(error)}"
        )
        return 1

    elapsed_ms = int((time.time() - started) * 1000)
    text_parts: list[str] = []
    for event in result.events:
        delta = getattr(event, "delta", None)
        if getattr(event, "type", "") == "content.delta" and isinstance(delta, str):
            text_parts.append(delta)
    selected = result.decision.selected
    console.print(
        Panel(
            "".join(text_parts).strip() or "[dim](empty response)[/dim]",
            title=f"{alias} · {elapsed_ms} ms",
        )
    )
    if selected is not None:
        console.print(
            f"[green]Served by candidate[/green] provider="
            f"[magenta]{selected.provider_id}[/magenta] model=[cyan]{selected.model}[/cyan]"
        )
    return 0


def _test_required_capabilities():
    from ppmlx.router import RequiredCapabilities

    return RequiredCapabilities(text=True)


def _one_line(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text[:200] if text else type(error).__name__


# ---------------------------------------------------------------------------
# Full-screen prompt_toolkit TUI
# ---------------------------------------------------------------------------


def _route_entry_for(doc: dict, alias: str) -> dict | None:
    """Return the ``openai-chat:<alias>`` route entry mapping, if present."""
    for entry in doc.get("routes", {}).get("entries") or []:
        if isinstance(entry, dict) and entry.get("key") == f"openai-chat:{alias}":
            return entry
    return None


def _status_hint(provider_id: str) -> str:
    """Short status column hint (credential expectation), never a secret."""
    if provider_id in _LOCAL_PROVIDER_IDS:
        return "local"
    if provider_id == "openai":
        return "OPENAI_API_KEY"
    if provider_id.startswith("anthropic"):
        return "ANTHROPIC_API_KEY"
    return ""


def _candidate_dict(kind: str, model: str, base_url: str) -> dict:
    cand: dict = {"provider": kind, "model": model}
    if base_url and kind != "anthropic-subscription":
        cand["base_url"] = base_url
    if kind.startswith("anthropic"):
        cand["provider_kind"] = "anthropic"
    else:
        cand.setdefault("provider_kind", "openai")
    return cand


def run_route_tui(*, path: Path | None = None) -> None:
    """Full-screen alias manager: table + detail panel + inline add/edit form."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import (
        ConditionalContainer,
        DynamicContainer,
        HSplit,
        Layout,
        ScrollOffsets,
        VSplit,
        Window,
    )
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    from prompt_toolkit.document import Document as _PTDocument

    from ppmlx.tui._style import get_style

    path = path or route_policy_path()

    state: dict = {
        "doc": load_route_document(path),
        "cursor": 0,
        "search": "",
        "mode": "table",  # table | form
        "confirm_delete": False,
        "flash": "",       # "" or "!error:<msg>"
        "detail": "",      # live-test summary lines for the detail panel
        "testing": False,
    }
    form: dict = {
        "kind_idx": 0,
        "edit_alias": None,
        "stop_buffers": [],   # ordered Buffers for tab cycling (incl. kind)
        "kind_stop": None,    # index of the provider-kind stop
    }

    # -- document plumbing --------------------------------------------------

    def _reload() -> None:
        state["doc"] = load_route_document(path)

    def _rows() -> list[dict]:
        rows: list[dict] = []
        doc = state["doc"]
        routes = doc.get("routes", {}) or {}
        raw_aliases = routes.get("aliases", {}) or {}
        entries_index = {
            e.get("key"): e
            for e in routes.get("entries", []) or []
            if isinstance(e, dict)
        }
        for alias, target in sorted(raw_aliases.items()):
            if (
                not isinstance(target, list)
                or len(target) != 2
                or any(not isinstance(p, str) or not p for p in target)
            ):
                continue
            provider_id, target_model = target
            entry = entries_index.get(f"openai-chat:{alias}")
            base_url = None
            model_id = target_model
            cands = (entry or {}).get("candidates") or []
            if cands and isinstance(cands[0], dict):
                base_url = cands[0].get("base_url")
                model_id = cands[0].get("model", model_id)
            rows.append(
                {
                    "alias": alias,
                    "provider": provider_id,
                    "model": model_id,
                    "base_url": base_url,
                    "entry": entry,
                    "hint": _status_hint(provider_id),
                }
            )
        needle = state["search"].lower()
        if needle:
            rows = [
                r
                for r in rows
                if needle in r["alias"].lower()
                or needle in r["provider"].lower()
                or needle in (r["model"] or "").lower()
            ]
        return rows

    def _selected_row() -> dict | None:
        rows = _rows()
        if not rows:
            return None
        state["cursor"] = max(0, min(state["cursor"], len(rows) - 1))
        return rows[state["cursor"]]

    def _flash(msg: str, *, error: bool = False) -> None:
        state["flash"] = ("!error:" if error else "") + msg

    def _save_doc(doc: dict) -> bool:
        try:
            backup = save_route_document(doc, path)
        except ValueError as exc:
            _flash(f"Validation failed: {exc}", error=True)
            return False
        state["doc"] = doc
        note = f"  [backup: {backup.name}]" if backup else ""
        _flash(f"Saved {path}{note}")
        return True

    def _build_form_doc(alias: str, kind: str, model: str, base_url: str, fallback_lines: list[str]) -> dict:
        doc = load_route_document(path) if path.exists() else _skeleton_doc()
        routes = doc.setdefault("routes", {})
        routes.setdefault("version", "1")
        routes.setdefault("aliases", {})[alias] = [kind, model]
        candidates = [_candidate_dict(kind, model, base_url)]
        for line in fallback_lines:
            parts = line.split(",", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                candidates.append(_candidate_dict(parts[0].strip(), parts[1].strip(), ""))
        key = f"openai-chat:{alias}"
        entries = [
            e
            for e in routes.setdefault("entries", [])
            if not (isinstance(e, dict) and e.get("key") == key)
        ]
        entries.append({"key": key, "candidates": candidates, "fallback_errors": []})
        routes["entries"] = entries
        return doc

    # -- rendering -----------------------------------------------------------

    W_ALIAS, W_PROV, W_MODEL, W_URL = 18, 24, 30, 32

    def _pad(text: object, w: int) -> str:
        s = str(text or "")
        return s[: w - 1] + "\u2026" if len(s) > w else s.ljust(w)

    def _get_header() -> list[tuple[str, str]]:
        return [
            ("class:header", " ppmlx routes"),
            ("class:dim", f"  \u2502 {path}"),
            ("", "\n"),
        ]

    def _get_table() -> list[tuple[str, str]]:
        rows = _rows()
        if not rows:
            return [("class:dim", "   No aliases yet — press a to add one.\n")]
        header = (
            f"     {_pad('alias', W_ALIAS)}{_pad('provider', W_PROV)}"
            f"{_pad('model', W_MODEL)}{_pad('base_url', W_URL)}status\n"
        )
        frag: list[tuple[str, str]] = [
            ("class:table.header", header),
            ("class:table.border", "\u2500" * (len(header) - 1) + "\n"),
        ]
        for i, r in enumerate(rows):
            cur = i == state["cursor"]
            style = "class:cursor" if cur else ""
            prefix = "  \u25b8 " if cur else "    "
            cells = (
                f"{_pad(r['alias'], W_ALIAS)}{_pad(r['provider'], W_PROV)}"
                f"{_pad(r['model'], W_MODEL)}{_pad(r['base_url'] or '', W_URL)}"
                f"{r['hint']}\n"
            )
            frag.append((style, prefix + cells))
        return frag

    def _get_detail() -> list[tuple[str, str]]:
        frag: list[tuple[str, str]] = [("class:section", " Details"), ("", "\n")]
        row = _selected_row()
        if row is None:
            frag.append(("class:dim", "  (no selection)\n"))
            return frag
        frag.append(("class:value", f"  alias      {row['alias']}\n"))
        frag.append(("class:value", f"  provider   {row['provider']}\n"))
        frag.append(("class:value", f"  model      {row['model']}\n"))
        frag.append(("class:value", f"  base_url   {row['base_url'] or '(default)'}\n"))
        entry = row["entry"]
        cands = ((entry or {}).get("candidates")) or []
        if len(cands) > 1:
            frag.append(("class:section", "  fallback candidates"), )
            frag.append(("", "\n"))
            for j, cand in enumerate(cands[1:], start=2):
                if isinstance(cand, dict):
                    frag.append((
                        "class:dim",
                        f"   #{j} {cand.get('provider')} / {cand.get('model')}"
                        f"  {cand.get('base_url') or ''}\n",
                    ))
        fb_errors = (entry or {}).get("fallback_errors") or []
        if fb_errors:
            frag.append(("fg:red bold", f"  fallback_errors: {fb_errors}\n"))
        if state["testing"]:
            frag.append(("class:unsaved", "  testing\u2026 (live request in flight)\n"))
        elif state["detail"]:
            for line in state["detail"].splitlines():
                cls = "fg:red" if line.startswith("!") else "class:dim"
                frag.append((cls, f"  {line}\n"))
        return frag

    def _get_footer() -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = [
            (
                "class:footer",
                " a add \u00b7 e edit \u00b7 d delete \u00b7 t test \u00b7 r refresh"
                " \u00b7 / search \u00b7 q quit",
            ),
        ]
        flash = state["flash"]
        if flash.startswith("!error:"):
            parts.append(("fg:red bold", f"   \u2717 {flash[len('!error:'):]}"))
        elif flash:
            parts.append(("class:checked", f"   \u2713 {flash}"))
        if state["confirm_delete"]:
            parts.append(("class:unsaved", "   delete selected alias? y/n"))
        parts.append(("", "\n"))
        return parts

    # -- form mode -------------------------------------------------------------

    def _form_field_values() -> dict:
        return {
            "alias": form["alias_field"].text.strip(),
            "model": form["model_field"].text.strip(),
            "url": form["url_field"].text.strip(),
            "fallbacks": [
                ln.strip()
                for ln in form["fallback_field"].text.splitlines()
                if ln.strip()
            ],
            "kind": PROVIDER_KINDS[form["kind_idx"]][0],
        }

    def _validate_form() -> str | None:
        vals = _form_field_values()
        if not vals["alias"] or ":" in vals["alias"] or "/" in vals["alias"]:
            return "Alias must be non-empty and contain no ':' or '/'"
        if not vals["model"]:
            return "Model id is required"
        known = set(state["doc"].get("routes", {}).get("aliases", {}) or {})
        if vals["alias"] in known and vals["alias"] != form["edit_alias"]:
            return f"Alias {vals['alias']!r} already exists"
        return None

    def _submit_form(app) -> None:
        err = _validate_form()
        if err:
            _flash(err, error=True)
            app.invalidate()
            return
        vals = _form_field_values()
        try:
            doc = _build_form_doc(
                vals["alias"], vals["kind"], vals["model"], vals["url"], vals["fallbacks"]
            )
        except ValueError as exc:
            _flash(str(exc), error=True)
            app.invalidate()
            return
        if _save_doc(doc):
            state["mode"] = "table"
            state["cursor"] = 0
            _flash(f"Saved alias {vals['alias']!r}")
        app.invalidate()

    def _cancel_form(app) -> None:
        state["mode"] = "table"
        _flash("")
        app.invalidate()

    def _enter_form(app, *, edit: bool) -> None:
        row = _selected_row()
        if edit and row is None:
            _flash("No alias selected", error=True)
            app.invalidate()
            return
        kind_idx = 0
        fallbacks_text = ""
        alias_text = model_text = url_text = ""
        if edit and row is not None:
            cands = ((row["entry"] or {}).get("candidates")) or []
            fallbacks_text = "\n".join(
                f"{c.get('provider', '')}, {c.get('model', '')}"
                for c in cands[1:]
                if isinstance(c, dict)
            )
            for i, (kind, _d) in enumerate(PROVIDER_KINDS):
                if kind == row["provider"]:
                    kind_idx = i
            alias_text = row["alias"]
            model_text = row["model"] or ""
            url_text = row["base_url"] or ""
        form["alias_field"] = TextArea(text=alias_text, multiline=False)
        form["model_field"] = TextArea(text=model_text, multiline=False)
        form["url_field"] = TextArea(text=url_text, multiline=False)
        form["fallback_field"] = TextArea(text=fallbacks_text, multiline=True)
        # Read-only stand-in for the provider-kind selector stop (left/right).
        kind_label, _kind_desc = PROVIDER_KINDS[kind_idx]
        form["kind_field"] = TextArea(text=kind_label, multiline=False, read_only=True)
        form["kind_idx"] = kind_idx
        form["edit_alias"] = row["alias"] if edit else None
        form["stop_buffers"] = [
            form["alias_field"].buffer,
            form["model_field"].buffer,
            form["url_field"].buffer,
            form["fallback_field"].buffer,
            form["kind_field"].buffer,
        ]
        form["kind_stop"] = len(form["stop_buffers"]) - 1
        form["focus_idx"] = 0
        state["mode"] = "form"
        state["confirm_delete"] = False
        _flash("")
        app.layout.focus(form["alias_field"].window)
        app.invalidate()

    def _stop_widget(idx: int):
        return {
            0: form["alias_field"],
            1: form["model_field"],
            2: form["url_field"],
            3: form["fallback_field"],
            form["kind_stop"]: form["kind_field"],
        }[idx]

    def _cycle_focus(app, delta: int) -> None:
        stops = form["stop_buffers"]
        form["focus_idx"] = (form["focus_idx"] + delta) % len(stops)
        # Sync the read-only kind selector text with the current selection.
        if form["focus_idx"] == form["kind_stop"]:
            form["kind_field"].buffer.set_document(
                _PTDocument(PROVIDER_KINDS[form["kind_idx"]][0])
            )
        app.layout.focus(_stop_widget(form["focus_idx"]).window)

    def _on_kind_stop() -> bool:
        return form.get("focus_idx", 0) == form["kind_stop"]

    def _get_form() -> list[tuple[str, str]]:
        kind_label, kind_desc = PROVIDER_KINDS[form["kind_idx"]]
        editing = form["edit_alias"] is not None
        frag: list[tuple[str, str]] = [
            ("class:header", f" {'Edit alias' if editing else 'New alias'}"),
            ("", "\n\n"),
        ]
        focus_idx = form.get("focus_idx", 0)
        rows_def = (
            ("alias", 0),
            ("model id", 1),
            ("base_url (optional)", 2),
        )
        for label, idx in rows_def:
            focused = idx == focus_idx
            frag.append(("class:checked" if focused else "class:dim", " \u25b8 " if focused else "   "))
            frag.append(("class:table.header" if focused else "class:dim", label + "\n"))
            frag.append(("", "\n"))
        # fallback field rendered via its window below; keep spacing stable here.
        focused_kind = _on_kind_stop()
        frag.append((
            "class:checked" if focused_kind else "class:dim",
            " \u25b8 " if focused_kind else "   ",
        ))
        frag.append(("class:section", "provider kind (\u2190/\u2192 to change)" + "\n"))
        frag.append(("class:value" if focused_kind else "class:dim", f" \u25c6 {kind_label}"))
        frag.append(("class:dim", f"  {kind_desc}\n\n"))
        return frag

    def _get_form_footer() -> list[tuple[str, str]]:
        frag: list[tuple[str, str]] = [(
            "class:footer",
            " tab/\u21f5 move \u00b7 enter save \u00b7 esc cancel ",
        )]
        flash = state["flash"]
        if flash.startswith("!error:"):
            frag.append(("fg:red bold", f"   \u2717 {flash[len('!error:'):]}"))
        elif flash:
            frag.append(("class:checked", f"   \u2713 {flash}"))
        frag.append(("", "\n"))
        return frag

    # -- live test (async, never blocks the UI thread) --------------------------

    def _run_live_test(alias: str, app) -> None:
        loop = asyncio.get_event_loop()

        def blocking() -> tuple[int, str]:
            import io

            buf = io.StringIO()
            captured = Console(file=buf, width=100)
            global_console = globals()["console"]
            globals()["console"] = captured
            try:
                code = route_test(alias)
            finally:
                globals()["console"] = global_console
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            summary = " \u2502 ".join(lines[-3:])[:400]
            return code, ("\u2717 " if code else "\u2713 ") + (summary or "(no output)")

        async def runner() -> None:
            try:
                code, summary = await loop.run_in_executor(None, blocking)
            except Exception as exc:  # secret-free surface: type name only
                summary = f"\u2717 {type(exc).__name__}: {_one_line(exc)}"
            state["testing"] = False
            state["detail"] = summary
            app.invalidate()

        state["testing"] = True
        state["detail"] = ""
        asyncio.ensure_future(runner())

    # -- table-mode key bindings -------------------------------------------------

    kb_table = KeyBindings()

    @kb_table.add("up", eager=True)
    @kb_table.add("k", eager=True)
    def _up(event):
        if state["cursor"] > 0:
            state["cursor"] -= 1

    @kb_table.add("down", eager=True)
    @kb_table.add("j", eager=True)
    def _down(event):
        if state["cursor"] < len(_rows()) - 1:
            state["cursor"] += 1

    @kb_table.add("pageup", eager=True)
    def _pageup(event):
        state["cursor"] = max(0, state["cursor"] - 10)

    @kb_table.add("pagedown", eager=True)
    def _pagedown(event):
        state["cursor"] = min(len(_rows()) - 1, state["cursor"] + 10)

    @kb_table.add("/", eager=True)
    def _start_search(event):
        state["search"] = ""

    @kb_table.add("backspace", eager=True)
    def _search_backspace(event):
        if state["search"]:
            state["search"] = state["search"][:-1]

    @kb_table.add("escape", eager=True)
    def _esc(event):
        if state["confirm_delete"]:
            state["confirm_delete"] = False
        elif state["search"]:
            state["search"] = ""

    @kb_table.add("a", eager=True)
    def _add(event):
        _enter_form(event.app, edit=False)

    @kb_table.add("e", eager=True)
    def _edit(event):
        _enter_form(event.app, edit=True)

    @kb_table.add("d", eager=True)
    def _delete(event):
        if _selected_row() is not None:
            state["confirm_delete"] = True
            event.app.invalidate()

    @kb_table.add("y", eager=True)
    def _confirm_yes(event):
        if not state["confirm_delete"]:
            return
        state["confirm_delete"] = False
        row = _selected_row()
        if row is None:
            return
        try:
            backup = route_alias_remove(row["alias"], path=path)
        except ValueError as exc:
            _flash(str(exc), error=True)
            event.app.invalidate()
            return
        _reload()
        note = f"  [backup: {backup.name}]" if backup else ""
        _flash(f"Removed {row['alias']!r}{note}")
        event.app.invalidate()

    @kb_table.add("n", eager=True)
    def _confirm_no(event):
        state["confirm_delete"] = False
        event.app.invalidate()

    @kb_table.add("t", eager=True)
    def _test(event):
        row = _selected_row()
        if row is None:
            return
        state["detail"] = ""
        _run_live_test(row["alias"], event.app)
        event.app.invalidate()

    @kb_table.add("r", eager=True)
    def _refresh(event):
        _reload()
        state["cursor"] = 0
        _flash(f"Reloaded {path}")
        event.app.invalidate()

    @kb_table.add("q", eager=True)
    @kb_table.add("c-c", eager=True)
    def _quit(event):
        event.app.exit(result=None)

    @kb_table.add("<any>", eager=True)
    def _any_char(event):
        ch = event.data
        if state["confirm_delete"]:
            return  # only y/n/esc handled above
        if ch.isprintable() and len(ch) == 1:
            state["search"] += ch
            state["cursor"] = 0

    # -- form-mode key bindings ----------------------------------------------------

    kb_form = KeyBindings()

    @kb_form.add("tab")
    def _f_next(event):
        _cycle_focus(event.app, +1)
        event.app.invalidate()

    @kb_form.add("s-tab")
    def _f_prev(event):
        _cycle_focus(event.app, -1)
        event.app.invalidate()

    @kb_form.add("down")
    def _f_down(event):
        idx = form["focus_idx"]
        if idx == 3:  # multiline fallback editor keeps native cursor motion
            form["fallback_field"].buffer.cursor_down()
        else:
            _cycle_focus(event.app, +1)
        event.app.invalidate()

    @kb_form.add("up")
    def _f_up(event):
        idx = form["focus_idx"]
        if idx == 3 and form["fallback_field"].buffer.document.cursor_position_row > 0:
            form["fallback_field"].buffer.cursor_up()
        else:
            _cycle_focus(event.app, -1)
        event.app.invalidate()

    @kb_form.add("left")
    @kb_form.add("right")
    def _f_kind_change(event):
        if _on_kind_stop():
            step = 1 if event.key_sequence[0].key == "right" else -1
            form["kind_idx"] = (form["kind_idx"] + step) % len(PROVIDER_KINDS)
            form["kind_field"].buffer.set_document(
                _PTDocument(PROVIDER_KINDS[form["kind_idx"]][0])
            )
            event.app.invalidate()

    @kb_form.add("c-m", eager=True)
    def _f_submit(event):
        if _on_kind_stop():
            _submit_form(event.app)
        else:
            _cycle_focus(event.app, +1)
            event.app.invalidate()

    @kb_form.add("escape", eager=True)
    def _f_cancel(event):
        _cancel_form(event.app)

    @kb_form.add("c-c", eager=True)
    def _f_quit(event):
        event.app.exit(result=None)

    # -- layout ----------------------------------------------------------------------

    header_window = Window(
        content=FormattedTextControl(_get_header), height=1, always_hide_cursor=True
    )
    table_control = FormattedTextControl(_get_table)
    table_control.get_cursor_position = lambda: __import__(
        "prompt_toolkit.data_structures", fromlist=["Point"]
    ).Point(x=0, y=max(0, state["cursor"]))
    table_window = Window(
        content=table_control,
        always_hide_cursor=True,
        scroll_offsets=ScrollOffsets(top=1, bottom=1),
    )
    detail_window = Window(
        content=FormattedTextControl(_get_detail),
        height=12,
        always_hide_cursor=True,
    )
    footer_window = Window(
        content=FormattedTextControl(_get_footer), height=1, always_hide_cursor=True
    )

    table_view = HSplit(
        [
            header_window,
            table_window,
            detail_window,
            footer_window,
        ]
    )

    def _make_form_view():
        """Build the form layout from the currently-installed widgets."""
        if "alias_field" not in form:
            return Window(content=FormattedTextControl(lambda: [("", "")]), height=1)
        return HSplit(
            [
                VSplit(
                    [
                        Window(width=3, always_hide_cursor=True),
                        HSplit(
                            [
                                form["alias_field"],
                                form["model_field"],
                                form["url_field"],
                            ]
                        ),
                    ]
                ),
                Window(height=1, char=" ", style=""),
                form["fallback_field"],
                Window(height=1, char=" ", style=""),
                Window(
                    content=FormattedTextControl(_get_form),
                    always_hide_cursor=True,
                ),
                Window(height=1, char=" ", style=""),
                Window(
                    content=FormattedTextControl(_get_form_footer),
                    height=1,
                    always_hide_cursor=True,
                ),
            ]
        )

    form_container = DynamicContainer(lambda: _make_form_view())

    layout = Layout(
        HSplit(
            [
                ConditionalContainer(table_view, filter=Condition(lambda: state["mode"] == "table")),
                ConditionalContainer(form_container, filter=Condition(lambda: state["mode"] == "form")),
            ]
        )
    )

    from prompt_toolkit.key_binding import (
        ConditionalKeyBindings,
        KeyBindings,
    )

    merged_kb = KeyBindings()
    merged_kb.bindings.extend(
        ConditionalKeyBindings(
            kb_table, Condition(lambda: state["mode"] == "table")
        ).bindings
    )
    merged_kb.bindings.extend(
        ConditionalKeyBindings(
            kb_form, Condition(lambda: state["mode"] == "form")
        ).bindings
    )

    app: Application = Application(
        layout=layout,
        key_bindings=merged_kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )

    app.run()


# ---------------------------------------------------------------------------
# Non-interactive action picker (kept for graceful degradation paths)
# ---------------------------------------------------------------------------


def _select_action() -> str:
    actions = ["list", "set", "add", "remove", "rename", "test", "quit"]
    try:
        import questionary

        answer = questionary.select("Action:", choices=actions).ask()
        if answer is None:
            raise EOFError
        return answer
    except ImportError:
        from rich.prompt import Prompt

        return Prompt.ask("Action", choices=actions, default="list")
