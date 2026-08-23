"""Interactive TUI for route/alias management (ADR 0005 route policies).

All mutations go through ``~/.ppmlx/routes.toml`` (or ``PPMLX_ROUTE_POLICY`` /
``[server] route_policy`` when configured). Writes are atomic with a
timestamped backup (reusing :mod:`ppmlx.onboard` helpers) and every document
is validated through :func:`ppmlx.router.policy_from_dict` before it touches
disk. API keys are never read or displayed here.
"""
from __future__ import annotations

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
# Interactive pieces
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
# Interactive loop
# ---------------------------------------------------------------------------


def run_route_tui(*, path: Path | None = None) -> None:
    """Interactive loop: list / set / add / remove / rename / test / quit."""
    path = path or route_policy_path()
    console.print(Panel("[bold]ppmlx route[/bold] — alias manager"))
    while True:
        try:
            action = _select_action()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            return
        if action == "quit":
            console.print("[dim]bye[/dim]")
            return
        try:
            if action == "list":
                rows = route_list(path=path)
                if rows:
                    console.print(render_alias_table(rows))
                else:
                    console.print(
                        f"[yellow]No aliases yet at {path}.[/yellow] Choose 'set' to add one."
                    )
            elif action == "set":
                route_set("", path=path)
            elif action == "add":
                from rich.prompt import Confirm, Prompt

                alias = Prompt.ask("New alias name").strip()
                provider, _ = _ask_provider_kind()
                model = Prompt.ask("Model id").strip()
                route_alias_add(alias, provider, model, path=path)
                console.print(f"[green]Added {alias}.[/green]")
            elif action == "remove":
                from rich.prompt import Confirm, Prompt

                rows = route_list(path=path)
                console.print(render_alias_table(rows))
                alias = Prompt.ask("Alias to remove").strip()
                if Confirm.ask(f"Really remove {alias!r}?", default=False):
                    route_alias_remove(alias, path=path)
                    console.print(f"[green]Removed {alias}.[/green]")
            elif action == "rename":
                from rich.prompt import Prompt

                old = Prompt.ask("Current alias").strip()
                new = Prompt.ask("New name").strip()
                route_alias_rename(old, new, path=path)
                console.print(f"[green]Renamed {old} -> {new}.[/green]")
            elif action == "test":
                from rich.prompt import Prompt

                rows = route_list(path=path)
                console.print(render_alias_table(rows))
                alias = Prompt.ask("Alias to test").strip()
                route_test(alias)
        except ValueError as error:
            console.print(f"[red]{error}[/red]")
        except SystemExit:
            pass
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")


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
