"""Onboarding wizard: persistent harness configs for Claude Code, Codex, OpenCode, pi.

Every writer function is idempotent (re-running produces no changes), creates a
timestamped backup before modifying an existing file, and supports ``dry_run``
(compute the change, touch nothing). All paths are derived from ``Path.home()``
unless an explicit ``home`` override is passed, so tests and ``HOME``-based
isolation work naturally.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:6767"
HARNESSES = ("claude", "codex", "opencode", "pi")

# Binary names to look for, plus config dirs that indicate an install.
_HARNESS_BINARIES = {
    "claude": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "pi": "pi",
}
_HARNESS_CONFIG_DIRS = {
    "claude": (".claude",),
    "codex": (".codex",),
    "opencode": (".config/opencode", ".opencode"),
    "pi": (".pi",),
}


@dataclass
class ChangeRecord:
    """One file-level change performed (or proposed) by the wizard."""

    harness: str
    path: Path
    changed: bool  # False => already up to date (idempotent no-op)
    backup: Path | None = None
    dry_run: bool = False
    detail: str = ""

    def describe(self) -> str:
        status = "DRY-RUN" if self.dry_run else ("updated" if self.changed else "already up to date")
        line = f"[{self.harness}] {self.path}: {status}"
        if self.backup is not None:
            line += f" (backup: {self.backup})"
        if self.detail:
            line += f" — {self.detail}"
        return line


@dataclass
class OnboardResult:
    changes: list[ChangeRecord] = field(default_factory=list)

    @property
    def applied(self) -> list[ChangeRecord]:
        return [c for c in self.changes if c.changed and not c.dry_run]

    def summary(self) -> str:
        lines = [c.describe() for c in self.changes]
        backups = sorted({str(c.backup) for c in self.changes if c.backup})
        if backups:
            lines.append("Backups created (delete these to fully revert):")
            lines.extend(f"  - {b}" for b in backups)
        if not any(c.changed or c.dry_run for c in self.changes):
            lines.append("No changes were needed.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_harnesses(home: Path | None = None) -> dict[str, bool]:
    """Return {harness: installed} based on binaries on PATH or config dirs."""
    home = home or Path.home()
    found: dict[str, bool] = {}
    for name in HARNESSES:
        binary = shutil.which(_HARNESS_BINARIES[name]) is not None
        config = any((home / d).exists() for d in _HARNESS_CONFIG_DIRS[name])
        found[name] = binary or config
    return found


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _backup(path: Path, now_text: str | None = None) -> Path:
    stamp = now_text or time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    return backup


def _write_changed(
    path: Path,
    new_text: str,
    *,
    dry_run: bool,
    harness: str,
    detail: str,
    result: OnboardResult,
) -> None:
    """Write new_text unless identical to what's on disk; back up first."""
    exists = path.exists()
    old_text = path.read_text() if exists else None
    changed = not exists or old_text != new_text
    backup: Path | None = None
    if changed and not dry_run and exists:
        backup = _backup(path)
    if changed and not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text)
    result.changes.append(
        ChangeRecord(
            harness=harness,
            path=path,
            changed=changed,
            backup=backup,
            dry_run=dry_run,
            detail=detail,
        )
    )


def _merge_json_dict(existing_text: str | None, patch: dict) -> dict:
    """Shallow-plus-one deep merge of patch into parsed JSON (preserves keys)."""
    data: dict = {}
    if existing_text and existing_text.strip():
        loaded = json.loads(existing_text)
        if isinstance(loaded, dict):
            data = loaded
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return data


def _upsert_toml_section(text: str, section: str, updates: dict[str, object]) -> str:
    """Replace (or append) a single ``[section]`` block with the given keys.

    Preserves everything outside the section, including comments.
    """
    header_re = re.compile(rf"^\[{re.escape(section)}\]\s*$", re.MULTILINE)
    text = text.rstrip("\n")
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if header_re.match(ln)), None)
    block = "".join(f"{k} = {_toml_value(v)}\n" for k, v in updates.items())
    new_block = f"[{section}]\n{block}"
    if start is None:
        sep = "\n\n" if text.strip() else ""
        return f"{text}{sep}{new_block}"
    # Find end of section (next top-level table header).
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^\[", lines[i]):
            end = i
            break
    # Preserve sibling keys/comments in the section that we're not updating.
    updated_keys = set(updates)
    kept = [
        ln
        for ln in lines[start + 1 : end]
        if not any(re.match(rf"^\s*{re.escape(k)}\s*=", ln) for k in updated_keys)
    ]
    body = "".join(f"{k} = {_toml_value(v)}\n" for k, v in updates.items())
    replacement = f"[{section}]\n{body}" + "".join(kept) + ("\n" if end < len(lines) else "")
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def _set_toml_top_level_key(text: str, key: str, value: object) -> str:
    """Set a top-level scalar TOML key (before the first table header)."""
    assignment = f"{key} = {_toml_value(value)}"
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    first_table = re.search(r"^\[", text, re.MULTILINE)
    if pattern.search(text) and (
        not first_table or pattern.search(text[: first_table.start()])
    ):
        return pattern.sub(lambda _: assignment, text, count=1)
    # Key missing (or only inside a table): insert above the first table header.
    if first_table:
        idx = first_table.start()
        prefix, rest = text[:idx], text[idx:]
        if prefix.strip():
            return f"{prefix.rstrip(chr(10))}\n{assignment}\n\n{rest}"
        return f"{assignment}\n\n{rest}"
    body = text.rstrip("\n")
    return f"{body}\n{assignment}\n" if body else f"{assignment}\n"


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Harness writers
# ---------------------------------------------------------------------------


def claude_config_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".claude" / "settings.json"


def write_claude_config(
    base_url: str = DEFAULT_BASE_URL,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    result: OnboardResult | None = None,
) -> ChangeRecord:
    """Point Claude Code at ppmlx via ~/.claude/settings.json env block."""
    result = result or OnboardResult()
    path = claude_config_path(home)
    existing = path.read_text() if path.exists() else None
    merged = _merge_json_dict(existing, {"env": {"ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_API_KEY": "local"}})
    _write_changed(
        path,
        json.dumps(merged, indent=2) + "\n",
        dry_run=dry_run,
        harness="claude",
        detail=f"env.ANTHROPIC_BASE_URL={base_url}",
        result=result,
    )
    return result.changes[-1]


def codex_config_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".codex" / "config.toml"


def write_codex_config(
    base_url: str = DEFAULT_BASE_URL,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    result: OnboardResult | None = None,
) -> ChangeRecord:
    """Add a persistent ppmlx model_provider entry to ~/.codex/config.toml."""
    result = result or OnboardResult()
    path = codex_config_path(home)
    text = path.read_text() if path.exists() else ""
    updated = _upsert_toml_section(
        text,
        "model_providers.ppmlx",
        {
            "name": "ppmlx",
            "base_url": base_url,
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
        },
    )
    updated = _set_toml_top_level_key(updated, "model_provider", "ppmlx")
    _write_changed(
        path,
        updated,
        dry_run=dry_run,
        harness="codex",
        detail=f"model_provider=ppmlx, base_url={base_url}",
        result=result,
    )
    return result.changes[-1]


def opencode_config_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "opencode" / "opencode.json"


def write_opencode_config(
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    result: OnboardResult | None = None,
) -> ChangeRecord:
    """Add a persistent ppmlx provider entry to OpenCode's opencode.json."""
    result = result or OnboardResult()
    path = opencode_config_path(home)
    existing = path.read_text() if path.exists() else None
    provider: dict = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "ppmlx (local MLX)",
        "options": {"baseURL": base_url, "apiKey": "local"},
        "models": {},
    }
    if model:
        provider["models"] = {model: {}}
    merged = _merge_json_dict(existing, {"$schema": "https://opencode.ai/config.json", "provider": {"ppmlx": provider}})
    _write_changed(
        path,
        json.dumps(merged, indent=2) + "\n",
        dry_run=dry_run,
        harness="opencode",
        detail=f"provider.ppmlx baseURL={base_url}",
        result=result,
    )
    return result.changes[-1]


def pi_models_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".pi" / "agent" / "models.json"


def write_pi_models(
    base_url: str,
    model: str,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    result: OnboardResult | None = None,
) -> ChangeRecord:
    """Persist a ppmlx provider into ~/.pi/agent/models.json (same as launch pi)."""
    result = result or OnboardResult()
    path = pi_models_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else None
    entry = {
        "api": "openai-completions",
        "apiKey": "local",
        "baseUrl": base_url,
        "models": [
            {
                "_launch": True,
                "contextWindow": 262144,
                "id": model,
                "input": ["text"],
                "reasoning": True,
            }
        ],
    }
    data: dict = {}
    if existing and existing.strip():
        try:
            loaded = json.loads(existing)
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}
    if isinstance(data.get("providers"), dict):
        providers = dict(data["providers"])
        providers["ppmlx"] = entry
        data["providers"] = providers
    else:
        data = {"providers": {"ppmlx": entry}}
    _write_changed(
        path,
        json.dumps(data, indent=2) + "\n",
        dry_run=dry_run,
        harness="pi",
        detail=f"providers.ppmlx baseUrl={base_url} model={model}",
        result=result,
    )
    return result.changes[-1]


# ---------------------------------------------------------------------------
# Route policy setup (opt-in)
# ---------------------------------------------------------------------------

EXAMPLE_ROUTES_TOML = """\
# Example ppmlx route policy (ADR 0005). Edit aliases/entries for your models.
[routes]
version = "1"
default_model = "local/default"

[routes.aliases]
# alias-name = ["provider-id", "model-id"]

[[routes.entries]]
key = "claude:claude-sonnet-4-5"
candidates = [{ provider = "local", model = "default" }]
"""


def routes_toml_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".ppmlx" / "routes.toml"


def write_route_policy(
    *,
    home: Path | None = None,
    dry_run: bool = False,
    result: OnboardResult | None = None,
) -> list[ChangeRecord]:
    """Write an example ~/.ppmlx/routes.toml (never overwrites) and wire it into
    [server] route_policy in ~/.ppmlx/config.toml."""
    result = result or OnboardResult()
    home = home or Path.home()
    routes_path = routes_toml_path(home)

    before = len(result.changes)
    if routes_path.exists():
        result.changes.append(
            ChangeRecord(
                harness="route-policy",
                path=routes_path,
                changed=False,
                dry_run=dry_run,
                detail="example policy skipped — file already exists",
            )
        )
    else:
        _write_changed(
            routes_path,
            EXAMPLE_ROUTES_TOML,
            dry_run=dry_run,
            harness="route-policy",
            detail="example route policy",
            result=result,
        )

    config_path = home / ".ppmlx" / "config.toml"
    text = config_path.read_text() if config_path.exists() else ""
    target = str(routes_path)
    updated = _upsert_toml_section(text, "server", {"route_policy": target})
    current = _load_toml(config_path).get("server", {}).get("route_policy", "")
    if current == target and updated == text:
        result.changes.append(  # idempotent no-op
            ChangeRecord(
                harness="route-policy",
                path=config_path,
                changed=False,
                dry_run=dry_run,
                detail="[server] route_policy already set",
            )
        )
    else:
        _write_changed(
            config_path,
            updated,
            dry_run=dry_run,
            harness="route-policy",
            detail="[server] route_policy set",
            result=result,
        )
    return result.changes[before:]


# ---------------------------------------------------------------------------
# Wizard orchestration
# ---------------------------------------------------------------------------


def _offer_route_tui(*, home: Path | None = None) -> None:
    """After route-policy opt-in, offer to launch the interactive alias TUI."""
    from rich.prompt import Confirm

    from ppmlx.cli import console

    if not Confirm.ask(
        "Open the interactive route TUI to define starter aliases?", default=False
    ):
        console.print("[dim]You can run `ppmlx route` anytime.[/dim]")
        return
    from ppmlx.route_tui import run_route_tui, route_policy_path

    run_route_tui(path=route_policy_path(home=home))


def configure_harnesses(
    harnesses: list[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> OnboardResult:
    """Run the persistent-config writers for the selected harnesses."""
    result = OnboardResult()
    unknown = [h for h in harnesses if h not in HARNESSES]
    if unknown:
        raise ValueError(f"Unknown harness(es): {', '.join(unknown)}. Valid: {', '.join(HARNESSES)}")
    for name in harnesses:
        if name == "claude":
            write_claude_config(base_url, home=home, dry_run=dry_run, result=result)
        elif name == "codex":
            write_codex_config(base_url, home=home, dry_run=dry_run, result=result)
        elif name == "opencode":
            write_opencode_config(base_url, model, home=home, dry_run=dry_run, result=result)
        elif name == "pi":
            if model:
                write_pi_models(base_url, model, home=home, dry_run=dry_run, result=result)
            else:
                # pi persistence is owned by `ppmlx launch pi`; without a model
                # there is nothing new to persist.
                result.changes.append(
                    ChangeRecord(
                        harness="pi",
                        path=pi_models_path(home),
                        changed=False,
                        dry_run=dry_run,
                        detail="already persisted by `ppmlx launch pi` — skipping (pass --model to rewrite)",
                    )
                )
    return result


def run_onboard_wizard(
    *,
    harnesses: list[str] | None = None,
    model: str | None = None,
    base_url: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    setup_routes: bool | None = None,
    home: Path | None = None,
) -> OnboardResult:
    """Interactive onboarding wizard. Non-interactive when ``harnesses`` is given.

    Flow: detect → select → write persistent configs → optional route policy →
    summary with undo instructions.
    """
    from rich.prompt import Confirm, Prompt

    from ppmlx.cli import console

    home = home or Path.home()
    base_url = base_url or DEFAULT_BASE_URL
    detected = detect_harnesses(home)

    console.print("[bold]ppmlx onboard[/bold] — persistent coding-harness configuration")
    console.print(f"Server endpoint: {base_url}")
    detected_names = [n for n, ok in detected.items() if ok]
    console.print(f"Detected harnesses: {', '.join(detected_names) or 'none'}")

    if harnesses is None:
        choices = ", ".join(HARNESSES)
        raw = Prompt.ask(
            "Which harnesses should be configured? (comma-separated; empty = all detected)",
            default=", ".join(detected_names) if detected_names else "",
        )
        picked = [p.strip().lower() for p in raw.split(",") if p.strip()]
        invalid = [p for p in picked if p not in HARNESSES]
        if invalid:
            raise ValueError(f"Unknown harness(es): {', '.join(invalid)}. Valid: {choices}")
        harnesses = picked

    if setup_routes is None and not dry_run:
        setup_routes = bool(
            Confirm.ask("Set up example route policy (~/.ppmlx/routes.toml + [server] route_policy)?", default=False)
        )

    result = configure_harnesses(harnesses or [], base_url=base_url, model=model, home=home, dry_run=dry_run)
    if setup_routes:
        write_route_policy(home=home, dry_run=dry_run, result=result)
        if not dry_run and not yes:
            _offer_route_tui(home=home)

    console.print()
    title = "Proposed changes (dry-run)" if dry_run else "Changes"
    console.print(f"[bold]{title}:[/bold]")
    console.print(result.summary())
    if any(c.changed and not c.dry_run for c in result.changes):
        console.print("[dim]To undo: restore the listed .bak-* files over their originals.[/dim]")
    return result
