"""Component manager for ppmlx optional dependencies.

Optional heavy packages (voice, vision, embeddings, analytics) are kept
out of the base install to keep the footprint small. This module provides
an interactive TUI to install / uninstall them on demand.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field

from rich.console import Console

console = Console()


@dataclass
class Component:
    key: str
    label: str
    description: str
    packages: list[str]
    check_imports: list[str]
    size_hint: str
    requires_brew: list[str] = field(default_factory=list)


COMPONENTS: list[Component] = [
    Component(
        key="vision",
        label="Vision / Multimodal",
        description="Image understanding with vision-language models (VLMs)",
        packages=["mlx-vlm>=0.1.18"],
        check_imports=["mlx_vlm"],
        size_hint="~500 MB",
    ),
    Component(
        key="embeddings",
        label="Embeddings",
        description="Local text embedding models (semantic search, RAG)",
        packages=["mlx-embeddings>=0.0.5"],
        check_imports=["mlx_embeddings"],
        size_hint="~200 MB",
    ),
    Component(
        key="voice",
        label="Voice I/O",
        description="Push-to-talk (Whisper STT) + spoken output (Voxtral TTS)",
        packages=["mlx-whisper>=0.4", "mlx-audio>=0.4", "sounddevice>=0.4", "soundfile>=0.12"],
        check_imports=["mlx_whisper", "mlx_audio", "sounddevice", "soundfile"],
        size_hint="~2 GB",
        requires_brew=["portaudio"],
    ),
    Component(
        key="analytics",
        label="Analytics",
        description="Anonymous usage statistics (opt-in)",
        packages=["posthog>=7.9,<8"],
        check_imports=["posthog"],
        size_hint="~200 KB",
    ),
]

_COMPONENT_MAP = {c.key: c for c in COMPONENTS}


# ── State helpers ──────────────────────────────────────────────────────

def _is_installed(component: Component) -> bool:
    for mod in component.check_imports:
        try:
            importlib.import_module(mod.replace("-", "_"))
        except ImportError:
            return False
    return True


def _brew_installed(formula: str) -> bool:
    try:
        return subprocess.run(
            ["brew", "list", formula], capture_output=True, timeout=5,
        ).returncode == 0
    except Exception:
        return False


def _pip_install(packages: list[str], timeout: int = 300) -> bool:
    """Install pip packages, using uv when available."""
    import os
    import shutil

    venv = str(__import__("pathlib").Path(sys.executable).parent.parent)

    if shutil.which("uv"):
        env = {**os.environ, "VIRTUAL_ENV": venv}
        r = subprocess.run(["uv", "pip", "install"] + packages, env=env, capture_output=True, timeout=timeout)
        if r.returncode == 0:
            return True

    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        capture_output=True, timeout=timeout,
    ).returncode == 0


def _pip_uninstall(package_names: list[str], timeout: int = 300) -> bool:
    """Uninstall pip packages, using uv when available."""
    import os
    import shutil

    venv = str(__import__("pathlib").Path(sys.executable).parent.parent)

    if shutil.which("uv"):
        env = {**os.environ, "VIRTUAL_ENV": venv}
        r = subprocess.run(["uv", "pip", "uninstall"] + package_names, env=env, capture_output=True, timeout=timeout)
        if r.returncode == 0:
            return True

    return subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes"] + package_names,
        capture_output=True, timeout=timeout,
    ).returncode == 0


# ── Install / uninstall ────────────────────────────────────────────────

def _download_voice_models() -> None:
    """Download default STT and TTS models after voice addon install."""
    import io
    import contextlib
    import os
    from ppmlx.config import load_config
    cfg = load_config().voice

    models = [
        ("STT", cfg.stt_model),
        ("TTS", cfg.tts_model),
    ]
    from ppmlx.models import _get_hf_token
    token = _get_hf_token()

    for label, repo_id in models:
        short_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        with console.status(
            f"[cyan]Downloading {label} model ({short_name})…[/cyan]",
            spinner="dots",
        ):
            try:
                from huggingface_hub import snapshot_download
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                with contextlib.redirect_stderr(io.StringIO()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    snapshot_download(repo_id, token=token)
                console.print(f"  [green]✓ {label} model ready[/green]")
            except Exception as e:
                console.print(f"  [yellow]⚠ {label} download failed: {e}[/yellow]")
                console.print("  [dim]Will download on first use.[/dim]")
            finally:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


def _install(comp: Component) -> bool:
    for formula in comp.requires_brew:
        if not _brew_installed(formula):
            if subprocess.run(["brew", "install", formula], capture_output=True, timeout=120).returncode != 0:
                return False
    return _pip_install(comp.packages)


def _uninstall(comp: Component) -> bool:
    names = [p.split(">=")[0].split("==")[0].split("<")[0].strip() for p in comp.packages]
    return _pip_uninstall(names)


def install_component(key: str) -> bool:
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        console.print(f"[red]Unknown component: {key!r}. Available: {', '.join(_COMPONENT_MAP)}[/red]")
        return False
    if _is_installed(comp):
        console.print(f"[green]{comp.label} is already installed.[/green]")
        return True
    console.print(f"Installing [bold]{comp.label}[/bold]…")
    with console.status("[cyan]Downloading and installing packages — this may take a moment…[/cyan]"):
        ok = _install(comp)
    console.print("  [green]✓ Done[/green]" if ok else "  [red]✗ Failed[/red]")
    if not ok:
        console.print(f"  [dim]Try: pip install {' '.join(comp.packages)}[/dim]")
        return False
    # Post-install: download default models for voice
    if key == "voice":
        _download_voice_models()
    return True


def uninstall_component(key: str) -> bool:
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        console.print(f"[red]Unknown component: {key!r}. Available: {', '.join(_COMPONENT_MAP)}[/red]")
        return False
    if not _is_installed(comp):
        console.print(f"[yellow]{comp.label} is not installed.[/yellow]")
        return True
    console.print(f"Removing [bold]{comp.label}[/bold]…")
    with console.status("[cyan]Removing packages…[/cyan]"):
        ok = _uninstall(comp)
    console.print("  [green]✓ Removed[/green]" if ok else "  [red]✗ Failed[/red]")
    return ok


# ── Interactive TUI ────────────────────────────────────────────────────

def _render(components: list[Component], cursor: int, statuses: list[bool]) -> str:
    """Build the full screen string for the addons TUI."""
    lines: list[str] = []
    lines.append("\033[1mppmlx addons\033[0m\n")

    col_label = 24
    col_size  = 10

    # Header
    lines.append(
        f"  {'':2}  \033[2m{'Component':<{col_label}}{'Size':>{col_size}}  Description\033[0m"
    )

    for i, (comp, installed) in enumerate(zip(components, statuses)):
        selected = i == cursor
        prefix = "\033[1;36m›\033[0m " if selected else "  "
        status = "\033[32m✓ installed\033[0m " if installed else "\033[2m○ not installed\033[0m"

        # Highlight selected row
        row_style = "\033[7m" if selected else ""
        reset     = "\033[0m" if selected else ""

        lines.append(
            f"{prefix}{row_style}"
            f"{status:28}{comp.label:<{col_label}}"
            f"{comp.size_hint:>{col_size}}  {comp.description}"
            f"{reset}"
        )

    lines.append("")
    lines.append(
        "\033[2m"
        "  \033[1mi\033[0m\033[2m install   "
        "\033[1mu\033[0m\033[2m uninstall   "
        "\033[1mq\033[0m\033[2m / Esc  quit"
        "\033[0m"
    )
    return "\n".join(lines)


def addons_tui() -> None:
    """Full-screen interactive addon manager."""
    import tty
    import termios

    components = list(COMPONENTS)
    cursor = 0
    statuses = [_is_installed(c) for c in components]

    def _refresh() -> None:
        # Clear screen and redraw
        print("\033[2J\033[H", end="", flush=True)
        print(_render(components, cursor, statuses), flush=True)

    def _read_key() -> str:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {"A": "up", "B": "down"}.get(ch3, "")
            return "esc"
        return ch

    def _do_install() -> None:
        comp = components[cursor]
        if statuses[cursor]:
            print("\033[2J\033[H", end="")
            console.print(f"[yellow]{comp.label} is already installed.[/yellow]")
            input("\nPress Enter to continue…")
            return
        print("\033[2J\033[H", end="")
        console.print(f"Installing [bold]{comp.label}[/bold] ({comp.size_hint})…\n")
        with console.status("[cyan]Downloading and installing packages — this may take a moment…[/cyan]"):
            ok = _install(comp)
        statuses[cursor] = _is_installed(comp)  # re-check
        if ok:
            console.print(f"\n[green]✓ {comp.label} installed.[/green]")
        else:
            console.print("\n[red]✗ Installation failed.[/red]")
            console.print(f"[dim]Try: pip install {' '.join(comp.packages)}[/dim]")
        input("\nPress Enter to continue…")

    def _do_uninstall() -> None:
        comp = components[cursor]
        if not statuses[cursor]:
            print("\033[2J\033[H", end="")
            console.print(f"[yellow]{comp.label} is not installed.[/yellow]")
            input("\nPress Enter to continue…")
            return
        print("\033[2J\033[H", end="")
        console.print(f"Removing [bold]{comp.label}[/bold]…\n")
        with console.status("[cyan]Removing packages…[/cyan]"):
            ok = _uninstall(comp)
        statuses[cursor] = _is_installed(comp)  # re-check
        if ok:
            console.print(f"\n[green]✓ {comp.label} removed.[/green]")
        else:
            console.print("\n[red]✗ Removal failed.[/red]")
        input("\nPress Enter to continue…")

    if not sys.stdin.isatty():
        # Non-interactive — just print status
        _print_status_plain(components, statuses)
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        _refresh()
        while True:
            key = _read_key()
            if key in ("q", "esc", "\x03"):
                break
            elif key == "up":
                cursor = (cursor - 1) % len(components)
                _refresh()
            elif key == "down":
                cursor = (cursor + 1) % len(components)
                _refresh()
            elif key == "i":
                _do_install()
                _refresh()
            elif key == "u":
                _do_uninstall()
                _refresh()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\033[2J\033[H", end="")  # clear on exit


def _print_status_plain(components: list[Component], statuses: list[bool]) -> None:
    for comp, installed in zip(components, statuses):
        mark = "✓" if installed else "✗"
        console.print(f"  {mark}  {comp.label:<26} {comp.size_hint:>10}  {comp.description}")


# ── On-demand prompt ───────────────────────────────────────────────────

def prompt_install_if_missing(key: str, feature_name: str) -> bool:
    """Offer to install a missing component interactively."""
    comp = _COMPONENT_MAP.get(key)
    if not comp or _is_installed(comp):
        return comp is not None and _is_installed(comp)

    import questionary
    console.print(
        f"\n[yellow]{feature_name} requires [bold]{comp.label}[/bold] ({comp.size_hint}).[/yellow]"
    )
    console.print(f"[dim]{comp.description}[/dim]\n")

    if not questionary.confirm(f"Install {comp.label} now?", default=True).ask():
        console.print("[dim]Run: ppmlx addons[/dim]")
        return False

    return install_component(key)
