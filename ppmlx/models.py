from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Any


# ── Built-in alias map (exact HuggingFace repo IDs from spec) ──────────

DEFAULT_ALIASES: dict[str, str] = {
    # Qwen 3.5 — released Feb 2026
    "qwen3.5:0.8b":       "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
    "qwen3.5:2b":         "mlx-community/Qwen3.5-2B-MLX-4bit",
    "qwen3.5:4b":         "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen3.5:9b":         "mlx-community/Qwen3.5-9B-MLX-4bit",
    "qwen3.5:27b":        "mlx-community/Qwen3.5-27B-4bit",
    "qwen3.5:35b-a3b":    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "qwen3.5:122b-a10b":  "mlx-community/Qwen3.5-122B-A10B-4bit",
    # GLM-4 — THUDM / mlx-community (matches Ollama naming)
    "glm-4.7-flash":      "mlx-community/GLM-4.7-Flash-4bit",
    # GPT-OSS (OpenAI open weights) — released Aug 2025
    "gpt-oss:20b":        "mlx-community/gpt-oss-20b-4bit",
    "gpt-oss:120b":       "mlx-community/gpt-oss-120b-4bit",
}

# Patterns for routing
_VISION_INDICATORS = ["-VL-", "-vlm"]
_TEXT_ONLY_INDICATORS = ["-text-", "-Text-", "OptiQ"]
_EMBED_PREFIXES = ("embed:",)


class ModelNotFoundError(Exception):
    pass


def _get_ppmlx_dir() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        return get_ppmlx_dir()
    except ImportError:
        return Path.home() / ".ppmlx"


def _get_models_dir() -> Path:
    d = _get_ppmlx_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_aliases_file() -> Path:
    return _get_ppmlx_dir() / "aliases.json"


def load_user_aliases() -> dict[str, str]:
    """Load user-defined aliases from ~/.ppmlx/aliases.json."""
    p = _get_aliases_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_user_alias(name: str, repo_id: str) -> None:
    """Add or update a user alias."""
    aliases = load_user_aliases()
    aliases[name] = repo_id
    p = _get_aliases_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(aliases, indent=2))


def remove_user_alias(name: str) -> bool:
    """Remove a user alias. Returns True if it existed."""
    aliases = load_user_aliases()
    if name in aliases:
        del aliases[name]
        _get_aliases_file().write_text(json.dumps(aliases, indent=2))
        return True
    return False


# ── Favorites ────────────────────────────────────────────────────────────

def _get_favorites_file() -> Path:
    return _get_ppmlx_dir() / "favorites.json"


def load_favorites() -> list[str]:
    """Load the ordered list of favorite model aliases."""
    p = _get_favorites_file()
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_favorites(favs: list[str]) -> None:
    p = _get_favorites_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(favs, indent=2))


def add_favorite(alias_or_repo: str) -> bool:
    """Add a model to favorites. Returns False if already a favorite."""
    favs = load_favorites()
    if alias_or_repo in favs:
        return False
    favs.append(alias_or_repo)
    _save_favorites(favs)
    return True


def remove_favorite(alias_or_repo: str) -> bool:
    """Remove a model from favorites. Returns True if it was present."""
    favs = load_favorites()
    if alias_or_repo in favs:
        favs.remove(alias_or_repo)
        _save_favorites(favs)
        return True
    return False


def _is_registry_enabled() -> bool:
    """Check if the registry is enabled in config."""
    try:
        from ppmlx.config import load_config
        return load_config().registry.enabled
    except Exception:
        return True


def _get_registry_aliases() -> dict[str, str]:
    """Return registry aliases if enabled, else empty dict."""
    if not _is_registry_enabled():
        return {}
    try:
        from ppmlx.registry import registry_aliases
        return registry_aliases()
    except Exception:
        return {}


def all_aliases() -> dict[str, str]:
    """Return merged dict: registry < defaults < user (user wins)."""
    merged = _get_registry_aliases()
    merged.update(DEFAULT_ALIASES)
    merged.update(load_user_aliases())
    return merged


def resolve_alias(name: str) -> str:
    """
    Resolve a model name to a HuggingFace repo ID.

    Priority: direct repo ID > user aliases > DEFAULT_ALIASES > registry > prefix match > error
    """
    # Strip provider prefix so that clients like pi that send
    # provider-qualified model names still resolve.
    # Supports both "ppmlx:model" and "ppmlx/model" formats.
    if name.startswith("ppmlx:"):
        name = name[len("ppmlx:"):]
    elif name.startswith("ppmlx/"):
        name = name[len("ppmlx/"):]

    if "/" in name:
        return name

    user = load_user_aliases()
    if name in user:
        return user[name]
    if name in DEFAULT_ALIASES:
        return DEFAULT_ALIASES[name]

    # Check registry
    reg = _get_registry_aliases()
    if name in reg:
        return reg[name]

    # Prefix match across all alias sources
    all_a = {**reg, **DEFAULT_ALIASES, **user}
    matches = [(k, v) for k, v in all_a.items() if k.startswith(name + ":") or k == name]
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    available = sorted(all_a.keys())
    raise ModelNotFoundError(
        f"Unknown model: '{name}'\n"
        f"Available aliases: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}\n"
        f"Or use a HuggingFace repo ID directly (e.g. 'mlx-community/model-name')."
    )


def is_vision_model(repo_id: str) -> bool:
    """Detect if a model should be loaded with mlx-vlm."""
    if any(ind in repo_id for ind in _TEXT_ONLY_INDICATORS):
        return False
    return any(ind in repo_id for ind in _VISION_INDICATORS)


def is_embed_model(alias_or_repo: str) -> bool:
    """Detect if a model is an embedding model."""
    if alias_or_repo.startswith(_EMBED_PREFIXES):
        return True
    lower = alias_or_repo.lower()
    return any(p in lower for p in ["embed", "embedding", "minilm", "bge-", "nomic-"])


def repo_to_local_name(repo_id: str) -> str:
    """Convert 'org/repo' -> 'org--repo' for local directory name."""
    return repo_id.replace("/", "--")


_DOWNLOAD_IGNORE_PATTERNS = ["*.md", "*.txt", "original/*"]


def _get_repo_size(repo_id: str, token: str | None = None) -> int | None:
    """Return total download size in bytes, excluding ignored patterns."""
    try:
        import fnmatch
        from huggingface_hub import list_repo_tree
        total = 0
        for f in list_repo_tree(repo_id, token=token, recursive=True):
            path = getattr(f, "rfilename", "") or ""
            size = getattr(f, "size", 0) or 0
            if any(fnmatch.fnmatch(path, pat) for pat in _DOWNLOAD_IGNORE_PATTERNS):
                continue
            total += size
        return total or None
    except Exception:
        return None


def _get_hf_token(explicit: str | None = None) -> str | None:
    """Return HF token: explicit arg > config.toml > HF_TOKEN env var."""
    if explicit:
        return explicit
    try:
        import tomllib
        cfg_path = _get_ppmlx_dir() / "config.toml"
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                data = tomllib.load(f)
            tok = data.get("auth", {}).get("hf_token")
            if tok:
                return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or None


def _tree_size(path: Path) -> int:
    """Total bytes of all files under *path* (recursive, follows symlinks)."""
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def download_model(alias_or_repo: str, token: str | None = None) -> Path:
    """
    Download a model from HuggingFace Hub with a single Rich progress bar.

    HuggingFace already emits byte-level progress through ``tqdm_class``.
    We adapt those callbacks into Rich instead of polling filesystem sizes.
    Xet is disabled by default because its progress callbacks can arrive in
    large bursts, making terminal progress appear frozen and then jump.
    """
    import threading
    from rich.progress import (
        Progress, BarColumn, DownloadColumn, TransferSpeedColumn,
        TimeRemainingColumn, TextColumn, SpinnerColumn,
    )

    # Prefer smoother, HTTP-backed progress by default. Users can still opt
    # into Xet explicitly with HF_HUB_DISABLE_XET=0.
    if "HF_HUB_DISABLE_XET" not in os.environ:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            from huggingface_hub import constants as hf_constants

            hf_constants.HF_HUB_DISABLE_XET = True
        except Exception:
            pass

    from huggingface_hub import snapshot_download

    token = _get_hf_token(token)
    repo_id = resolve_alias(alias_or_repo)
    local_name = repo_to_local_name(repo_id)
    local_path = _get_models_dir() / local_name

    if local_path.exists() and any(local_path.iterdir()):
        return local_path

    local_path.mkdir(parents=True, exist_ok=True)
    expected_total = _get_repo_size(repo_id, token)

    # ── colours ──────────────────────────────────────────────────────
    BLUE, GREEN, ORANGE, RED, WHITE = "blue", "green", "#d78700", "red", "white"

    bar = BarColumn(bar_width=None, complete_style=BLUE, finished_style=GREEN)
    progress_lock = threading.RLock()
    progress_state = {
        "completed": 0.0,
        "total": float(expected_total or 0),
    }

    def _display_total(completed: float, total: float) -> float | None:
        current = max(total, float(expected_total or 0), completed)
        return current or None

    with Progress(
        SpinnerColumn(style=WHITE),
        TextColumn("[bold blue]{task.description}"),
        bar,
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=10,
        auto_refresh=False,
    ) as progress:
        task = progress.add_task(f"↓ {alias_or_repo}", total=expected_total)

        class _RichTqdm:
            """Small tqdm-compatible adapter used by HuggingFace Hub.

            HF invokes tqdm callbacks from worker/Xet threads. Keep those
            callbacks cheap and thread-safe; the main thread renders Rich at a
            fixed cadence below so terminal refresh never depends on callback
            timing or on Rich being called from background threads.
            """

            _lock = threading.RLock()

            @classmethod
            def get_lock(cls):
                return cls._lock

            @classmethod
            def set_lock(cls, lock) -> None:
                cls._lock = lock

            def __init__(
                self,
                iterable=None,
                desc: str | None = None,
                total: int | float | None = None,
                initial: int | float = 0,
                unit: str = "it",
                unit_scale: bool = False,
                disable: bool = False,
                name: str | None = None,
                **_: object,
            ) -> None:
                self.iterable = iterable
                self.desc = desc or ""
                self.n = float(initial or 0)
                self.unit = unit
                self.unit_scale = unit_scale
                self.disable = disable
                self.name = name
                self._is_bytes = (
                    unit == "B"
                    or name == "huggingface_hub.snapshot_download"
                    or self.desc.startswith("Downloading")
                )
                self._total = float(total or 0)
                self._sync_state()

            @property
            def total(self) -> float:
                return self._total

            @total.setter
            def total(self, value: int | float | None) -> None:
                self._total = float(value or 0)
                self._sync_state()

            def _sync_state(self) -> None:
                if not self._is_bytes:
                    return
                with progress_lock:
                    progress_state["completed"] = max(progress_state["completed"], self.n)
                    progress_state["total"] = max(progress_state["total"], self._total)

            def __iter__(self):
                if self.iterable is None:
                    return
                for item in self.iterable:
                    yield item
                    self.update(1)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                self.close()

            def update(self, n: int | float | None = 1) -> None:
                if n is None:
                    n = 1
                with type(self)._lock:
                    self.n += float(n)
                    self._sync_state()

            def refresh(self, *args: object, **kwargs: object) -> None:
                self._sync_state()

            def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
                self.desc = desc or ""
                if refresh:
                    self.refresh()

            def close(self) -> None:
                self._sync_state()

            def clear(self, *args: object, **kwargs: object) -> None:
                pass

            def display(self, *args: object, **kwargs: object) -> None:
                pass

        result: dict[str, object] = {}

        def _bg_download() -> None:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_path),
                    token=token,
                    ignore_patterns=_DOWNLOAD_IGNORE_PATTERNS,
                    tqdm_class=_RichTqdm,
                )
            except BaseException as exc:
                result["error"] = exc

        dl_thread = threading.Thread(target=_bg_download, daemon=True)
        dl_thread.start()

        try:
            while dl_thread.is_alive():
                with progress_lock:
                    completed = progress_state["completed"]
                    total = progress_state["total"]
                progress.update(
                    task,
                    completed=completed,
                    total=_display_total(completed, total),
                )
                progress.refresh()
                dl_thread.join(timeout=0.1)
        except KeyboardInterrupt:
            bar.complete_style = ORANGE  # type: ignore[assignment]
            bar.finished_style = ORANGE  # type: ignore[assignment]
            progress.update(task, description=f"[bold {ORANGE}]✗ {alias_or_repo}")
            progress.refresh()
            progress.stop()
            shutil.rmtree(local_path, ignore_errors=True)
            raise

        with progress_lock:
            completed = progress_state["completed"]
            total = progress_state["total"]
        progress.update(
            task,
            completed=completed,
            total=_display_total(completed, total),
        )
        progress.refresh()

        exc = result.get("error")
        if exc:
            bar.complete_style = RED  # type: ignore[assignment]
            bar.finished_style = RED  # type: ignore[assignment]
            progress.update(task, description=f"[bold {RED}]✗ {alias_or_repo}")
            progress.refresh()
            shutil.rmtree(local_path, ignore_errors=True)
            if isinstance(exc, KeyboardInterrupt):
                raise exc
            raise ModelNotFoundError(
                f"Failed to download '{repo_id}': {exc}"
            ) from exc

        # success
        final_total = expected_total or _tree_size(local_path)
        bar.complete_style = GREEN  # type: ignore[assignment]
        progress.update(
            task,
            total=final_total,
            completed=final_total,
            description=f"[bold {GREEN}]✓ {alias_or_repo}",
        )
        progress.refresh()

    return local_path


def resolve_model_path(repo_id: str) -> str:
    """Resolve a repo_id to a local path if available, otherwise return the
    repo_id for direct HuggingFace loading."""
    local = get_model_path(repo_id)
    if local:
        return str(local)
    return repo_id


def get_model_path(alias_or_repo: str) -> Path | None:
    """Return local path if model exists, else None."""
    try:
        repo_id = resolve_alias(alias_or_repo)
    except ModelNotFoundError:
        repo_id = alias_or_repo

    local_name = repo_to_local_name(repo_id)
    p = _get_models_dir() / local_name
    if p.exists() and any(p.iterdir()):
        return p
    return None


def list_local_models() -> list[dict[str, Any]]:
    """List all locally downloaded models."""
    models_dir = _get_models_dir()
    result = []
    if not models_dir.exists():
        return result

    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024 ** 3)
        repo_id = d.name.replace("--", "/", 1)
        aliases_for_model = [k for k, v in all_aliases().items() if v == repo_id]
        result.append({
            "name": d.name,
            "repo_id": repo_id,
            "alias": aliases_for_model[0] if aliases_for_model else repo_id,
            "size_gb": round(size_gb, 2),
            "path": d,
        })
    return result


def remove_model(alias_or_repo: str) -> bool:
    """Remove a locally downloaded model. Returns True if removed."""
    path = get_model_path(alias_or_repo)
    if path is None:
        return False
    shutil.rmtree(path)
    return True
