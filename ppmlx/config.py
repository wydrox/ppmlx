from __future__ import annotations
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ANALYTICS_HOST = "https://eu.i.posthog.com"
DEFAULT_ANALYTICS_PROJECT_API_KEY = "phc_rnwLcbdiern6SwykkoSt9BPLoB8zAjVsae3TmgrXw2kA"
LEGACY_ANALYTICS_HOSTS = {"https://analytics.ppmlx.dev"}


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6767
    cors: bool = True
    cors_origins: list[str] = field(default_factory=list)
    max_loaded_models: int = 2
    max_request_body_mb: int = 10
    max_tokens_cap: int = 32768
    max_tools_tokens: int = 12000


@dataclass
class DefaultsConfig:
    model: str = "qwen3.5:0.8b"
    embed_model: str = "embed:all-minilm"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048


@dataclass
class LoggingConfig:
    enabled: bool = True
    snapshot_interval_seconds: int = 60


@dataclass
class MemoryConfig:
    wired_limit_mb: int = 0
    mode: str = "off"  # off | shadow | compact | inject
    max_candidates_per_event: int = 12
    rolling_tokens: int = 10000
    hot_tail_tokens: int = 6500
    session_context_tokens: int = 2000
    compact_threshold_tokens: int = 12000
    max_context_items: int = 40
    extractor: str = "rule_based"
    extraction_model: str = "gemma-4-e2b"
    extraction_workers: int = 1
    extraction_max_tokens: int = 1200
    extraction_timeout_seconds: float = 45.0


@dataclass
class RegistryConfig:
    enabled: bool = True


@dataclass
class ToolAwarenessConfig:
    mode: str = "no_tools_only"


@dataclass
class ThinkingConfig:
    enabled: bool = True
    default_reasoning_budget: int = 2048
    effort_base: int = 256

    def effort_to_budget(self, effort: str) -> int | None:
        """Map reasoning_effort (low/medium/high) to token budget using effort_base."""
        multipliers = {"low": 1, "medium": 4, "high": 32}
        m = multipliers.get(effort.lower())
        return self.effort_base * m if m is not None else None


@dataclass
class AnalyticsConfig:
    enabled: bool = False
    provider: str = "posthog"
    host: str = DEFAULT_ANALYTICS_HOST
    project_api_key: str = DEFAULT_ANALYTICS_PROJECT_API_KEY
    respect_do_not_track: bool = True


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    tool_awareness: ToolAwarenessConfig = field(default_factory=ToolAwarenessConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)


def get_ppmlx_dir() -> Path:
    """Return ~/.ppmlx, creating it if needed."""
    d = Path.home() / ".ppmlx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_bool(v: str) -> bool:
    return v.lower() not in ("0", "false", "no")


def _normalize_memory_mode(value: Any) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "shadow",
        "true": "shadow",
        "yes": "shadow",
        "on": "shadow",
        "shadow": "shadow",
        "compact": "compact",
        "inject": "inject",
    }
    return aliases.get(raw, "off")


def _normalize_tool_awareness_mode(value: Any) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "all",
        "true": "all",
        "yes": "all",
        "all": "all",
        "no_tools_only": "no_tools_only",
    }
    return aliases.get(raw, "no_tools_only")


def _normalize_analytics_host(value: Any) -> str:
    host = str(value).strip().rstrip("/")
    if host in LEGACY_ANALYTICS_HOSTS:
        return DEFAULT_ANALYTICS_HOST
    return host


def _normalize_analytics_provider(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"", "posthog"}:
        return "posthog"
    return raw


def load_config(cli_overrides: dict[str, Any] | None = None) -> Config:
    """Load config with priority: CLI overrides > env vars > TOML file > defaults."""
    cfg = Config()
    toml_path = Path.home() / ".ppmlx" / "config.toml"
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _apply_toml(cfg, data)
    except Exception:
        pass
    _apply_env(cfg)
    if cli_overrides:
        _apply_cli(cfg, cli_overrides)
    return cfg


def _apply_toml(cfg: Config, data: dict) -> None:
    if "server" in data:
        s = data["server"]
        if "host" in s: cfg.server.host = str(s["host"])
        if "port" in s: cfg.server.port = int(s["port"])
        if "cors" in s: cfg.server.cors = bool(s["cors"])
        if "cors_origins" in s:
            origins = s["cors_origins"]
            if isinstance(origins, list):
                cfg.server.cors_origins = [str(o) for o in origins]
        if "max_loaded_models" in s: cfg.server.max_loaded_models = int(s["max_loaded_models"])
        if "max_request_body_mb" in s: cfg.server.max_request_body_mb = int(s["max_request_body_mb"])
        if "max_tokens_cap" in s: cfg.server.max_tokens_cap = int(s["max_tokens_cap"])
        if "max_tools_tokens" in s: cfg.server.max_tools_tokens = int(s["max_tools_tokens"])
    if "defaults" in data:
        d = data["defaults"]
        if "model" in d: cfg.defaults.model = str(d["model"])
        if "embed_model" in d: cfg.defaults.embed_model = str(d["embed_model"])
        if "temperature" in d: cfg.defaults.temperature = float(d["temperature"])
        if "top_p" in d: cfg.defaults.top_p = float(d["top_p"])
        if "max_tokens" in d: cfg.defaults.max_tokens = int(d["max_tokens"])
    if "logging" in data:
        lg = data["logging"]
        if "enabled" in lg: cfg.logging.enabled = bool(lg["enabled"])
        if "snapshot_interval_seconds" in lg:
            cfg.logging.snapshot_interval_seconds = int(lg["snapshot_interval_seconds"])
    if "memory" in data:
        m = data["memory"]
        if "wired_limit_mb" in m: cfg.memory.wired_limit_mb = int(m["wired_limit_mb"])
        if "mode" in m: cfg.memory.mode = _normalize_memory_mode(m["mode"])
        if "max_candidates_per_event" in m: cfg.memory.max_candidates_per_event = int(m["max_candidates_per_event"])
        if "rolling_tokens" in m: cfg.memory.rolling_tokens = int(m["rolling_tokens"])
        if "hot_tail_tokens" in m: cfg.memory.hot_tail_tokens = int(m["hot_tail_tokens"])
        if "session_context_tokens" in m: cfg.memory.session_context_tokens = int(m["session_context_tokens"])
        if "compact_threshold_tokens" in m: cfg.memory.compact_threshold_tokens = int(m["compact_threshold_tokens"])
        if "max_context_items" in m: cfg.memory.max_context_items = int(m["max_context_items"])
        if "extractor" in m: cfg.memory.extractor = str(m["extractor"])
        if "extraction_model" in m: cfg.memory.extraction_model = str(m["extraction_model"])
        if "extraction_workers" in m: cfg.memory.extraction_workers = int(m["extraction_workers"])
        if "extraction_max_tokens" in m: cfg.memory.extraction_max_tokens = int(m["extraction_max_tokens"])
        if "extraction_timeout_seconds" in m: cfg.memory.extraction_timeout_seconds = float(m["extraction_timeout_seconds"])
    if "registry" in data:
        r = data["registry"]
        if "enabled" in r: cfg.registry.enabled = bool(r["enabled"])
    if "tool_awareness" in data:
        ta = data["tool_awareness"]
        if "mode" in ta:
            cfg.tool_awareness.mode = _normalize_tool_awareness_mode(ta["mode"])
    if "thinking" in data:
        th = data["thinking"]
        if "enabled" in th: cfg.thinking.enabled = bool(th["enabled"])
        if "default_reasoning_budget" in th: cfg.thinking.default_reasoning_budget = int(th["default_reasoning_budget"])
        if "effort_base" in th: cfg.thinking.effort_base = int(th["effort_base"])
    if "analytics" in data:
        an = data["analytics"]
        if "enabled" in an: cfg.analytics.enabled = bool(an["enabled"])
        if "provider" in an: cfg.analytics.provider = _normalize_analytics_provider(an["provider"])
        if "host" in an: cfg.analytics.host = _normalize_analytics_host(an["host"])
        if "project_api_key" in an:
            cfg.analytics.project_api_key = str(an["project_api_key"]).strip()
        elif "website_id" in an:
            cfg.analytics.project_api_key = str(an["website_id"]).strip()
        if "respect_do_not_track" in an:
            cfg.analytics.respect_do_not_track = bool(an["respect_do_not_track"])


def _apply_env(cfg: Config) -> None:
    mapping = {
        "PPMLX_HOST": ("server", "host", str),
        "PPMLX_PORT": ("server", "port", int),
        "PPMLX_CORS": ("server", "cors", _parse_bool),
        "PPMLX_MAX_LOADED_MODELS": ("server", "max_loaded_models", int),
        "PPMLX_MAX_TOOLS_TOKENS": ("server", "max_tools_tokens", int),
        "PPMLX_DEFAULT_MODEL": ("defaults", "model", str),
        "PPMLX_DEFAULT_EMBED_MODEL": ("defaults", "embed_model", str),
        "PPMLX_TEMP": ("defaults", "temperature", float),
        "PPMLX_TOP_P": ("defaults", "top_p", float),
        "PPMLX_MAX_TOKENS": ("defaults", "max_tokens", int),
        "PPMLX_LOG_ENABLED": ("logging", "enabled", _parse_bool),
        "PPMLX_LOG_SNAPSHOT_INTERVAL": ("logging", "snapshot_interval_seconds", int),
        "PPMLX_MEMORY_WIRED_LIMIT": ("memory", "wired_limit_mb", int),
        "PPMLX_MEMORY_MODE": ("memory", "mode", _normalize_memory_mode),
        "PPMLX_MEMORY_MAX_CANDIDATES": ("memory", "max_candidates_per_event", int),
        "PPMLX_MEMORY_ROLLING_TOKENS": ("memory", "rolling_tokens", int),
        "PPMLX_MEMORY_HOT_TAIL_TOKENS": ("memory", "hot_tail_tokens", int),
        "PPMLX_MEMORY_SESSION_CONTEXT_TOKENS": ("memory", "session_context_tokens", int),
        "PPMLX_MEMORY_COMPACT_THRESHOLD_TOKENS": ("memory", "compact_threshold_tokens", int),
        "PPMLX_MEMORY_MAX_CONTEXT_ITEMS": ("memory", "max_context_items", int),
        "PPMLX_MEMORY_EXTRACTOR": ("memory", "extractor", str),
        "PPMLX_MEMORY_EXTRACTION_MODEL": ("memory", "extraction_model", str),
        "PPMLX_MEMORY_EXTRACTION_WORKERS": ("memory", "extraction_workers", int),
        "PPMLX_MEMORY_EXTRACTION_MAX_TOKENS": ("memory", "extraction_max_tokens", int),
        "PPMLX_MEMORY_EXTRACTION_TIMEOUT": ("memory", "extraction_timeout_seconds", float),
        "PPMLX_REGISTRY_ENABLED": ("registry", "enabled", _parse_bool),
        "PPMLX_INJECT_TOOL_AWARENESS": ("tool_awareness", "mode", _normalize_tool_awareness_mode),
        "PPMLX_THINKING_ENABLED": ("thinking", "enabled", _parse_bool),
        "PPMLX_THINKING_BUDGET": ("thinking", "default_reasoning_budget", int),
        "PPMLX_EFFORT_BASE": ("thinking", "effort_base", int),
        "PPMLX_ANALYTICS_ENABLED": ("analytics", "enabled", _parse_bool),
        "PPMLX_ANALYTICS_PROVIDER": ("analytics", "provider", _normalize_analytics_provider),
        "PPMLX_ANALYTICS_HOST": ("analytics", "host", _normalize_analytics_host),
        "PPMLX_ANALYTICS_PROJECT_API_KEY": ("analytics", "project_api_key", str),
        "PPMLX_ANALYTICS_RESPECT_DNT": ("analytics", "respect_do_not_track", _parse_bool),
    }
    for env_key, (section, attr, coerce) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(getattr(cfg, section), attr, coerce(val))
            except (ValueError, AttributeError):
                pass
    legacy_website_id = os.environ.get("PPMLX_ANALYTICS_WEBSITE_ID")
    if legacy_website_id and not os.environ.get("PPMLX_ANALYTICS_PROJECT_API_KEY"):
        cfg.analytics.project_api_key = legacy_website_id.strip()


def _apply_cli(cfg: Config, overrides: dict) -> None:
    for key, val in overrides.items():
        if val is None:
            continue
        if key == "host": cfg.server.host = str(val)
        elif key == "port": cfg.server.port = int(val)
        elif key == "cors": cfg.server.cors = bool(val)
        elif key == "model": cfg.defaults.model = str(val)
        elif key == "temperature": cfg.defaults.temperature = float(val)
        elif key == "max_tokens": cfg.defaults.max_tokens = int(val)


def check_first_run() -> None:
    """Show analytics opt-in prompt on first run."""
    try:
        marker = get_ppmlx_dir() / ".first_run_done"
        if marker.exists():
            return
        if not sys.stdin.isatty():
            marker.touch()
            return

        from rich.console import Console

        console = Console()
        console.print(
            "\n[bold magenta]Welcome to the ppmlx beta[/bold magenta] ✨\n"
            "We are tiny, early, and trying to make local LLMs on Macs genuinely great.\n\n"
            "If you are comfortable with it, please allow [bold]anonymous usage analytics[/bold]. "
            "It helps us see which commands work, which APIs matter, and where beta testers get stuck.\n\n"
            "[dim]We never send prompts, responses, tool arguments, file contents, file paths, "
            "tokens, repo IDs, or model input/output. Only event names, version, OS/arch, "
            "an anonymous install id, and coarse booleans/counters.[/dim]\n"
        )
        enabled = _ask_analytics_opt_in()
        _save_analytics_preference(enabled)
        if enabled:
            console.print("[green]Thank you — this genuinely helps the beta. 🫶[/green]")
        else:
            console.print("[dim]No worries — analytics are off. You can enable them later with `ppmlx config --analytics`.[/dim]")
        marker.touch()
    except Exception:
        pass


def _ask_analytics_opt_in() -> bool:
    try:
        import questionary

        answer = questionary.select(
            "Help improve ppmlx by sending anonymous beta usage analytics?",
            choices=[
                questionary.Choice("Yes — help the tiny beta goblin learn 🐣", True),
                questionary.Choice("No — not today", False),
            ],
            default=False,
            use_indicator=True,
        ).ask()
        return bool(answer) if answer is not None else False
    except Exception:
        answer = input("Enable anonymous beta analytics? [y/N] ").strip().lower()
        return answer in ("y", "yes")


def _save_analytics_preference(enabled: bool) -> None:
    """Save analytics preference to config.toml."""
    import tomli_w
    cfg_path = get_ppmlx_dir() / "config.toml"
    data: dict = {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        pass
    analytics = data.setdefault("analytics", {})
    analytics["enabled"] = enabled
    analytics.setdefault("provider", "posthog")
    analytics.setdefault("host", DEFAULT_ANALYTICS_HOST)
    analytics.setdefault("project_api_key", DEFAULT_ANALYTICS_PROJECT_API_KEY)
    analytics.setdefault("respect_do_not_track", True)
    with open(cfg_path, "wb") as f:
        tomli_w.dump(data, f)
