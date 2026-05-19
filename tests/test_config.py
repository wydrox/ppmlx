"""Tests for ppmlx.config module."""
from __future__ import annotations

import pytest

from ppmlx.config import (
    Config,
    DEFAULT_ANALYTICS_HOST,
    DEFAULT_ANALYTICS_PROJECT_API_KEY,
    DefaultsConfig,
    LoggingConfig,
    MemoryConfig,
    RegistryConfig,
    ServerConfig,
    ToolAwarenessConfig,
    get_ppmlx_dir,
    load_config,
)


class TestDefaultValues:
    def test_server_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 6767
        assert cfg.cors is True
        assert cfg.max_loaded_models == 2

    def test_defaults_config_defaults(self):
        cfg = DefaultsConfig()
        assert cfg.model == "qwen3.5:0.8b"
        assert cfg.embed_model == "embed:all-minilm"
        assert cfg.temperature == 0.7
        assert cfg.top_p == 1.0
        assert cfg.max_tokens == 2048

    def test_logging_defaults(self):
        cfg = LoggingConfig()
        assert cfg.enabled is True
        assert cfg.snapshot_interval_seconds == 60

    def test_memory_defaults(self):
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.wired_limit_mb == 0
        assert cfg.mode == "off"
        assert cfg.max_candidates_per_event == 12
        assert cfg.rolling_tokens == 10000
        assert cfg.hot_tail_tokens == 6500
        assert cfg.session_context_tokens == 2000
        assert cfg.compact_threshold_tokens == 12000
        assert cfg.max_context_items == 40
        assert cfg.extractor == "hybrid"
        assert cfg.extraction_model == "gemma-4-e2b"
        assert cfg.extraction_workers == 1
        assert cfg.extraction_max_tokens == 1200
        assert cfg.extraction_input_tokens == 6000
        assert cfg.extraction_overlap_tokens == 600
        assert cfg.extraction_max_chunks_per_event == 32
        assert cfg.extraction_timeout_seconds == 45.0

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.server.port == 6767
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.cors is True
        assert cfg.server.max_loaded_models == 2
        assert cfg.tool_awareness.mode == "no_tools_only"
        assert cfg.analytics.enabled is False
        assert cfg.analytics.provider == "posthog"
        assert cfg.analytics.host == DEFAULT_ANALYTICS_HOST
        assert cfg.analytics.project_api_key == DEFAULT_ANALYTICS_PROJECT_API_KEY

    def test_tool_awareness_defaults(self):
        cfg = ToolAwarenessConfig()
        assert cfg.mode == "no_tools_only"

    def test_registry_defaults(self):
        cfg = RegistryConfig()
        assert cfg.enabled is True
        assert cfg.refresh == "weekly"
        assert cfg.display_limit == 50


class TestLoadConfigDefaults:
    def test_no_file_returns_defaults(self, tmp_home):
        cfg = load_config()
        assert cfg.server.port == 6767
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.cors is True
        assert cfg.server.max_loaded_models == 2
        assert cfg.defaults.model == "qwen3.5:0.8b"
        assert cfg.defaults.temperature == 0.7
        assert cfg.defaults.max_tokens == 2048
        assert cfg.tool_awareness.mode == "no_tools_only"


class TestTomlLoading:
    def test_load_from_toml(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        toml_content = """
[server]
host = "0.0.0.0"
port = 8080
cors = false
max_loaded_models = 4

[defaults]
model = "llama3:8b"
embed_model = "embed:custom"
temperature = 0.5
top_p = 0.9
max_tokens = 4096

[logging]
enabled = false
snapshot_interval_seconds = 120

[memory]
enabled = true
wired_limit_mb = 1024
mode = "compact"
max_candidates_per_event = 8
rolling_tokens = 9000
hot_tail_tokens = 6000
session_context_tokens = 1800
compact_threshold_tokens = 11000
max_context_items = 30
extractor = "llm"
extraction_model = "qwen3.5:0.8b"
extraction_workers = 3
extraction_max_tokens = 900
extraction_input_tokens = 4096
extraction_overlap_tokens = 512
extraction_max_chunks_per_event = 16
extraction_timeout_seconds = 12.5

[tool_awareness]
mode = "all"

[registry]
enabled = true
refresh = "monthly"
display_limit = 25

[analytics]
enabled = false
provider = "posthog"
host = "https://stats.example.com"
project_api_key = "phc_test_123"
respect_do_not_track = true
"""
        (config_dir / "config.toml").write_text(toml_content)
        cfg = load_config()
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.port == 8080
        assert cfg.server.cors is False
        assert cfg.server.max_loaded_models == 4
        assert cfg.defaults.model == "llama3:8b"
        assert cfg.defaults.embed_model == "embed:custom"
        assert cfg.defaults.temperature == 0.5
        assert cfg.defaults.top_p == 0.9
        assert cfg.defaults.max_tokens == 4096
        assert cfg.logging.enabled is False
        assert cfg.logging.snapshot_interval_seconds == 120
        assert cfg.memory.enabled is True
        assert cfg.memory.wired_limit_mb == 1024
        assert cfg.memory.mode == "compact"
        assert cfg.memory.max_candidates_per_event == 8
        assert cfg.memory.rolling_tokens == 9000
        assert cfg.memory.hot_tail_tokens == 6000
        assert cfg.memory.session_context_tokens == 1800
        assert cfg.memory.compact_threshold_tokens == 11000
        assert cfg.memory.max_context_items == 30
        assert cfg.memory.extractor == "hybrid"
        assert cfg.memory.extraction_model == "qwen3.5:0.8b"
        assert cfg.memory.extraction_workers == 3
        assert cfg.memory.extraction_max_tokens == 900
        assert cfg.memory.extraction_input_tokens == 4096
        assert cfg.memory.extraction_overlap_tokens == 512
        assert cfg.memory.extraction_max_chunks_per_event == 16
        assert cfg.memory.extraction_timeout_seconds == 12.5
        assert cfg.tool_awareness.mode == "all"
        assert cfg.registry.enabled is True
        assert cfg.registry.refresh == "monthly"
        assert cfg.registry.display_limit == 25
        assert cfg.analytics.enabled is False
        assert cfg.analytics.provider == "posthog"
        assert cfg.analytics.host == "https://stats.example.com"
        assert cfg.analytics.project_api_key == "phc_test_123"
        assert cfg.analytics.respect_do_not_track is True

    def test_partial_toml(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        cfg = load_config()
        assert cfg.server.port == 9000
        assert cfg.server.host == "127.0.0.1"

    def test_malformed_toml_silently_ignored(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("this is not valid toml ][[[")
        cfg = load_config()
        assert cfg.server.port == 6767


class TestEnvVarOverrides:
    def test_port_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_PORT", "9999")
        cfg = load_config()
        assert cfg.server.port == 9999

    def test_host_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_HOST", "0.0.0.0")
        cfg = load_config()
        assert cfg.server.host == "0.0.0.0"

    def test_cors_false_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "false")
        cfg = load_config()
        assert cfg.server.cors is False

    def test_cors_zero_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "0")
        cfg = load_config()
        assert cfg.server.cors is False

    def test_cors_true_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "true")
        cfg = load_config()
        assert cfg.server.cors is True

    def test_cors_no_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_CORS", "no")
        cfg = load_config()
        assert cfg.server.cors is False

    def test_max_loaded_models_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MAX_LOADED_MODELS", "5")
        cfg = load_config()
        assert cfg.server.max_loaded_models == 5

    def test_default_model_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_DEFAULT_MODEL", "mistral:7b")
        cfg = load_config()
        assert cfg.defaults.model == "mistral:7b"

    def test_temperature_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_TEMP", "0.3")
        cfg = load_config()
        assert cfg.defaults.temperature == 0.3

    def test_max_tokens_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MAX_TOKENS", "512")
        cfg = load_config()
        assert cfg.defaults.max_tokens == 512

    def test_log_enabled_false(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_LOG_ENABLED", "false")
        cfg = load_config()
        assert cfg.logging.enabled is False

    def test_log_snapshot_interval(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_LOG_SNAPSHOT_INTERVAL", "300")
        cfg = load_config()
        assert cfg.logging.snapshot_interval_seconds == 300

    def test_memory_enabled_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_ENABLED", "false")
        cfg = load_config()
        assert cfg.memory.enabled is False

    def test_memory_wired_limit(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_WIRED_LIMIT", "2048")
        cfg = load_config()
        assert cfg.memory.wired_limit_mb == 2048

    def test_memory_mode_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_MODE", "shadow")
        cfg = load_config()
        assert cfg.memory.mode == "shadow"

    def test_memory_mode_bool_env_var_maps_to_shadow(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_MODE", "true")
        cfg = load_config()
        assert cfg.memory.mode == "shadow"

    def test_memory_max_candidates_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_MAX_CANDIDATES", "6")
        cfg = load_config()
        assert cfg.memory.max_candidates_per_event == 6

    def test_memory_compact_budget_env_vars(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_ROLLING_TOKENS", "9000")
        monkeypatch.setenv("PPMLX_MEMORY_HOT_TAIL_TOKENS", "5000")
        monkeypatch.setenv("PPMLX_MEMORY_SESSION_CONTEXT_TOKENS", "1500")
        monkeypatch.setenv("PPMLX_MEMORY_COMPACT_THRESHOLD_TOKENS", "10000")
        monkeypatch.setenv("PPMLX_MEMORY_MAX_CONTEXT_ITEMS", "25")
        cfg = load_config()
        assert cfg.memory.rolling_tokens == 9000
        assert cfg.memory.hot_tail_tokens == 5000
        assert cfg.memory.session_context_tokens == 1500
        assert cfg.memory.compact_threshold_tokens == 10000
        assert cfg.memory.max_context_items == 25

    def test_memory_extraction_env_vars(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTOR", "llm_json")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MODEL", "llama3:8b")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_WORKERS", "4")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MAX_TOKENS", "1000")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_INPUT_TOKENS", "4096")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_OVERLAP_TOKENS", "512")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_MAX_CHUNKS", "16")
        monkeypatch.setenv("PPMLX_MEMORY_EXTRACTION_TIMEOUT", "30.5")
        cfg = load_config()
        assert cfg.memory.extractor == "hybrid"
        assert cfg.memory.extraction_model == "llama3:8b"
        assert cfg.memory.extraction_workers == 4
        assert cfg.memory.extraction_max_tokens == 1000
        assert cfg.memory.extraction_input_tokens == 4096
        assert cfg.memory.extraction_overlap_tokens == 512
        assert cfg.memory.extraction_max_chunks_per_event == 16
        assert cfg.memory.extraction_timeout_seconds == 30.5

    def test_invalid_env_var_ignored(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_PORT", "not_a_number")
        cfg = load_config()
        assert cfg.server.port == 6767

    def test_tool_awareness_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_INJECT_TOOL_AWARENESS", "all")
        cfg = load_config()
        assert cfg.tool_awareness.mode == "all"

    def test_registry_refresh_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_REGISTRY_REFRESH", "never")
        cfg = load_config()
        assert cfg.registry.refresh == "never"

    def test_invalid_registry_refresh_env_var_falls_back(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_REGISTRY_REFRESH", "hourly")
        cfg = load_config()
        assert cfg.registry.refresh == "weekly"

    def test_registry_display_limit_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_REGISTRY_DISPLAY_LIMIT", "25")
        cfg = load_config()
        assert cfg.registry.display_limit == 25

    def test_registry_display_limit_env_var_is_clamped(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_REGISTRY_DISPLAY_LIMIT", "500")
        cfg = load_config()
        assert cfg.registry.display_limit == 100

    def test_tool_awareness_env_var_legacy_true_maps_to_all(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_INJECT_TOOL_AWARENESS", "true")
        cfg = load_config()
        assert cfg.tool_awareness.mode == "all"

    def test_tool_awareness_env_var_legacy_false_maps_to_off(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_INJECT_TOOL_AWARENESS", "false")
        cfg = load_config()
        assert cfg.tool_awareness.mode == "off"

    def test_analytics_enabled_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_ENABLED", "false")
        cfg = load_config()
        assert cfg.analytics.enabled is False

    def test_analytics_host_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_HOST", "https://stats.example.com")
        cfg = load_config()
        assert cfg.analytics.host == "https://stats.example.com"

    def test_analytics_project_api_key_env_var(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_PROJECT_API_KEY", "phc_test_123")
        cfg = load_config()
        assert cfg.analytics.project_api_key == "phc_test_123"

    def test_analytics_legacy_website_id_env_var_maps_to_project_key(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_ANALYTICS_WEBSITE_ID", "legacy-site-123")
        cfg = load_config()
        assert cfg.analytics.project_api_key == "legacy-site-123"

    def test_analytics_legacy_tunnel_host_maps_to_default_ingest(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            '[analytics]\nhost = "https://analytics.ppmlx.dev"\n'
        )
        cfg = load_config()
        assert cfg.analytics.host == DEFAULT_ANALYTICS_HOST


class TestCliOverrides:
    def test_port_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"port": 8080})
        assert cfg.server.port == 8080

    def test_host_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"host": "0.0.0.0"})
        assert cfg.server.host == "0.0.0.0"

    def test_cors_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"cors": False})
        assert cfg.server.cors is False

    def test_model_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"model": "phi3:mini"})
        assert cfg.defaults.model == "phi3:mini"

    def test_temperature_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"temperature": 0.1})
        assert cfg.defaults.temperature == 0.1

    def test_max_tokens_cli_override(self, tmp_home):
        cfg = load_config(cli_overrides={"max_tokens": 100})
        assert cfg.defaults.max_tokens == 100

    def test_none_value_ignored(self, tmp_home):
        cfg = load_config(cli_overrides={"port": None})
        assert cfg.server.port == 6767


class TestPriority:
    def test_cli_overrides_env(self, tmp_home, monkeypatch):
        monkeypatch.setenv("PPMLX_PORT", "9000")
        cfg = load_config(cli_overrides={"port": 7777})
        assert cfg.server.port == 7777

    def test_env_overrides_toml(self, tmp_home, monkeypatch):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        monkeypatch.setenv("PPMLX_PORT", "8888")
        cfg = load_config()
        assert cfg.server.port == 8888

    def test_cli_overrides_toml(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        cfg = load_config(cli_overrides={"port": 5555})
        assert cfg.server.port == 5555

    def test_cli_overrides_env_overrides_toml(self, tmp_home, monkeypatch):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        monkeypatch.setenv("PPMLX_PORT", "8888")
        cfg = load_config(cli_overrides={"port": 7777})
        assert cfg.server.port == 7777

    def test_toml_overrides_defaults(self, tmp_home):
        config_dir = tmp_home / ".ppmlx"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text("[server]\nport = 9000\n")
        cfg = load_config()
        assert cfg.server.port == 9000
        assert cfg.server.host == "127.0.0.1"


class TestGetPpLlmDir:
    def test_creates_directory(self, tmp_home):
        d = get_ppmlx_dir()
        assert d.exists()
        assert d.is_dir()
        assert d == tmp_home / ".ppmlx"

    def test_idempotent(self, tmp_home):
        d1 = get_ppmlx_dir()
        d2 = get_ppmlx_dir()
        assert d1 == d2
        assert d1.exists()


class TestFirstRunAnalyticsOnboarding:
    def test_save_analytics_preference_writes_default_sink(self, tmp_home, monkeypatch):
        from ppmlx.config import _save_analytics_preference

        monkeypatch.delenv("PPMLX_ANALYTICS_ENABLED", raising=False)
        _save_analytics_preference(True)
        cfg = load_config()
        assert cfg.analytics.enabled is True
        assert cfg.analytics.provider == "posthog"
        assert cfg.analytics.host == DEFAULT_ANALYTICS_HOST
        assert cfg.analytics.project_api_key == DEFAULT_ANALYTICS_PROJECT_API_KEY
        assert cfg.analytics.respect_do_not_track is True

    def test_check_first_run_uses_selector_choice(self, tmp_home, monkeypatch):
        import sys
        from ppmlx import config as config_module

        monkeypatch.delenv("PPMLX_ANALYTICS_ENABLED", raising=False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(config_module, "_ask_analytics_opt_in", lambda: True)

        config_module.check_first_run()

        assert (tmp_home / ".ppmlx" / ".first_run_done").exists()
        assert load_config().analytics.enabled is True
