import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

# Mock all ppmlx modules before importing cli
for mod_name in ["ppmlx.models", "ppmlx.engine", "ppmlx.db",
                  "ppmlx.config", "ppmlx.memory",
                  "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
                  "ppmlx.registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.cli import app

runner = CliRunner()


def test_version():
    """--version returns current version and exits 0."""
    from ppmlx import __version__
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help():
    """--help exits 0 and mentions ppmlx."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ppmlx" in result.output


def _setup_model_mocks(
    defaults=None,
    user_aliases=None,
    local_models=None,
    favorites=None,
    registry_entries=None,
):
    """Configure ppmlx.models mocks for commands using _build_model_records."""
    if defaults is None:
        defaults = {}
    if user_aliases is None:
        user_aliases = {}
    if local_models is None:
        local_models = []
    if favorites is None:
        favorites = []
    if registry_entries is None:
        registry_entries = {}
    merged = {**defaults, **user_aliases}
    sys.modules["ppmlx.models"].DEFAULT_ALIASES = defaults
    sys.modules["ppmlx.models"].load_user_aliases = MagicMock(return_value=user_aliases)
    sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value=merged)
    sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=local_models)
    sys.modules["ppmlx.models"].load_favorites = MagicMock(return_value=favorites)
    sys.modules["ppmlx.registry"].registry_entries = MagicMock(return_value=registry_entries)


def test_list_command_empty():
    """list command shows 'No models' message when no models are downloaded."""
    _setup_model_mocks()

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models" in result.output


def test_visible_rows_filters_by_selected_column():
    """Model table search can target any table column, with All as broad search."""
    from ppmlx.cli import _PickerRow, _visible_rows

    rows = [
        _PickerRow("", None, False, "Available"),
        _PickerRow("qwen", 4.0, False, None, params_b=7.0, precision="4bit", downloads=150_000, updated_at="2026-05-18"),
        _PickerRow("gemma", 8.0, False, None, params_b=27.0, precision="bf16", downloads=2_500, updated_at="2025-01-01"),
    ]

    assert [r.alias for r in _visible_rows(rows, "bf16", "precision") if r.section_header is None] == ["gemma"]
    assert [r.alias for r in _visible_rows(rows, "150k", "downloads") if r.section_header is None] == ["qwen"]
    assert [r.alias for r in _visible_rows(rows, "8.0", "size") if r.section_header is None] == ["gemma"]
    assert [r.alias for r in _visible_rows(rows, "2026", "updated") if r.section_header is None] == ["qwen"]
    assert [r.alias for r in _visible_rows(rows, "27", "all") if r.section_header is None] == ["gemma"]


def test_sort_rows_sorts_active_column_with_direction():
    """Rows can be sorted asc/desc by the active filter column."""
    from ppmlx.cli import _FILTER_COLUMNS, _PickerRow, _sort_rows

    rows = [
        _PickerRow("", None, False, "Available"),
        _PickerRow("qwen", 4.0, False, None, params_b=7.0, precision="4bit", downloads=150_000, updated_at="2026-05-18"),
        _PickerRow("gemma", 8.0, False, None, params_b=27.0, precision="bf16", downloads=2_500, updated_at="2025-01-01"),
    ]

    assert _FILTER_COLUMNS == ["alias", "params", "precision", "size", "downloads", "updated", "all"]
    assert [r.alias for r in _sort_rows(rows, "params") if r.section_header is None] == ["qwen", "gemma"]
    assert [r.alias for r in _sort_rows(rows, "params", descending=True) if r.section_header is None] == ["gemma", "qwen"]
    assert [r.alias for r in _sort_rows(rows, "size", descending=True) if r.section_header is None] == ["gemma", "qwen"]
    assert [r.alias for r in _sort_rows(rows, "downloads", descending=True) if r.section_header is None] == ["qwen", "gemma"]
    assert [r.alias for r in _sort_rows(rows, "updated", descending=True) if r.section_header is None] == ["qwen", "gemma"]


def test_build_picker_rows_limits_available_by_downloads():
    """available_limit caps available pull rows, keeping highest-download registry entries."""
    from ppmlx.cli import _build_picker_rows

    registry_entries = {
        "low:1b": {"repo_id": "mlx-community/Low-1B-4bit", "downloads": 100},
        "high:1b": {"repo_id": "mlx-community/High-1B-4bit", "downloads": 5000},
        "mid:1b": {"repo_id": "mlx-community/Mid-1B-4bit", "downloads": 1000},
    }
    _setup_model_mocks(registry_entries=registry_entries)

    rows = _build_picker_rows(local_only=False, available_limit=2)
    aliases = [r.alias for r in rows if r.section_header is None]

    assert aliases == ["high:1b", "mid:1b"]


def test_build_picker_rows_available_contains_only_hf_registry_models():
    """User aliases that are not downloaded are not shown in Available."""
    from ppmlx.cli import _build_picker_rows

    registry_entries = {
        "hf:1b": {"repo_id": "mlx-community/HF-1B-4bit", "downloads": 1000},
    }
    _setup_model_mocks(
        user_aliases={"custom:1b": "myorg/Custom-1B"},
        registry_entries=registry_entries,
    )

    rows = _build_picker_rows(local_only=False)
    aliases = [r.alias for r in rows if r.section_header is None]

    assert aliases == ["hf:1b"]


def test_build_picker_rows_downloaded_contains_user_downloads_only_there():
    """Downloaded user/local models are shown under Downloaded, not Available."""
    from ppmlx.cli import _build_picker_rows

    local_models = [
        {
            "name": "myorg--Custom-1B",
            "alias": "myorg/Custom-1B",
            "repo_id": "myorg/Custom-1B",
            "size_gb": 1.0,
            "path": "/tmp/myorg--Custom-1B",
        }
    ]
    _setup_model_mocks(
        user_aliases={"custom:1b": "myorg/Custom-1B"},
        local_models=local_models,
    )

    rows = _build_picker_rows(local_only=False)
    sections: dict[str, list[str]] = {}
    current = ""
    for row in rows:
        if row.section_header is not None:
            current = row.section_header
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(row.alias)

    assert sections == {"Downloaded": ["custom:1b"]}


def test_list_command_with_model():
    """list command opens TUI browser when models are present."""
    from unittest.mock import patch

    mock_models = [
        {
            "name": "Meta-Llama-3-8B-Instruct-4bit",
            "alias": "llama3",
            "repo_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "size_gb": 4.5,
            "path": "/Users/test/.ppmlx/models/llama3",
        }
    ]
    _setup_model_mocks(
        defaults={"llama3": "mlx-community/Meta-Llama-3-8B-Instruct-4bit"},
        local_models=mock_models,
    )

    with patch("ppmlx.cli._is_interactive_terminal", return_value=True), patch("ppmlx.tui.browse_models") as mock_browse:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_browse.assert_called_once()
        rows = mock_browse.call_args[0][0]
        aliases = [r.alias for r in rows if r.section_header is None]
        assert "llama3" in aliases


def test_list_command_non_tty_prints_plain_text():
    """list command does not launch TUI in non-interactive environments."""
    mock_models = [
        {
            "name": "Meta-Llama-3-8B-Instruct-4bit",
            "alias": "llama3",
            "repo_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "size_gb": 4.5,
            "path": "/Users/test/.ppmlx/models/llama3",
        }
    ]
    _setup_model_mocks(
        defaults={"llama3": "mlx-community/Meta-Llama-3-8B-Instruct-4bit"},
        local_models=mock_models,
    )

    with patch("ppmlx.cli._is_interactive_terminal", return_value=False), patch("ppmlx.tui.browse_models") as mock_browse:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_browse.assert_not_called()
        assert "Local Models" in result.output
        assert "llama3" in result.output


def test_config_command_non_tty_prints_redacted_config(tmp_path):
    """config command prints safe text config instead of launching TUI without a terminal."""
    fake_config_dir = tmp_path / ".ppmlx"
    fake_config_dir.mkdir()
    (fake_config_dir / "config.toml").write_text(
        "[auth]\nhf_token = 'secret-token'\n"
        "[server]\nmax_tools_tokens = 3000\n"
        "[defaults]\nmax_tokens = 2048\n"
    )
    sys.modules["ppmlx.config"].get_ppmlx_dir = MagicMock(return_value=fake_config_dir)

    with patch("ppmlx.cli._is_interactive_terminal", return_value=False), patch("ppmlx.tui.config_menu") as mock_menu:
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        mock_menu.assert_not_called()
        assert "Config:" in result.output
        assert "hf_token = \"***\"" in result.output
        assert "secret-token" not in result.output
        assert "max_tokens = 2048" in result.output
        assert "max_tools_tokens = 3000" in result.output


def test_memory_config_command_sets_memory_flags(tmp_path):
    """memory config command writes memory on/off and extraction settings."""
    import tomllib

    fake_config_dir = tmp_path / ".ppmlx"
    fake_config_dir.mkdir()
    sys.modules["ppmlx.config"].get_ppmlx_dir = MagicMock(return_value=fake_config_dir)

    result = runner.invoke(app, [
        "memory", "config",
        "--enabled",
        "--extractor", "model_memory_json",
        "--model", "qwen3.5:0.8b",
        "--max-jobs-per-event", "7",
        "--output-limit", "900",
        "--workers", "2",
        "--input-limit", "4096",
        "--overlap", "512",
        "--max-chunks", "12",
        "--timeout", "30",
    ])

    assert result.exit_code == 0
    with open(fake_config_dir / "config.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["memory"]["enabled"] is True
    assert data["memory"]["mode"] == "shadow"
    assert data["memory"]["extractor"] == "model_memory_json"
    assert data["memory"]["extraction_model"] == "qwen3.5:0.8b"
    assert data["memory"]["max_candidates_per_event"] == 7
    assert data["memory"]["extraction_max_tokens"] == 900
    assert data["memory"]["extraction_workers"] == 2
    assert data["memory"]["extraction_input_tokens"] == 4096
    assert data["memory"]["extraction_overlap_tokens"] == 512
    assert data["memory"]["extraction_max_chunks_per_event"] == 12
    assert data["memory"]["extraction_timeout_seconds"] == 30.0


def test_memory_config_command_non_tty_prints_toml_section(tmp_path):
    fake_config_dir = tmp_path / ".ppmlx"
    fake_config_dir.mkdir()
    sys.modules["ppmlx.config"].get_ppmlx_dir = MagicMock(return_value=fake_config_dir)

    with patch("ppmlx.cli._is_interactive_terminal", return_value=False):
        result = runner.invoke(app, ["memory", "config"])

    assert result.exit_code == 0
    assert "Memory config:" in result.output
    assert "[memory]" in result.output


def test_config_command_help_does_not_include_memory_flags():
    result = runner.invoke(app, ["config", "--help"])

    assert result.exit_code == 0
    assert "--memory-extractor" not in result.output
    assert "--memory-mode" not in result.output


def test_pull_command():
    """pull command calls download_model with the correct model name."""
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["ppmlx.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(return_value="mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    sys.modules["ppmlx.models"].download_model = MagicMock(return_value="/tmp/mistral")
    sys.modules["ppmlx.memory"].check_memory_warning = MagicMock(return_value=None)
    sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

    result = runner.invoke(app, ["pull", "mistral"])
    assert result.exit_code == 0
    sys.modules["ppmlx.models"].download_model.assert_called_once()
    call_args = sys.modules["ppmlx.models"].download_model.call_args
    assert call_args[0][0] == "mistral" or call_args[1].get("model") == "mistral" or "mistral" in str(call_args)


def test_pull_refreshes_registry_before_interactive_picker():
    """pull --refresh force-refreshes registry before opening the selector."""
    cfg = SimpleNamespace(registry=SimpleNamespace(display_limit=25))
    with (
        patch("ppmlx.registry.refresh_registry") as mock_refresh,
        patch("ppmlx.config.load_config", return_value=cfg),
        patch("ppmlx.tui.pick_models", return_value=[]) as mock_pick,
    ):
        result = runner.invoke(app, ["pull", "--refresh"])
    assert result.exit_code == 0
    mock_refresh.assert_called_once()
    mock_pick.assert_called_once_with(local_only=False, available_limit=25)
    assert "Registry refreshed" in result.output


def test_pull_unknown_model():
    """pull command exits with code 1 when model is not found."""
    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["ppmlx.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=ModelNotFoundError("not found"))

    result = runner.invoke(app, ["pull", "nonexistent-model-xyz"])
    assert result.exit_code == 1


def test_rm_with_force():
    """rm --force skips confirmation and calls remove_model."""
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(return_value="mlx-community/some-model")
    sys.modules["ppmlx.models"].get_model_path = MagicMock(return_value="/tmp/some-model")
    sys.modules["ppmlx.models"].remove_model = MagicMock(return_value=True)

    result = runner.invoke(app, ["rm", "some-model", "--force"])
    assert result.exit_code == 0
    sys.modules["ppmlx.models"].remove_model.assert_called_once_with("some-model")


def test_serve_help():
    """serve --help shows available options."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output or "host" in result.output
    assert "--port" in result.output or "port" in result.output


def _make_real_db(tmp_path):
    """Create a real ppmlx Database for testing logs/stats commands."""
    import importlib
    # The db module may have been replaced with a MagicMock; reload the real one.
    import ppmlx.db
    importlib.reload(ppmlx.db)
    db = ppmlx.db.Database(tmp_path / "ppmlx.db")
    db.init()
    return db


def test_logs_command_empty_db(tmp_path):
    """logs command with empty DB prints a friendly message and doesn't crash."""
    from unittest.mock import patch

    db = _make_real_db(tmp_path)
    db.flush()

    with patch("ppmlx.config.get_ppmlx_dir", return_value=tmp_path), \
         patch("ppmlx.db.get_db", return_value=db):
        result = runner.invoke(app, ["logs"])
    assert result.exit_code == 0
    assert "No requests" in result.output or "no" in result.output.lower()
    db.close()


def test_stats_command_empty_db(tmp_path):
    """stats command with empty DB prints a friendly message and doesn't crash."""
    from unittest.mock import patch

    db = _make_real_db(tmp_path)
    db.flush()

    with patch("ppmlx.config.get_ppmlx_dir", return_value=tmp_path), \
         patch("ppmlx.db.get_db", return_value=db):
        result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "No requests" in result.output or "no" in result.output.lower()
    db.close()


def test_logs_json_output(tmp_path):
    """logs --json produces valid JSON output when there are requests."""
    import json as json_mod
    from unittest.mock import patch

    db = _make_real_db(tmp_path)
    db.log_request(
        request_id="test-1",
        endpoint="/v1/chat/completions",
        model_alias="llama3",
        model_repo="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        status="ok",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        total_duration_ms=500.0,
        tokens_per_second=40.0,
        time_to_first_token_ms=50.0,
    )
    db.flush()

    with patch("ppmlx.config.get_ppmlx_dir", return_value=tmp_path), \
         patch("ppmlx.db.get_db", return_value=db):
        result = runner.invoke(app, ["logs", "--json"])
    assert result.exit_code == 0
    parsed = json_mod.loads(result.output)
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
    assert parsed[0]["model_alias"] == "llama3"
    db.close()
# ── pull --quantize tests ────────────────────────────────────────────────


def _setup_pull_quantize_mocks():
    """Set up mocks for pull --quantize tests."""
    from pathlib import Path

    ModelNotFoundError = type("ModelNotFoundError", (Exception,), {})
    sys.modules["ppmlx.models"].ModelNotFoundError = ModelNotFoundError
    sys.modules["ppmlx.models"].resolve_alias = MagicMock(
        return_value="mlx-community/Mistral-7B-Instruct-v0.3"
    )
    sys.modules["ppmlx.models"].download_model = MagicMock(
        return_value=Path("/tmp/mistral-fp")
    )
    sys.modules["ppmlx.memory"].check_memory_warning = MagicMock(return_value=None)
    sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

    QuantizationError = type("QuantizationError", (Exception,), {})
    QuantizeConfig = MagicMock()
    sys.modules["ppmlx.quantize"].QuantizationError = QuantizationError
    sys.modules["ppmlx.quantize"].QuantizeConfig = QuantizeConfig
    sys.modules["ppmlx.quantize"].quantize = MagicMock(
        return_value=Path("/tmp/mistral-4bit")
    )
    return QuantizeConfig, QuantizationError


def test_pull_quantize_downloads_and_quantizes():
    """pull --quantize downloads the model then runs quantization."""
    QuantizeConfig, _ = _setup_pull_quantize_mocks()

    with patch("shutil.rmtree") as mock_rmtree:
        result = runner.invoke(app, ["pull", "mistral", "--quantize"])

    assert result.exit_code == 0
    # Download should be called
    sys.modules["ppmlx.models"].download_model.assert_called_once()
    # Quantize should be called
    sys.modules["ppmlx.quantize"].quantize.assert_called_once()
    call_kwargs = sys.modules["ppmlx.quantize"].quantize.call_args
    # Should pass local_path keyword
    assert "local_path" in call_kwargs.kwargs or (
        len(call_kwargs) > 1 and call_kwargs[1].get("local_path") is not None
    )


def test_pull_quantize_with_bits():
    """pull --quantize --bits 8 passes the correct bit depth."""
    QuantizeConfig, _ = _setup_pull_quantize_mocks()

    with patch("shutil.rmtree"):
        result = runner.invoke(app, ["pull", "mistral", "--quantize", "--bits", "8"])

    assert result.exit_code == 0
    # QuantizeConfig should have been called with bits=8
    cfg_call = QuantizeConfig.call_args
    assert cfg_call is not None
    assert cfg_call.kwargs.get("bits") == 8 or (cfg_call.args and 8 in cfg_call.args)


def test_pull_quantize_invalid_bits():
    """pull --quantize --bits 5 fails with a validation error."""
    _setup_pull_quantize_mocks()

    result = runner.invoke(app, ["pull", "mistral", "--quantize", "--bits", "5"])
    assert result.exit_code == 1
    assert "Invalid --bits" in result.output


def test_pull_quantize_keep_original():
    """pull --quantize --keep-original does not remove the original download."""
    _setup_pull_quantize_mocks()

    with patch("shutil.rmtree") as mock_rmtree:
        result = runner.invoke(
            app, ["pull", "mistral", "--quantize", "--keep-original"]
        )

    assert result.exit_code == 0
    # rmtree should NOT have been called since we asked to keep original
    mock_rmtree.assert_not_called()


def test_pull_quantize_removes_original_by_default():
    """pull --quantize without --keep-original removes the original download."""
    _setup_pull_quantize_mocks()

    with patch("shutil.rmtree") as mock_rmtree:
        result = runner.invoke(app, ["pull", "mistral", "--quantize"])

    assert result.exit_code == 0
    # rmtree should have been called to remove the original
    mock_rmtree.assert_called_once()


def test_pull_quantize_failure():
    """pull --quantize exits with 1 when quantization fails."""
    _, QuantizationError = _setup_pull_quantize_mocks()
    sys.modules["ppmlx.quantize"].quantize = MagicMock(
        side_effect=QuantizationError("conversion failed")
    )

    result = runner.invoke(app, ["pull", "mistral", "--quantize"])
    assert result.exit_code == 1
    assert "Quantization failed" in result.output


def test_pull_without_quantize_unchanged():
    """pull without --quantize still works as before (no quantization)."""
    _setup_pull_quantize_mocks()

    result = runner.invoke(app, ["pull", "mistral"])
    assert result.exit_code == 0
    # Quantize should NOT be called
    sys.modules["ppmlx.quantize"].quantize.assert_not_called()
