import sys
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
    merged = {**defaults, **user_aliases}
    sys.modules["ppmlx.models"].DEFAULT_ALIASES = defaults
    sys.modules["ppmlx.models"].load_user_aliases = MagicMock(return_value=user_aliases)
    sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value=merged)
    sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=local_models)
    sys.modules["ppmlx.models"].load_favorites = MagicMock(return_value=favorites)
    sys.modules["ppmlx.registry"].registry_entries = MagicMock(return_value={})


def test_list_command_empty():
    """list command shows 'No models' message when no models are downloaded."""
    _setup_model_mocks()

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models" in result.output


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

    with patch("ppmlx.tui.browse_models") as mock_browse:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_browse.assert_called_once()
        rows = mock_browse.call_args[0][0]
        aliases = [r.alias for r in rows if r.section_header is None]
        assert "llama3" in aliases


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
