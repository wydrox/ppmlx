"""Tests for ppmlx.processor — batch document processing."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure ppmlx modules are stubbed (conftest handles MLX stubs)
for mod_name in ["ppmlx.models", "ppmlx.engine", "ppmlx.db",
                  "ppmlx.config", "ppmlx.memory",
                  "ppmlx.quantize", "ppmlx.engine_embed", "ppmlx.engine_vlm",
                  "ppmlx.registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from ppmlx.processor import (
    BUILTIN_TASKS,
    SUPPORTED_EXTENSIONS,
    BatchStats,
    CheckpointState,
    ProcessResult,
    TaskDefinition,
    discover_files,
    make_custom_task,
    process_batch,
    process_single_file,
    _read_file_content,
)


# ── File discovery tests ──────────────────────────────────────────────


class TestDiscoverFiles:
    def test_finds_supported_files(self, tmp_path):
        """discover_files returns files with supported extensions."""
        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "data.csv").write_text("a,b\n1,2")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"readme.md", "main.py", "data.csv"}

    def test_skips_binary_extensions(self, tmp_path):
        """discover_files ignores files with unsupported extensions."""
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")
        (tmp_path / "ok.txt").write_text("hello")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"ok.txt"}

    def test_skips_hidden_files(self, tmp_path):
        """discover_files skips dotfiles."""
        (tmp_path / ".hidden.txt").write_text("secret")
        (tmp_path / "visible.txt").write_text("public")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"visible.txt"}

    def test_skips_system_files(self, tmp_path):
        """discover_files skips well-known system files."""
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")
        (tmp_path / "ok.txt").write_text("hello")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"ok.txt"}

    def test_recursive_discovery(self, tmp_path):
        """discover_files recurses into subdirectories."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top")
        (sub / "nested.txt").write_text("nested")

        files = discover_files(tmp_path, recursive=True)
        names = {f.name for f in files}
        assert names == {"top.txt", "nested.txt"}

    def test_non_recursive(self, tmp_path):
        """discover_files with recursive=False only finds top-level files."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top")
        (sub / "nested.txt").write_text("nested")

        files = discover_files(tmp_path, recursive=False)
        names = {f.name for f in files}
        assert names == {"top.txt"}

    def test_respects_gitignore(self, tmp_path):
        """discover_files respects .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("ignored.txt\n*.log\n")
        (tmp_path / "ignored.txt").write_text("skip me")
        (tmp_path / "debug.log").write_text("skip me too")
        (tmp_path / "keep.txt").write_text("keep me")

        files = discover_files(tmp_path, respect_gitignore=True)
        names = {f.name for f in files}
        assert names == {"keep.txt"}

    def test_skips_empty_files(self, tmp_path):
        """discover_files skips empty (0-byte) files."""
        (tmp_path / "empty.txt").write_text("")
        (tmp_path / "content.txt").write_text("hello")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"content.txt"}

    def test_skips_large_files(self, tmp_path):
        """discover_files skips files larger than 1 MB."""
        (tmp_path / "large.txt").write_text("x" * 1_100_000)
        (tmp_path / "small.txt").write_text("hello")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"small.txt"}

    def test_custom_extensions(self, tmp_path):
        """discover_files respects custom extension filters."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.md").write_text("world")

        files = discover_files(tmp_path, extensions={".txt"})
        names = {f.name for f in files}
        assert names == {"a.txt"}

    def test_returns_sorted(self, tmp_path):
        """discover_files returns files in sorted order."""
        (tmp_path / "c.txt").write_text("c")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        files = discover_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["a.txt", "b.txt", "c.txt"]

    def test_empty_directory(self, tmp_path):
        """discover_files returns empty list for empty directory."""
        files = discover_files(tmp_path)
        assert files == []


# ── Task definitions tests ────────────────────────────────────────────


class TestTaskDefinitions:
    def test_builtin_tasks_exist(self):
        """All expected built-in tasks are defined."""
        assert "summarize" in BUILTIN_TASKS
        assert "translate" in BUILTIN_TASKS
        assert "extract_entities" in BUILTIN_TASKS
        assert "classify" in BUILTIN_TASKS

    def test_task_has_system_prompt(self):
        """Each task has a non-empty system prompt."""
        for name, task in BUILTIN_TASKS.items():
            assert task.system_prompt, f"Task {name} has empty system prompt"
            assert task.output_suffix, f"Task {name} has empty output suffix"

    def test_custom_task(self):
        """make_custom_task creates a task with the given prompt."""
        task = make_custom_task("Explain this code")
        assert task.name == "custom"
        assert "Explain this code" in task.system_prompt
        assert task.output_suffix == ".processed.md"

    def test_build_user_message(self):
        """TaskDefinition.build_user_message includes filename and content."""
        task = BUILTIN_TASKS["summarize"]
        msg = task.build_user_message("Hello world", "test.txt")
        assert "test.txt" in msg
        assert "Hello world" in msg


# ── Checkpoint tests ──────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        """Checkpoint can be saved and loaded."""
        cp = CheckpointState(
            task="summarize",
            model="llama3",
            directory=str(tmp_path),
            started_at="2024-01-01T00:00:00",
        )
        cp.mark_completed("/path/to/file1.txt")
        cp.mark_failed("/path/to/file2.txt", "timeout")
        cp.save(tmp_path)

        loaded = CheckpointState.load(tmp_path)
        assert loaded is not None
        assert loaded.task == "summarize"
        assert loaded.model == "llama3"
        assert loaded.is_completed("/path/to/file1.txt")
        assert not loaded.is_completed("/path/to/file2.txt")
        assert loaded.failed["/path/to/file2.txt"] == "timeout"

    def test_load_nonexistent(self, tmp_path):
        """Loading from a directory without checkpoint returns None."""
        assert CheckpointState.load(tmp_path) is None

    def test_load_corrupt(self, tmp_path):
        """Loading corrupt checkpoint returns None."""
        (tmp_path / ".ppmlx-process-state.json").write_text("{bad json")
        assert CheckpointState.load(tmp_path) is None

    def test_cleanup(self, tmp_path):
        """Cleanup removes the checkpoint file."""
        cp = CheckpointState(task="summarize", model="m", directory=str(tmp_path))
        cp.save(tmp_path)
        assert (tmp_path / ".ppmlx-process-state.json").exists()
        cp.cleanup(tmp_path)
        assert not (tmp_path / ".ppmlx-process-state.json").exists()

    def test_mark_completed_clears_failed(self, tmp_path):
        """Marking a file completed removes it from the failed set."""
        cp = CheckpointState(task="t", model="m", directory=".")
        cp.mark_failed("file.txt", "error")
        assert "file.txt" in cp.failed
        cp.mark_completed("file.txt")
        assert "file.txt" not in cp.failed
        assert cp.is_completed("file.txt")


# ── Single-file processing tests ──────────────────────────────────────


class TestProcessSingleFile:
    def test_success(self, tmp_path):
        """process_single_file returns success with LLM output."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Summary: Hello world"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = process_single_file(
                test_file,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
            )

        assert result.success
        assert result.output_text == "Summary: Hello world"
        assert result.tokens_used == 15
        assert result.duration_secs >= 0

    def test_connection_error(self, tmp_path):
        """process_single_file handles connection errors."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")

        import httpx

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = process_single_file(
                test_file,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
            )

        assert not result.success
        assert "Cannot connect" in result.error

    def test_timeout_error(self, tmp_path):
        """process_single_file handles timeout errors."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")

        import httpx

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.TimeoutException("timeout")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = process_single_file(
                test_file,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
            )

        assert not result.success
        assert "timed out" in result.error

    def test_unreadable_file(self, tmp_path):
        """process_single_file handles unreadable files."""
        fake_path = tmp_path / "nonexistent.txt"

        result = process_single_file(
            fake_path,
            BUILTIN_TASKS["summarize"],
            "llama3",
            "http://localhost:6767",
        )

        assert not result.success
        assert "Failed to read" in result.error

    def test_output_dir(self, tmp_path):
        """process_single_file writes to specified output directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")
        out_dir = tmp_path / "output"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Summary"}}],
            "usage": {"total_tokens": 10},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = process_single_file(
                test_file,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                output_dir=out_dir,
            )

        assert result.success
        assert result.output_path is not None
        assert "output" in result.output_path
        assert result.output_path.endswith(".summary.md")

    def test_api_error(self, tmp_path):
        """process_single_file handles HTTP error responses."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")

        import httpx

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_resp,
            )
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = process_single_file(
                test_file,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
            )

        assert not result.success
        assert "API error" in result.error


# ── Batch processing tests ────────────────────────────────────────────


def _make_mock_client(responses: list[dict] | None = None):
    """Create a mock httpx.Client that returns canned API responses."""
    default_response = {
        "choices": [{"message": {"content": "Processed output"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    mock_client = MagicMock()
    if responses:
        side_effects = []
        for r in responses:
            mock_resp = MagicMock()
            mock_resp.json.return_value = r
            mock_resp.raise_for_status = MagicMock()
            side_effects.append(mock_resp)
        mock_client.post.side_effect = side_effects
    else:
        mock_resp = MagicMock()
        mock_resp.json.return_value = default_response
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestProcessBatch:
    def test_sequential_processing(self, tmp_path):
        """Batch processes files sequentially with parallel=1."""
        (tmp_path / "a.txt").write_text("File A")
        (tmp_path / "b.txt").write_text("File B")
        files = discover_files(tmp_path)

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            MockClient.return_value = _make_mock_client()

            results, stats = process_batch(
                files,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                parallel=1,
                stdout_mode=True,
            )

        assert stats.completed == 2
        assert stats.failed == 0
        assert stats.total_files == 2
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_parallel_processing(self, tmp_path):
        """Batch processes files in parallel."""
        (tmp_path / "a.txt").write_text("File A")
        (tmp_path / "b.txt").write_text("File B")
        (tmp_path / "c.txt").write_text("File C")
        files = discover_files(tmp_path)

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            MockClient.return_value = _make_mock_client()

            results, stats = process_batch(
                files,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                parallel=2,
                stdout_mode=True,
            )

        assert stats.completed == 3
        assert stats.failed == 0

    def test_resume_skips_completed(self, tmp_path):
        """Batch processing skips already-completed files on resume."""
        (tmp_path / "a.txt").write_text("File A")
        (tmp_path / "b.txt").write_text("File B")
        files = discover_files(tmp_path)

        checkpoint = CheckpointState(
            task="summarize", model="llama3", directory=str(tmp_path),
        )
        checkpoint.mark_completed(str(files[0]))

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            MockClient.return_value = _make_mock_client()

            results, stats = process_batch(
                files,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                checkpoint=checkpoint,
                checkpoint_dir=tmp_path,
                stdout_mode=True,
            )

        assert stats.skipped == 1
        assert stats.completed == 1
        assert len(results) == 1

    def test_writes_output_files(self, tmp_path):
        """Batch processing writes output files."""
        (tmp_path / "a.txt").write_text("File A")
        files = discover_files(tmp_path)
        out_dir = tmp_path / "output"

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            MockClient.return_value = _make_mock_client()

            results, stats = process_batch(
                files,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                output_dir=out_dir,
            )

        assert stats.completed == 1
        assert out_dir.exists()
        output_files = list(out_dir.iterdir())
        assert len(output_files) == 1
        assert output_files[0].name == "a.summary.md"

    def test_progress_callback(self, tmp_path):
        """Progress callback is called for each processed file."""
        (tmp_path / "a.txt").write_text("File A")
        files = discover_files(tmp_path)

        callback_calls = []

        def on_progress(fp, result):
            callback_calls.append((fp, result))

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            MockClient.return_value = _make_mock_client()

            results, stats = process_batch(
                files,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                progress_callback=on_progress,
                stdout_mode=True,
            )

        assert len(callback_calls) == 1
        assert callback_calls[0][1].success

    def test_checkpoint_updated_on_progress(self, tmp_path):
        """Checkpoint file is updated as files are processed."""
        (tmp_path / "a.txt").write_text("File A")
        files = discover_files(tmp_path)

        checkpoint = CheckpointState(
            task="summarize", model="llama3", directory=str(tmp_path),
        )

        with patch("ppmlx.processor.httpx.Client") as MockClient:
            MockClient.return_value = _make_mock_client()

            results, stats = process_batch(
                files,
                BUILTIN_TASKS["summarize"],
                "llama3",
                "http://localhost:6767",
                checkpoint=checkpoint,
                checkpoint_dir=tmp_path,
                stdout_mode=True,
            )

        # Checkpoint file should have been written
        cp_file = tmp_path / ".ppmlx-process-state.json"
        assert cp_file.exists()
        data = json.loads(cp_file.read_text())
        assert str(files[0]) in data["completed"]


# ── BatchStats tests ──────────────────────────────────────────────────


class TestBatchStats:
    def test_success_rate_all_success(self):
        stats = BatchStats(completed=5, failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_mixed(self):
        stats = BatchStats(completed=3, failed=2)
        assert stats.success_rate == 0.6

    def test_success_rate_no_files(self):
        stats = BatchStats(completed=0, failed=0)
        assert stats.success_rate == 0.0


# ── CLI integration tests ────────────────────────────────────────────


class TestProcessCLI:
    """Test the CLI 'process' command via typer runner."""

    @pytest.fixture
    def runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    @pytest.fixture
    def cli_app(self):
        from ppmlx.cli import app
        return app

    def test_process_help(self, runner, cli_app):
        """process --help shows usage information."""
        result = runner.invoke(cli_app, ["process", "--help"])
        assert result.exit_code == 0
        assert "summarize" in result.output
        assert "--task" in result.output
        assert "--model" in result.output

    def test_process_nonexistent_dir(self, runner, cli_app):
        """process exits with error for nonexistent directory."""
        result = runner.invoke(cli_app, ["process", "/nonexistent/path", "--model", "llama3"])
        assert result.exit_code == 1
        assert "Not a directory" in result.output

    def test_process_invalid_task(self, runner, cli_app, tmp_path):
        """process exits with error for invalid task name."""
        result = runner.invoke(cli_app, [
            "process", str(tmp_path),
            "--task", "nonexistent_task",
            "--model", "llama3",
        ])
        assert result.exit_code == 1
        assert "Unknown task" in result.output

    def test_process_no_model(self, runner, cli_app, tmp_path):
        """process exits with error when no model is specified."""
        result = runner.invoke(cli_app, ["process", str(tmp_path)])
        assert result.exit_code == 1
        assert "No model" in result.output

    def test_process_dry_run(self, runner, cli_app, tmp_path):
        """process --dry-run lists files without processing."""
        (tmp_path / "test.txt").write_text("Hello world content here")

        result = runner.invoke(cli_app, [
            "process", str(tmp_path),
            "--task", "summarize",
            "--model", "llama3",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "test.txt" in result.output
        assert "Dry Run" in result.output

    def test_process_empty_dir(self, runner, cli_app, tmp_path):
        """process exits cleanly for empty directory."""
        result = runner.invoke(cli_app, [
            "process", str(tmp_path),
            "--task", "summarize",
            "--model", "llama3",
        ])
        assert result.exit_code == 0
        assert "No processable files" in result.output

    def test_process_custom_task(self, runner, cli_app, tmp_path):
        """process accepts custom:prompt syntax for dry-run."""
        (tmp_path / "code.py").write_text("def hello(): pass")

        result = runner.invoke(cli_app, [
            "process", str(tmp_path),
            "--task", "custom:Explain this code",
            "--model", "llama3",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "code.py" in result.output

    def test_process_custom_task_empty_prompt(self, runner, cli_app, tmp_path):
        """process rejects custom: with empty prompt."""
        result = runner.invoke(cli_app, [
            "process", str(tmp_path),
            "--task", "custom:",
            "--model", "llama3",
        ])
        assert result.exit_code == 1
        assert "requires a prompt" in result.output


# ── Helper function tests ─────────────────────────────────────────────


class TestReadFileContent:
    def test_reads_utf8(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")
        assert _read_file_content(f) == "Hello world"

    def test_handles_encoding_errors(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"Hello \xff world")
        content = _read_file_content(f)
        assert "Hello" in content
        assert "world" in content
