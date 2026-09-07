"""Tests for ppmlx.agent — AgentRuntime with mocked engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ppmlx.agent import (
    AgentConfig,
    AgentRuntime,
    AgentStep,
    ToolResult,
    BUILTIN_TOOL_DEFINITIONS,
    BUILTIN_TOOL_NAMES,
    DEFAULT_COMMAND_ALLOWLIST,
    _resolve_path,
    _validate_path,
    execute_tool,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_engine(responses: list[str]) -> MagicMock:
    """Create a mock engine that returns predetermined responses.

    Each call to generate() pops the next response from the list.
    """
    engine = MagicMock()
    call_count = {"n": 0}

    def _generate(repo_id, messages, **kwargs):
        from ppmlx.engine import GenerateResult
        idx = call_count["n"]
        call_count["n"] += 1
        text = responses[idx] if idx < len(responses) else "Done."
        return GenerateResult(text=text, reasoning=None, prompt_tokens=10, completion_tokens=5)

    engine.generate.side_effect = _generate
    return engine


def _make_config(tmp_path: Path, **overrides) -> AgentConfig:
    """Create an AgentConfig with defaults pointing to tmp_path."""
    defaults = {
        "model": "test-model",
        "working_dir": str(tmp_path),
        "allowed_directories": [str(tmp_path)],
        "max_iterations": 5,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ── AgentConfig tests ────────────────────────────────────────────────────


class TestAgentConfig:
    def test_defaults(self, tmp_path):
        config = AgentConfig(model="m", working_dir=str(tmp_path))
        assert config.max_iterations == 10
        assert config.sandbox is False
        assert str(tmp_path) in config.allowed_directories
        assert config.command_allowlist is not None
        assert "ls" in config.command_allowlist

    def test_sandbox_restricts_allowlist(self, tmp_path):
        config = AgentConfig(
            model="m", working_dir=str(tmp_path), sandbox=True,
        )
        assert config.command_allowlist is not None
        assert "git" not in config.command_allowlist
        assert "python" not in config.command_allowlist
        assert "ls" in config.command_allowlist


# ── Path validation tests ────────────────────────────────────────────────


class TestPathValidation:
    def test_resolve_relative(self, tmp_path):
        resolved = _resolve_path("foo/bar.txt", str(tmp_path))
        assert resolved == (tmp_path / "foo" / "bar.txt").resolve()

    def test_resolve_absolute(self, tmp_path):
        abs_path = str(tmp_path / "abs.txt")
        resolved = _resolve_path(abs_path, "/other")
        assert resolved == Path(abs_path).resolve()

    def test_validate_within_allowed(self, tmp_path):
        result = _validate_path("test.txt", str(tmp_path), [str(tmp_path)])
        assert result == (tmp_path / "test.txt").resolve()

    def test_validate_rejects_escape(self, tmp_path):
        with pytest.raises(ValueError, match="outside allowed directories"):
            _validate_path("/etc/passwd", str(tmp_path), [str(tmp_path)])

    def test_validate_rejects_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="outside allowed directories"):
            _validate_path("../../etc/passwd", str(tmp_path), [str(tmp_path)])


# ── Built-in tool tests ─────────────────────────────────────────────────


class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!")
        result = execute_tool(
            "read_file",
            {"path": "hello.txt"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert result == "Hello, world!"

    def test_read_missing_file(self, tmp_path):
        result = execute_tool(
            "read_file",
            {"path": "missing.txt"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "not found" in result

    def test_read_no_path(self, tmp_path):
        result = execute_tool(
            "read_file",
            {},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "required" in result

    def test_read_outside_allowed(self, tmp_path):
        with pytest.raises(ValueError, match="outside allowed"):
            execute_tool(
                "read_file",
                {"path": "/etc/passwd"},
                working_dir=str(tmp_path),
                allowed_directories=[str(tmp_path)],
            )


class TestWriteFile:
    def test_write_creates_file(self, tmp_path):
        result = execute_tool(
            "write_file",
            {"path": "out.txt", "content": "test content"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "Successfully wrote" in result and "characters" in result
        assert (tmp_path / "out.txt").read_text() == "test content"

    def test_write_creates_subdirectories(self, tmp_path):
        result = execute_tool(
            "write_file",
            {"path": "sub/dir/out.txt", "content": "nested"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "Successfully wrote" in result and "characters" in result
        assert (tmp_path / "sub" / "dir" / "out.txt").read_text() == "nested"

    def test_write_blocked_in_sandbox(self, tmp_path):
        result = execute_tool(
            "write_file",
            {"path": "out.txt", "content": "test"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
            sandbox=True,
        )
        assert "disabled in sandbox" in result

    def test_write_outside_allowed(self, tmp_path):
        with pytest.raises(ValueError, match="outside allowed"):
            execute_tool(
                "write_file",
                {"path": "/tmp/evil.txt", "content": "bad"},
                working_dir=str(tmp_path),
                allowed_directories=[str(tmp_path)],
            )


class TestListFiles:
    def test_list_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = execute_tool(
            "list_files",
            {"directory": str(tmp_path)},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "a.py" in result
        assert "b.txt" in result

    def test_list_with_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = execute_tool(
            "list_files",
            {"directory": str(tmp_path), "pattern": "*.py"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "a.py" in result
        assert "b.txt" not in result

    def test_list_missing_directory(self, tmp_path):
        result = execute_tool(
            "list_files",
            {"directory": str(tmp_path / "nope")},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "not found" in result

    def test_list_empty_directory(self, tmp_path):
        sub = tmp_path / "empty"
        sub.mkdir()
        result = execute_tool(
            "list_files",
            {"directory": str(sub)},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "empty" in result.lower()


class TestRunCommand:
    def test_run_allowed_command(self, tmp_path):
        result = execute_tool(
            "run_command",
            {"command": "echo hello"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
            command_allowlist=["echo"],
        )
        assert "hello" in result

    def test_run_blocked_command(self, tmp_path):
        result = execute_tool(
            "run_command",
            {"command": "curl http://evil.com"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
            command_allowlist=["echo", "ls"],
        )
        assert "not in the allowlist" in result

    def test_run_sandbox_blocks_writes(self, tmp_path):
        result = execute_tool(
            "run_command",
            {"command": "rm -rf /"},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
            sandbox=True,
            command_allowlist=["rm"],  # even if on allowlist
        )
        assert "blocked in sandbox" in result

    def test_run_no_command(self, tmp_path):
        result = execute_tool(
            "run_command",
            {},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "required" in result

    def test_unknown_tool(self, tmp_path):
        result = execute_tool(
            "unknown_tool",
            {},
            working_dir=str(tmp_path),
            allowed_directories=[str(tmp_path)],
        )
        assert "unknown tool" in result


# ── AgentRuntime tests ───────────────────────────────────────────────────


class TestAgentRuntime:
    def test_single_step_no_tools(self, tmp_path):
        """Model responds without tool calls -- single iteration."""
        engine = _make_engine(["The answer is 42."])
        config = _make_config(tmp_path)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("What is the answer?")
        assert answer == "The answer is 42."
        assert len(steps) == 1
        assert steps[0].tool_calls == []

    def test_tool_call_then_answer(self, tmp_path):
        """Model makes a tool call, then answers."""
        (tmp_path / "data.txt").write_text("important data")

        responses = [
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "data.txt"}}\n</tool_call>',
            "The file contains: important data",
        ]
        engine = _make_engine(responses)
        config = _make_config(tmp_path)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Read data.txt")
        assert "important data" in answer
        assert len(steps) == 2
        assert len(steps[0].tool_calls) == 1
        assert steps[0].tool_calls[0]["name"] == "read_file"
        assert steps[0].tool_results[0].output == "important data"
        assert steps[1].tool_calls == []

    def test_multiple_tool_calls_one_response(self, tmp_path):
        """Model emits multiple tool calls in a single response."""
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        responses = [
            (
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>\n'
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "b.txt"}}\n</tool_call>'
            ),
            "Files contain: aaa and bbb",
        ]
        engine = _make_engine(responses)
        config = _make_config(tmp_path)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Read both files")
        assert len(steps) == 2
        assert len(steps[0].tool_calls) == 2
        assert len(steps[0].tool_results) == 2

    def test_max_iterations_limit(self, tmp_path):
        """Agent stops after max_iterations even if model keeps calling tools."""
        tc = '<tool_call>\n{"name": "list_files", "arguments": {}}\n</tool_call>'
        engine = _make_engine([tc] * 10)
        config = _make_config(tmp_path, max_iterations=3)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Keep listing")
        assert len(steps) == 3

    def test_sandbox_blocks_write_file(self, tmp_path):
        """In sandbox mode, write_file tool is not available."""
        responses = [
            '<tool_call>\n{"name": "write_file", "arguments": {"path": "x.txt", "content": "bad"}}\n</tool_call>',
            "I couldn't write the file.",
        ]
        engine = _make_engine(responses)
        config = _make_config(tmp_path, sandbox=True)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Write a file")
        # The tool call should fail because write_file is not available in sandbox
        assert len(steps[0].tool_results) == 1
        assert steps[0].tool_results[0].is_error
        assert "not available" in steps[0].tool_results[0].output

    def test_enabled_tools_filter(self, tmp_path):
        """Only enabled tools are available."""
        responses = [
            '<tool_call>\n{"name": "run_command", "arguments": {"command": "echo hi"}}\n</tool_call>',
            "ok",
        ]
        engine = _make_engine(responses)
        config = _make_config(tmp_path, enabled_tools=["read_file", "list_files"])
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Run a command")
        assert steps[0].tool_results[0].is_error
        assert "not available" in steps[0].tool_results[0].output

    def test_on_step_callback(self, tmp_path):
        """on_step callback is called for each step."""
        engine = _make_engine(["Hello!"])
        config = _make_config(tmp_path)
        received_steps: list[AgentStep] = []
        runtime = AgentRuntime(
            config=config, engine=engine, on_step=received_steps.append,
        )

        runtime.run("Hi")
        assert len(received_steps) == 1
        assert received_steps[0].assistant_text == "Hello!"

    def test_invalid_tool_arguments(self, tmp_path):
        """Invalid JSON in tool arguments is handled gracefully."""
        responses = [
            '<tool_call>\n{"name": "read_file", "arguments": "not-json"}\n</tool_call>',
            "Hmm, that didn't work.",
        ]
        engine = _make_engine(responses)
        config = _make_config(tmp_path)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Read something")
        assert steps[0].tool_results[0].is_error
        assert "invalid JSON" in steps[0].tool_results[0].output

    def test_path_escape_rejected(self, tmp_path):
        """Tool calls with paths outside allowed directories are rejected."""
        responses = [
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "/etc/passwd"}}\n</tool_call>',
            "Couldn't read it.",
        ]
        engine = _make_engine(responses)
        config = _make_config(tmp_path)
        runtime = AgentRuntime(config=config, engine=engine)

        answer, steps = runtime.run("Read /etc/passwd")
        assert steps[0].tool_results[0].is_error
        assert "outside allowed" in steps[0].tool_results[0].output


# ── Tool definitions structure ───────────────────────────────────────────


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for td in BUILTIN_TOOL_DEFINITIONS:
            assert td["type"] == "function"
            fn = td["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_builtin_tool_names(self):
        assert BUILTIN_TOOL_NAMES == {
            "read_file", "write_file", "list_files", "run_command",
            "find_files", "search_files", "patch_file", "delegate", "check_task",
        }
