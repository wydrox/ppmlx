"""Built-in agent runtime for executing tool calls emitted by models.

Implements a generate -> parse tool calls -> execute -> feed back loop
with configurable safety controls.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ── Background task tracking for delegate/check_task ────────────────────
_BACKGROUND_TASKS: dict[str, dict[str, Any]] = {}


# ── Safety defaults ──────────────────────────────────────────────────────

DEFAULT_COMMAND_ALLOWLIST: list[str] = [
    "ls", "cat", "head", "tail", "find", "grep", "rg", "wc",
    "pwd", "echo", "date", "whoami", "which", "env", "tree",
    "diff", "sort", "uniq", "tr", "cut", "file", "du", "df",
    "uname", "python", "python3", "pip", "uv", "git",
]


# ── Tool result types ────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of executing a single tool call."""
    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class AgentStep:
    """One iteration of the agent loop."""
    iteration: int
    assistant_text: str
    tool_calls: list[dict]
    tool_results: list[ToolResult]


# ── Built-in tool definitions (OpenAI function calling schema) ───────────

BUILTIN_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. For large files, use start_line/end_line "
                "to read only the needed range and avoid flooding the context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based, inclusive). Omit to start from the beginning.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based, inclusive). Omit to read to the end.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, optionally filtering by a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list (default: working directory).",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py'). Default: '*'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search file contents for a regex pattern (like grep). "
                "Returns matching lines with file paths and line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: working directory).",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter (e.g. '*.py', '*.ts'). Default: all files.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return (default: 50).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "Find files by name pattern recursively. "
                "Returns matching file paths relative to the search directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match file names (e.g. '*.py', 'test_*.py', '**/*.json').",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: working directory).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Replace a specific string in a file. More efficient than rewriting "
                "the whole file with write_file. The old_string must appear exactly "
                "once in the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace (must be unique in the file).",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Delegate a task to an external CLI agent (Claude Code or Codex) "
                "asynchronously. Returns a task_id to check progress with check_task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Which agent to delegate to: 'claude' or 'codex'.",
                        "enum": ["claude", "codex"],
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task description / prompt to send to the agent.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the agent process. Defaults to the current working directory.",
                    },
                },
                "required": ["agent", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_task",
            "description": (
                "Check the status and output of a background task started with delegate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned by delegate.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
]

BUILTIN_TOOL_NAMES: set[str] = {
    t["function"]["name"] for t in BUILTIN_TOOL_DEFINITIONS
}


# ── Path validation ─────────────────────────────────────────────────────

def _resolve_path(path_str: str, working_dir: str) -> Path:
    """Resolve a path relative to the working directory."""
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(working_dir) / p
    return p.resolve()


def _validate_path(
    path_str: str,
    working_dir: str,
    allowed_directories: list[str],
) -> Path:
    """Validate that a path is within allowed directories.

    Raises ValueError if the path escapes the allowed directories.
    """
    resolved = _resolve_path(path_str, working_dir)
    for allowed in allowed_directories:
        allowed_resolved = Path(allowed).resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Path '{resolved}' is outside allowed directories: "
        f"{[str(Path(d).resolve()) for d in allowed_directories]}"
    )


# ── Built-in tool implementations ────────────────────────────────────────

def _tool_read_file(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
    max_read_lines: int = 200,
) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required"
    resolved = _validate_path(path_str, working_dir, allowed_directories)
    if not resolved.exists():
        return f"Error: file not found: {resolved}"
    if not resolved.is_file():
        return f"Error: not a file: {resolved}"
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    total = len(lines)

    start = args.get("start_line")
    end = args.get("end_line")
    if start is not None or end is not None:
        s = max(1, int(start)) if start is not None else 1
        e = min(total, int(end)) if end is not None else total
        if s > total:
            return f"Error: start_line {s} exceeds file length ({total} lines)"
        selected = lines[s - 1 : e]
        numbered = "".join(f"{s + i:>6}\t{ln}" for i, ln in enumerate(selected))
        header = f"[{resolved} lines {s}-{min(e, total)} of {total}]\n"
        return header + numbered

    # Full file — cap at max_read_lines to keep context clean
    if total > max_read_lines:
        selected = lines[:max_read_lines]
        numbered = "".join(f"{i + 1:>6}\t{ln}" for i, ln in enumerate(selected))
        header = f"[{resolved} lines 1-{max_read_lines} of {total} — use start_line/end_line to read more]\n"
        return header + numbered

    return "".join(lines)


def _tool_write_file(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    if sandbox:
        return "Error: write_file is disabled in sandbox mode"
    path_str = args.get("path", "")
    content = args.get("content", "")
    if not path_str:
        return "Error: 'path' argument is required"
    resolved = _validate_path(path_str, working_dir, allowed_directories)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {resolved}"


def _tool_list_files(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    directory = args.get("directory", working_dir)
    pattern = args.get("pattern", "*")
    resolved = _validate_path(directory, working_dir, allowed_directories)
    if not resolved.exists():
        return f"Error: directory not found: {resolved}"
    if not resolved.is_dir():
        return f"Error: not a directory: {resolved}"
    matches = sorted(resolved.glob(pattern))
    # Limit output
    max_entries = 500
    lines = []
    for i, m in enumerate(matches):
        if i >= max_entries:
            lines.append(f"... and {len(matches) - max_entries} more")
            break
        suffix = "/" if m.is_dir() else ""
        try:
            rel = m.relative_to(resolved)
        except ValueError:
            rel = m
        lines.append(f"{rel}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


def _extract_base_command(command: str) -> str:
    """Extract the base command name, skipping env var assignments."""
    for part in command.split():
        if "=" not in part:
            return os.path.basename(part)
    return ""


_SANDBOX_BLOCKED_COMMANDS = frozenset({
    "rm", "rmdir", "mv", "cp", "dd", "mkfs", "chmod", "chown",
})


def _tool_run_command(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
    command_allowlist: list[str] | None = None,
) -> str:
    command = args.get("command", "")
    if not command:
        return "Error: 'command' argument is required"

    base_cmd = _extract_base_command(command)

    if command_allowlist is not None and base_cmd not in command_allowlist:
        return (
            f"Error: command '{base_cmd}' is not in the allowlist. "
            f"Allowed: {', '.join(sorted(command_allowlist))}"
        )

    if sandbox and base_cmd in _SANDBOX_BLOCKED_COMMANDS:
        return f"Error: command '{base_cmd}' is blocked in sandbox mode"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=working_dir,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(output_parts) if output_parts else "(no output)"
        # Truncate
        max_chars = 50_000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [truncated, {len(output)} chars total]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as exc:
        return f"Error executing command: {exc}"


def _tool_search_files(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    """Grep-like search across files."""
    import re as _re

    pattern_str = args.get("pattern", "")
    if not pattern_str:
        return "Error: 'pattern' argument is required"
    directory = args.get("directory", working_dir)
    file_glob = args.get("glob", "")
    max_results = min(int(args.get("max_results", 50)), 200)

    resolved = _validate_path(directory, working_dir, allowed_directories)
    if not resolved.is_dir():
        return f"Error: not a directory: {resolved}"

    try:
        regex = _re.compile(pattern_str)
    except _re.error as exc:
        return f"Error: invalid regex: {exc}"

    glob_pattern = f"**/{file_glob}" if file_glob else "**/*"
    matches: list[str] = []
    for fp in sorted(resolved.glob(glob_pattern)):
        if not fp.is_file():
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        try:
            rel = fp.relative_to(resolved)
        except ValueError:
            rel = fp
        for lineno, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{rel}:{lineno}: {line}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n... (limit {max_results} reached)"
    return "\n".join(matches) if matches else "(no matches)"


def _tool_find_files(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    """Recursive file name search via glob."""
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: 'pattern' argument is required"
    directory = args.get("directory", working_dir)
    resolved = _validate_path(directory, working_dir, allowed_directories)
    if not resolved.is_dir():
        return f"Error: not a directory: {resolved}"

    # Ensure recursive
    if "**" not in pattern:
        pattern = f"**/{pattern}"

    results: list[str] = []
    for fp in sorted(resolved.glob(pattern)):
        try:
            rel = fp.relative_to(resolved)
        except ValueError:
            rel = fp
        suffix = "/" if fp.is_dir() else ""
        results.append(f"{rel}{suffix}")
        if len(results) >= 500:
            results.append("... (limit 500 reached)")
            break
    return "\n".join(results) if results else "(no matches)"


def _tool_patch_file(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    """Replace a unique string in a file."""
    if sandbox:
        return "Error: patch_file is disabled in sandbox mode"
    path_str = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    if not path_str:
        return "Error: 'path' argument is required"
    if not old_string:
        return "Error: 'old_string' argument is required"
    resolved = _validate_path(path_str, working_dir, allowed_directories)
    if not resolved.is_file():
        return f"Error: file not found: {resolved}"
    content = resolved.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_string)
    if count == 0:
        return "Error: old_string not found in file"
    if count > 1:
        return f"Error: old_string appears {count} times — must be unique"
    new_content = content.replace(old_string, new_string, 1)
    resolved.write_text(new_content, encoding="utf-8")
    return f"Successfully patched {resolved}"


def _tool_delegate(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    """Delegate a task to an external CLI agent (claude or codex)."""
    if sandbox:
        return "Error: delegate is disabled in sandbox mode"
    agent = args.get("agent", "")
    prompt = args.get("prompt", "")
    cwd = args.get("working_dir", "") or working_dir
    if not agent:
        return "Error: 'agent' argument is required"
    if not prompt:
        return "Error: 'prompt' argument is required"
    if agent not in ("claude", "codex"):
        return f"Error: agent must be 'claude' or 'codex', got '{agent}'"

    if agent == "claude":
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
    else:
        cmd = ["codex", "-q", prompt]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
    except Exception as exc:
        return f"Error: failed to start {agent}: {exc}"

    task_id = f"task_{len(_BACKGROUND_TASKS)}"
    _BACKGROUND_TASKS[task_id] = {
        "process": process,
        "status": "running",
        "output": "",
        "agent": agent,
        "prompt": prompt,
    }
    return f"Started {task_id} ({agent}). Use check_task to see results."


def _tool_check_task(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    """Check the status of a background task."""
    task_id = args.get("task_id", "")
    if not task_id:
        return "Error: 'task_id' argument is required"
    task = _BACKGROUND_TASKS.get(task_id)
    if task is None:
        return f"Error: unknown task '{task_id}'"

    if task["status"] != "running":
        # Already collected
        return task["output"]

    process = task["process"]
    retcode = process.poll()
    if retcode is None:
        return f"Task {task_id} is still running."

    # Process finished — collect output
    stdout = process.stdout.read() if process.stdout else ""
    stderr = process.stderr.read() if process.stderr else ""
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if retcode != 0:
        parts.append(f"[exit code: {retcode}]")
        task["status"] = "error"
    else:
        task["status"] = "done"
    output = "\n".join(parts) if parts else "(no output)"
    # Truncate to 50000 chars
    if len(output) > 50_000:
        output = output[:50_000] + f"\n... [truncated, {len(output)} chars total]"
    task["output"] = output
    return output


# ── Tool dispatcher ──────────────────────────────────────────────────────

_TOOL_HANDLERS: dict[str, Callable] = {
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "list_files": _tool_list_files,
    "run_command": _tool_run_command,
    "search_files": _tool_search_files,
    "find_files": _tool_find_files,
    "patch_file": _tool_patch_file,
    "delegate": _tool_delegate,
    "check_task": _tool_check_task,
}


def execute_tool(
    name: str,
    arguments: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool = False,
    command_allowlist: list[str] | None = None,
    max_read_lines: int = 200,
    max_output_chars: int = 20_000,
) -> str:
    """Execute a built-in tool by name. Returns the output string.

    All tool output is capped at *max_output_chars* to protect the
    model's context window from oversized results.
    """
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"

    kwargs: dict[str, Any] = {
        "args": arguments,
        "working_dir": working_dir,
        "allowed_directories": allowed_directories,
        "sandbox": sandbox,
    }
    if name == "run_command":
        kwargs["command_allowlist"] = command_allowlist
    if name == "read_file":
        kwargs["max_read_lines"] = max_read_lines
    output = handler(**kwargs)
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + f"\n... [truncated, {len(output)} total chars — use narrower queries]"
    return output


# ── Permission system ───────────────────────────────────────────────────

_TOOL_PERMISSIONS: dict[str, str] = {
    "read_file": "readonly",
    "list_files": "readonly",
    "search_files": "readonly",
    "find_files": "readonly",
    "check_task": "readonly",
    "write_file": "write",
    "patch_file": "write",
    "run_command": "execute",
    "delegate": "execute",
}

_PERMISSION_ORDER: list[str] = ["readonly", "write", "execute", "full"]


# ── Agent Runtime ────────────────────────────────────────────────────────

def _make_tool_call_id(iteration: int, index: int) -> str:
    return f"call_{iteration}_{index}"


VOICE_SYSTEM_PROMPT: str = """\
You are a voice assistant. The user is speaking to you and will hear your reply read aloud.

Response style:
- Be concise and conversational — 2-3 sentences per turn is ideal.
- Use progressive disclosure: give the key point first, then ask if the user wants details.
- Never use markdown, code blocks, tables, bullet lists, or numbered lists — they sound terrible when read aloud.
- Instead of listing items, mention the most important one or two and offer to continue.
- Use natural spoken connectors: "first", "then", "also", "by the way".
- Spell out abbreviations and symbols: say "about 5 seconds" not "~5s".
- When referring to commands or code, describe what to do rather than dictating syntax. If the user explicitly asks for a command, say it slowly and clearly.
- If a task requires a long explanation, break it into a dialogue — do one step, confirm, then continue.
- Match the user's language — if they speak Polish, reply in Polish.\
"""


def _build_system_prompt(
    user_system_prompt: str | None,
    tools: list[dict],
) -> str:
    """Build the system prompt that instructs the model to use tools."""
    parts: list[str] = []

    if user_system_prompt:
        parts.append(user_system_prompt)

    parts.append(
        "You are an agent with access to tools. "
        "When you need to use a tool, emit a <tool_call> block with JSON inside:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "</tool_call>\n\n"
        "You may call multiple tools in one response by emitting multiple "
        "<tool_call> blocks. After each tool call round, you will receive the "
        "results and can decide what to do next.\n\n"
        "When you have completed the task, respond with your final answer "
        "without any tool calls."
    )

    tool_descriptions = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        tool_descriptions.append(
            f"- **{name}**: {desc}\n"
            f"  Parameters: {json.dumps(params)}"
        )

    if tool_descriptions:
        parts.append("Available tools:\n" + "\n".join(tool_descriptions))

    return "\n\n".join(parts)


@dataclass
class AgentConfig:
    """Configuration for the agent runtime."""
    model: str
    max_iterations: int = 10
    sandbox: bool = False
    allowed_directories: list[str] = field(default_factory=list)
    command_allowlist: list[str] | None = field(default=None)
    working_dir: str = field(default_factory=os.getcwd)
    enabled_tools: list[str] | None = None  # None = all built-in tools
    temperature: float = 0.7
    max_tokens: int | None = None
    max_read_lines: int = 200
    max_output_chars: int = 20_000
    permission_level: str = "full"  # "readonly", "write", "execute", "full"

    def __post_init__(self) -> None:
        if not self.allowed_directories:
            self.allowed_directories = [self.working_dir]
        if self.command_allowlist is None and not self.sandbox:
            self.command_allowlist = list(DEFAULT_COMMAND_ALLOWLIST)
        elif self.command_allowlist is None and self.sandbox:
            # In sandbox, only allow read-only commands
            self.command_allowlist = [
                "ls", "cat", "head", "tail", "find", "grep", "rg", "wc",
                "pwd", "echo", "date", "whoami", "which", "tree",
                "diff", "sort", "uniq", "file", "du", "df", "uname",
            ]


class AgentRuntime:
    """Agent runtime that executes tool calls emitted by models.

    Implements a generate -> parse tool calls -> execute -> feed back loop.
    """

    def __init__(
        self,
        config: AgentConfig,
        engine: Any | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
        confirm_callback: Callable[[str], bool] | None = None,
    ):
        self.config = config
        self._engine = engine
        self._on_step = on_step
        self._confirm_callback = confirm_callback
        self._tool_defs = self._get_tool_definitions()
        self._enabled_names: set[str] = {
            t["function"]["name"] for t in self._tool_defs
        }

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        from ppmlx.engine import get_engine
        return get_engine()

    def _get_tool_definitions(self) -> list[dict]:
        """Get the tool definitions based on enabled_tools config."""
        if self.config.enabled_tools is None:
            defs = list(BUILTIN_TOOL_DEFINITIONS)
        else:
            enabled = set(self.config.enabled_tools)
            defs = [
                t for t in BUILTIN_TOOL_DEFINITIONS
                if t["function"]["name"] in enabled
            ]
        # In sandbox mode, exclude write tools
        if self.config.sandbox:
            _write_tools = {"write_file", "patch_file", "delegate"}
            defs = [
                t for t in defs
                if t["function"]["name"] not in _write_tools
            ]
        return defs

    def _parse_tool_calls(self, text: str) -> tuple[str, list[dict]]:
        """Parse tool calls from model output. Reuses server logic."""
        from ppmlx.server import _parse_tool_calls
        return _parse_tool_calls(text)

    def _execute_tool_call(
        self, name: str, arguments_str: str, call_id: str,
    ) -> ToolResult:
        """Execute a single tool call and return the result."""
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except (json.JSONDecodeError, TypeError):
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output=f"Error: invalid JSON arguments: {arguments_str}",
                is_error=True,
            )

        if name not in self._enabled_names:
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output=f"Error: tool '{name}' is not available",
                is_error=True,
            )

        # ── Permission check ────────────────────────────────────────
        required = _TOOL_PERMISSIONS.get(name, "full")
        current = self.config.permission_level
        req_idx = _PERMISSION_ORDER.index(required) if required in _PERMISSION_ORDER else len(_PERMISSION_ORDER)
        cur_idx = _PERMISSION_ORDER.index(current) if current in _PERMISSION_ORDER else -1
        if cur_idx < req_idx:
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output=f"Permission denied: '{name}' requires '{required}' level (current: '{current}')",
                is_error=True,
            )

        # ── Confirmation for dangerous tools when permission_level is "full" ─
        if self._confirm_callback and required in ("write", "execute"):
            desc = f"Agent wants to call '{name}'"
            if isinstance(arguments, dict):
                # Provide a short summary of arguments
                if name == "run_command":
                    desc += f": {arguments.get('command', '')}"
                elif name in ("write_file", "patch_file"):
                    desc += f" on {arguments.get('path', '')}"
            if not self._confirm_callback(desc):
                return ToolResult(
                    tool_call_id=call_id,
                    name=name,
                    output="User denied tool execution",
                    is_error=True,
                )

        try:
            output = execute_tool(
                name=name,
                arguments=arguments,
                working_dir=self.config.working_dir,
                allowed_directories=self.config.allowed_directories,
                sandbox=self.config.sandbox,
                command_allowlist=self.config.command_allowlist,
                max_read_lines=self.config.max_read_lines,
                max_output_chars=self.config.max_output_chars,
            )
            is_error = output.startswith("Error:")
        except ValueError as exc:
            output = str(exc)
            is_error = True
        except Exception as exc:
            output = f"Error: {exc}"
            is_error = True

        return ToolResult(
            tool_call_id=call_id,
            name=name,
            output=output,
            is_error=is_error,
        )

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, list[AgentStep]]:
        """Run the agent loop.

        Returns (final_answer, steps).
        """
        engine = self._get_engine()
        full_system = _build_system_prompt(system_prompt, self._tool_defs)

        messages: list[dict] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        steps: list[AgentStep] = []

        for iteration in range(self.config.max_iterations):
            # Generate response
            result = engine.generate(
                repo_id=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                strip_thinking=True,
            )
            text = result.text

            # Parse tool calls
            remaining_text, tool_calls = self._parse_tool_calls(text)

            if not tool_calls:
                # No tool calls -- this is the final answer
                step = AgentStep(
                    iteration=iteration,
                    assistant_text=remaining_text or text,
                    tool_calls=[],
                    tool_results=[],
                )
                steps.append(step)
                if self._on_step:
                    self._on_step(step)
                return remaining_text or text, steps

            # Execute tool calls
            tool_results: list[ToolResult] = []
            for i, tc in enumerate(tool_calls):
                call_id = _make_tool_call_id(iteration, i)
                result = self._execute_tool_call(
                    name=tc["name"],
                    arguments_str=tc["arguments"],
                    call_id=call_id,
                )
                tool_results.append(result)

            step = AgentStep(
                iteration=iteration,
                assistant_text=remaining_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            steps.append(step)
            if self._on_step:
                self._on_step(step)

            # Build assistant message with tool call blocks (for context)
            messages.append({"role": "assistant", "content": text})

            # Feed tool results back as tool messages
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "name": tr.name,
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.output,
                })

        # Hit max iterations -- return whatever we have
        final_text = steps[-1].assistant_text if steps else "Max iterations reached without a final answer."
        return final_text, steps
