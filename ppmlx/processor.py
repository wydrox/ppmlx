"""Batch document processing with LLM tasks.

Processes directories of documents (summarize, translate, extract entities,
classify, or custom prompt) by calling the local ppmlx API.
"""
from __future__ import annotations

import fnmatch
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx


# ── Supported file extensions ─────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".csv",
    ".html", ".xml", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".rst", ".log", ".sh", ".bash", ".zsh", ".rb", ".go",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".swift",
    ".kt", ".scala", ".r", ".sql", ".dockerfile",
}

# Files to always skip
_SKIP_NAMES: set[str] = {
    ".DS_Store", "Thumbs.db", ".gitignore", ".gitattributes",
    "package-lock.json", "yarn.lock", "poetry.lock", "uv.lock",
}

# Max file size to process (1 MB)
_MAX_FILE_SIZE = 1_048_576


# ── Task definitions ──────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    """A processing task with a system prompt template and output format."""
    name: str
    system_prompt: str
    output_suffix: str  # appended to output filename, e.g. ".summary.md"

    def build_user_message(self, content: str, filename: str) -> str:
        """Build the user message for this task."""
        return f"File: {filename}\n\n{content}"


BUILTIN_TASKS: dict[str, TaskDefinition] = {
    "summarize": TaskDefinition(
        name="summarize",
        system_prompt=(
            "You are a document summarizer. Read the provided document and "
            "produce a clear, concise summary capturing the key points. "
            "Use bullet points for clarity. Keep the summary under 500 words."
        ),
        output_suffix=".summary.md",
    ),
    "translate": TaskDefinition(
        name="translate",
        system_prompt=(
            "You are a translator. Translate the provided document to English. "
            "If the document is already in English, translate it to Spanish. "
            "Preserve formatting and structure."
        ),
        output_suffix=".translated.md",
    ),
    "extract_entities": TaskDefinition(
        name="extract_entities",
        system_prompt=(
            "You are an entity extractor. Read the provided document and extract "
            "all named entities (people, organizations, locations, dates, amounts). "
            "Output as a JSON array of objects with 'entity', 'type', and 'context' fields."
        ),
        output_suffix=".entities.json",
    ),
    "classify": TaskDefinition(
        name="classify",
        system_prompt=(
            "You are a document classifier. Classify the provided document into "
            "one of these categories: technical, business, personal, academic, "
            "creative, legal, medical, or other. Output a JSON object with "
            "'category', 'confidence' (0-1), and 'reasoning' fields."
        ),
        output_suffix=".classification.json",
    ),
}


def make_custom_task(prompt: str) -> TaskDefinition:
    """Create a custom task from a user-provided prompt."""
    return TaskDefinition(
        name="custom",
        system_prompt=prompt,
        output_suffix=".processed.md",
    )


# ── File discovery ────────────────────────────────────────────────────

def _load_gitignore_patterns(directory: Path) -> list[str]:
    """Load .gitignore patterns from a directory (simple glob matching)."""
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text(errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _is_gitignored(path: Path, base_dir: Path, patterns: list[str]) -> bool:
    """Check if a path matches any gitignore pattern (simplified)."""
    rel = str(path.relative_to(base_dir))
    name = path.name
    for pattern in patterns:
        pattern_clean = pattern.rstrip("/")
        if pattern_clean == name:
            return True
        if pattern_clean == rel:
            return True
        # Directory pattern (ends with /)
        if pattern.endswith("/") and path.is_dir() and name == pattern_clean:
            return True
        # Wildcard patterns
        if "*" in pattern_clean:
            if fnmatch.fnmatch(name, pattern_clean):
                return True
            if fnmatch.fnmatch(rel, pattern_clean):
                return True
    return False


def discover_files(
    directory: Path,
    *,
    recursive: bool = True,
    extensions: set[str] | None = None,
    respect_gitignore: bool = True,
) -> list[Path]:
    """Discover processable files in a directory.

    Returns a sorted list of file paths.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    gitignore_patterns = (
        _load_gitignore_patterns(directory) if respect_gitignore else []
    )

    files: list[Path] = []
    iterator: Iterator[Path] = (
        directory.rglob("*") if recursive else directory.iterdir()
    )

    for path in iterator:
        if not path.is_file():
            continue
        if path.name in _SKIP_NAMES:
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in extensions:
            continue
        file_size = path.stat().st_size
        if file_size == 0 or file_size > _MAX_FILE_SIZE:
            continue
        if respect_gitignore and _is_gitignored(path, directory, gitignore_patterns):
            continue
        files.append(path)

    files.sort()
    return files


# ── Checkpoint / resume ───────────────────────────────────────────────

_CHECKPOINT_FILE = ".ppmlx-process-state.json"


@dataclass
class CheckpointState:
    """Tracks processing progress for resume support."""
    task: str
    model: str
    directory: str
    completed: dict[str, str] = field(default_factory=dict)  # file_path -> status
    failed: dict[str, str] = field(default_factory=dict)  # file_path -> error
    started_at: str = ""

    def save(self, directory: Path) -> None:
        """Save checkpoint to the processing directory."""
        path = directory / _CHECKPOINT_FILE
        data = {
            "task": self.task,
            "model": self.model,
            "directory": self.directory,
            "completed": self.completed,
            "failed": self.failed,
            "started_at": self.started_at,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: Path) -> CheckpointState | None:
        """Load checkpoint from the processing directory, or None if not found."""
        path = directory / _CHECKPOINT_FILE
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(
                task=data["task"],
                model=data["model"],
                directory=data["directory"],
                completed=data.get("completed", {}),
                failed=data.get("failed", {}),
                started_at=data.get("started_at", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def is_completed(self, file_path: str) -> bool:
        return file_path in self.completed

    def mark_completed(self, file_path: str) -> None:
        self.completed[file_path] = "ok"
        self.failed.pop(file_path, None)

    def mark_failed(self, file_path: str, error: str) -> None:
        self.failed[file_path] = error

    def cleanup(self, directory: Path) -> None:
        """Remove checkpoint file after successful completion."""
        path = directory / _CHECKPOINT_FILE
        path.unlink(missing_ok=True)


# ── Single-file processing ────────────────────────────────────────────

@dataclass
class ProcessResult:
    """Result of processing a single file."""
    file_path: str
    success: bool
    output_path: str | None = None
    output_text: str | None = None
    error: str | None = None
    tokens_used: int = 0
    duration_secs: float = 0.0


def _read_file_content(path: Path) -> str:
    """Read file content, handling encoding issues."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def process_single_file(
    file_path: Path,
    task: TaskDefinition,
    model: str,
    base_url: str,
    *,
    output_dir: Path | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float = 120.0,
) -> ProcessResult:
    """Process a single file through the LLM API.

    Args:
        file_path: Path to the file to process.
        task: The task definition to apply.
        model: Model name/alias for the API.
        base_url: Base URL for the ppmlx API (e.g. http://localhost:6767).
        output_dir: Where to write output. If None, writes alongside originals.
        temperature: Sampling temperature.
        max_tokens: Max tokens for generation (None = API default).
        timeout: HTTP request timeout in seconds.

    Returns:
        ProcessResult with success/failure info and output.
    """
    start = time.monotonic()

    try:
        content = _read_file_content(file_path)
    except OSError as e:
        return ProcessResult(
            file_path=str(file_path),
            success=False,
            error=f"Failed to read file: {e}",
        )

    messages = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": task.build_user_message(content, file_path.name)},
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return ProcessResult(
            file_path=str(file_path),
            success=False,
            error="Request timed out",
            duration_secs=time.monotonic() - start,
        )
    except httpx.HTTPStatusError as e:
        return ProcessResult(
            file_path=str(file_path),
            success=False,
            error=f"API error {e.response.status_code}: {e.response.text[:200]}",
            duration_secs=time.monotonic() - start,
        )
    except httpx.ConnectError:
        return ProcessResult(
            file_path=str(file_path),
            success=False,
            error=f"Cannot connect to ppmlx server at {base_url}. Is it running?",
            duration_secs=time.monotonic() - start,
        )

    try:
        output_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
    except (KeyError, IndexError) as e:
        return ProcessResult(
            file_path=str(file_path),
            success=False,
            error=f"Unexpected API response: {e}",
            duration_secs=time.monotonic() - start,
        )

    duration = time.monotonic() - start

    # Determine output path
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = file_path.stem + task.output_suffix
        out_path = output_dir / out_name
    else:
        out_path = file_path.parent / (file_path.stem + task.output_suffix)

    return ProcessResult(
        file_path=str(file_path),
        success=True,
        output_path=str(out_path),
        output_text=output_text,
        tokens_used=tokens_used,
        duration_secs=duration,
    )


# ── Batch processing engine ──────────────────────────────────────────

@dataclass
class BatchStats:
    """Aggregate stats for a batch processing run."""
    total_files: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_tokens: int = 0
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        processed = self.completed + self.failed
        if processed == 0:
            return 0.0
        return self.completed / processed


def process_batch(
    files: list[Path],
    task: TaskDefinition,
    model: str,
    base_url: str,
    *,
    output_dir: Path | None = None,
    parallel: int = 1,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float = 120.0,
    checkpoint: CheckpointState | None = None,
    checkpoint_dir: Path | None = None,
    progress_callback: Any | None = None,
    stdout_mode: bool = False,
) -> tuple[list[ProcessResult], BatchStats]:
    """Process a batch of files with optional parallelism and progress tracking.

    Args:
        files: List of file paths to process.
        task: The task definition.
        model: Model name/alias.
        base_url: ppmlx API base URL.
        output_dir: Output directory (None = alongside originals).
        parallel: Number of concurrent workers.
        temperature: Sampling temperature.
        max_tokens: Max tokens per request.
        timeout: Per-request timeout.
        checkpoint: Checkpoint state for resume support.
        checkpoint_dir: Directory for checkpoint file.
        progress_callback: Called with (file_path, result) after each file.
        stdout_mode: If True, collect output text for stdout printing instead of writing files.

    Returns:
        Tuple of (results_list, batch_stats).
    """
    results: list[ProcessResult] = []
    stats = BatchStats(total_files=len(files))

    def _process_one(fp: Path) -> ProcessResult:
        return process_single_file(
            fp, task, model, base_url,
            output_dir=output_dir if not stdout_mode else None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def _handle_result(fp: Path, result: ProcessResult) -> None:
        """Update stats, write output, checkpoint, and notify progress."""
        results.append(result)

        if result.success:
            stats.completed += 1
            stats.total_tokens += result.tokens_used
            if not stdout_mode and result.output_text is not None and result.output_path is not None:
                Path(result.output_path).write_text(result.output_text)
            if checkpoint is not None:
                checkpoint.mark_completed(str(fp))
        else:
            stats.failed += 1
            if checkpoint is not None:
                checkpoint.mark_failed(str(fp), result.error or "unknown")

        stats.total_duration += result.duration_secs

        if checkpoint is not None and checkpoint_dir is not None:
            checkpoint.save(checkpoint_dir)

        if progress_callback is not None:
            progress_callback(fp, result)

    # Filter already-completed files when resuming
    pending_files = files
    if checkpoint is not None:
        pending_files = [
            f for f in files if not checkpoint.is_completed(str(f))
        ]
        stats.skipped = len(files) - len(pending_files)

    if parallel <= 1:
        for fp in pending_files:
            _handle_result(fp, _process_one(fp))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_file = {
                executor.submit(_process_one, fp): fp for fp in pending_files
            }
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ProcessResult(
                        file_path=str(fp),
                        success=False,
                        error=str(e),
                    )
                _handle_result(fp, result)

    return results, stats
