"""Continuous batching layer for ppmlx.

Wraps :class:`ppmlx.engine.TextEngine` with an async request queue that
reduces head-of-line blocking when multiple requests arrive concurrently.

MLX does not natively support batched inference across independent requests,
so the ``BatchEngine`` processes requests sequentially under the hood.  The
value is architectural: incoming requests are queued and dispatched
asynchronously, so one slow generation does not block the event loop for
other callers.  When ``batch_mode=False`` (the default), the server uses
:class:`TextEngine` directly with no overhead.

Usage::

    engine = get_batch_engine()         # singleton
    result = await engine.generate(...)   # non-streaming
    async for chunk in engine.stream_generate(...):  # streaming
        ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

log = logging.getLogger("ppmlx.batch")


@dataclass
class _GenerateRequest:
    """Internal representation of a queued generation request."""
    repo_id: str
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    strip_thinking: bool = True
    enable_thinking: bool = True
    tools: list[dict] | None = None
    stream: bool = False
    future: asyncio.Future | None = None
    queue: asyncio.Queue | None = None  # for streaming: token chunks


class BatchEngine:
    """Async batching layer around :class:`ppmlx.engine.TextEngine`.

    Maintains a request queue and a background worker that processes
    requests one at a time (since MLX does not support true cross-request
    batching).  This architecture provides:

    * **Async dispatch** -- callers ``await`` their result without blocking
      the event loop.
    * **Fair ordering** -- FIFO queue ensures no request starves.
    * **Reduced head-of-line blocking** -- the event loop stays responsive
      for health checks, model listing, etc. while generation runs.

    Parameters
    ----------
    max_batch_size : int
        Maximum number of requests to accumulate before forcing a dispatch.
        Currently each request is processed individually, but the parameter
        is here for future true-batch support.
    batch_window_ms : float
        Maximum time (milliseconds) to wait for more requests before
        dispatching.  Set to 0 for immediate dispatch.
    max_loaded : int
        Passed through to the underlying :class:`TextEngine`.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        batch_window_ms: float = 50.0,
        max_loaded: int = 2,
    ):
        self._max_batch_size = max_batch_size
        self._batch_window_ms = batch_window_ms
        self._max_loaded = max_loaded

        self._queue: asyncio.Queue[_GenerateRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._started = False
        self._stopping = False

        # Lazily created TextEngine
        self._engine: Any = None
        self._engine_lock = threading.Lock()

    def _get_engine(self):
        """Return the underlying TextEngine (created on first access)."""
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from ppmlx.engine import TextEngine
                    self._engine = TextEngine(max_loaded=self._max_loaded)
        return self._engine

    def _ensure_started(self) -> None:
        """Start the background worker if not already running."""
        if self._started:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Background coroutine that drains the request queue."""
        log.info("BatchEngine worker started (batch_window=%.0fms, max_batch=%d)",
                 self._batch_window_ms, self._max_batch_size)
        while not self._stopping:
            try:
                # Wait for the first request
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Collect a batch (currently processed one by one)
                batch: list[_GenerateRequest] = [req]
                if self._batch_window_ms > 0:
                    deadline = time.monotonic() + self._batch_window_ms / 1000.0
                    while len(batch) < self._max_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            more = await asyncio.wait_for(
                                self._queue.get(), timeout=remaining
                            )
                            batch.append(more)
                        except asyncio.TimeoutError:
                            break

                log.debug("Processing batch of %d request(s)", len(batch))

                # Process each request sequentially (MLX limitation)
                for request in batch:
                    await self._process_request(request)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("BatchEngine worker error")

        log.info("BatchEngine worker stopped")

    async def _process_request(self, req: _GenerateRequest) -> None:
        """Process a single request, dispatching the result to its future/queue."""
        engine = self._get_engine()
        loop = asyncio.get_running_loop()

        if req.stream:
            await self._process_stream_request(req, engine, loop)
        else:
            await self._process_generate_request(req, engine, loop)

    async def _process_generate_request(
        self, req: _GenerateRequest, engine: Any, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Run a non-streaming generation in a thread and set the future result."""
        future = req.future
        if future is None or future.cancelled():
            return

        def _run():
            return engine.generate(
                req.repo_id,
                req.messages,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                stop=req.stop,
                seed=req.seed,
                repetition_penalty=req.repetition_penalty,
                strip_thinking=req.strip_thinking,
                enable_thinking=req.enable_thinking,
                tools=req.tools,
            )

        try:
            result = await loop.run_in_executor(None, _run)
            if not future.cancelled():
                future.set_result(result)
        except Exception as exc:
            if not future.cancelled():
                future.set_exception(exc)

    async def _process_stream_request(
        self, req: _GenerateRequest, engine: Any, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Run a streaming generation in a thread, pushing chunks to the queue."""
        q = req.queue
        if q is None:
            return

        _SENTINEL = None  # signals end of stream

        def _run():
            try:
                gen = engine.stream_generate(
                    req.repo_id,
                    req.messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    stop=req.stop,
                    seed=req.seed,
                    repetition_penalty=req.repetition_penalty,
                    enable_thinking=req.enable_thinking,
                    strip_thinking=req.strip_thinking,
                    tools=req.tools,
                )
                for chunk in gen:
                    loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
                loop.call_soon_threadsafe(q.put_nowait, ("done", _SENTINEL))
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("error", exc))

        await loop.run_in_executor(None, _run)

    # ── Public API ────────────────────────────────────────────────────

    async def generate(
        self,
        repo_id: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        strip_thinking: bool = True,
        enable_thinking: bool = True,
        tools: list[dict] | None = None,
    ) -> tuple[str, str | None, int, int]:
        """Queue a non-streaming generation request and await the result.

        Returns the same ``(text, reasoning, prompt_tokens, completion_tokens)``
        tuple as :meth:`TextEngine.generate`.
        """
        self._ensure_started()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        req = _GenerateRequest(
            repo_id=repo_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            repetition_penalty=repetition_penalty,
            strip_thinking=strip_thinking,
            enable_thinking=enable_thinking,
            tools=tools,
            stream=False,
            future=future,
        )
        await self._queue.put(req)
        return await future

    async def stream_generate(
        self,
        repo_id: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        strip_thinking: bool = True,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Queue a streaming generation request and yield token chunks.

        Usage::

            async for chunk in engine.stream_generate(...):
                print(chunk, end="")
        """
        self._ensure_started()
        q: asyncio.Queue = asyncio.Queue()

        req = _GenerateRequest(
            repo_id=repo_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            strip_thinking=strip_thinking,
            tools=tools,
            stream=True,
            queue=q,
        )
        await self._queue.put(req)

        while True:
            msg_type, payload = await q.get()
            if msg_type == "chunk":
                yield payload
            elif msg_type == "done":
                break
            elif msg_type == "error":
                raise payload

    def list_loaded(self) -> list[str]:
        """Proxy to underlying TextEngine."""
        return self._get_engine().list_loaded()

    def load(self, repo_id: str):
        """Proxy to underlying TextEngine."""
        return self._get_engine().load(repo_id)

    def unload(self, repo_id: str) -> bool:
        """Proxy to underlying TextEngine."""
        return self._get_engine().unload(repo_id)

    def unload_all(self) -> None:
        """Proxy to underlying TextEngine."""
        return self._get_engine().unload_all()

    async def shutdown(self) -> None:
        """Gracefully stop the background worker."""
        self._stopping = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._started = False
        self._stopping = False

    @property
    def queue_size(self) -> int:
        """Number of requests currently waiting in the queue."""
        return self._queue.qsize()

    @property
    def engine(self):
        """Access the underlying TextEngine (for direct use or testing)."""
        return self._get_engine()


# ── Module-level singleton ────────────────────────────────────────────

_batch_engine: BatchEngine | None = None
_batch_lock = threading.Lock()


def get_batch_engine(
    max_batch_size: int = 8,
    batch_window_ms: float = 50.0,
    max_loaded: int = 2,
) -> BatchEngine:
    """Return the module-level singleton BatchEngine."""
    global _batch_engine
    if _batch_engine is None:
        with _batch_lock:
            if _batch_engine is None:
                _batch_engine = BatchEngine(
                    max_batch_size=max_batch_size,
                    batch_window_ms=batch_window_ms,
                    max_loaded=max_loaded,
                )
    return _batch_engine


def reset_batch_engine() -> None:
    """Reset the singleton (for testing)."""
    global _batch_engine
    _batch_engine = None
