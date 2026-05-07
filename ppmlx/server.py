from __future__ import annotations
import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

log = logging.getLogger("ppmlx.server")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    log.addHandler(_h)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ppmlx import __version__

_start_time = time.time()


_DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_MAX_TOKENS_CAP = 32_768
_MAX_EMBED_INPUTS = 256


_cached_server_config = None
_server_config_loaded = False


def _load_server_config():
    """Load the server config section, returning None on failure.

    Caches the result so multiple callers at import time share one load.
    """
    global _cached_server_config, _server_config_loaded
    if _server_config_loaded:
        return _cached_server_config
    _server_config_loaded = True
    try:
        from ppmlx.config import load_config
        cfg = load_config()
        # Guard against mocked config objects in tests
        if hasattr(cfg.server, "host") and isinstance(cfg.server.host, str):
            _cached_server_config = cfg.server
    except Exception:
        pass
    return _cached_server_config


def _get_max_request_body_bytes() -> int:
    """Load max request body size from config (default 10 MB)."""
    srv = _load_server_config()
    if srv is not None:
        val = getattr(srv, "max_request_body_mb", None)
        if isinstance(val, int):
            return val * 1024 * 1024
    return _DEFAULT_MAX_BODY_BYTES


def _get_max_tokens_cap() -> int:
    """Load max_tokens cap from config (default 32768)."""
    srv = _load_server_config()
    if srv is not None:
        val = getattr(srv, "max_tokens_cap", None)
        if isinstance(val, int):
            return val
    return _DEFAULT_MAX_TOKENS_CAP


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies that exceed a configurable size limit."""

    def __init__(self, app, max_bytes: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: StarletteRequest, call_next
    ) -> StarletteResponse:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"error": {"message": "Request body too large", "type": "invalid_request_error"}},
                    )
            except (ValueError, TypeError):
                pass
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    try:
        from ppmlx.db import get_db
        db = get_db()
        db.init()
        app.state.db = db
    except ImportError:
        app.state.db = None

    try:
        from ppmlx.config import load_config
        cfg = load_config()
        interval = cfg.logging.snapshot_interval_seconds
    except ImportError:
        interval = 60

    snapshot_task = asyncio.create_task(_snapshot_loop(interval))

    yield

    snapshot_task.cancel()
    try:
        await snapshot_task
    except asyncio.CancelledError:
        pass

    try:
        if app.state.db:
            app.state.db.flush()
            app.state.db.close()
    except Exception:
        pass


app = FastAPI(
    title="ppmlx",
    version=__version__,
    description="OpenAI-compatible LLM API for Apple Silicon via MLX",
    docs_url="/docs", redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS: default to localhost-only; configurable via config.toml.
# Set [server] cors_origins = ["*"] to restore permissive behavior.
_DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def _get_cors_config() -> tuple[list[str], str | None]:
    """Return (origins_list, origin_regex) from config.

    If the user sets cors_origins = ["*"], we pass that directly and skip regex.
    Otherwise we use a regex that matches localhost on any port.
    """
    srv = _load_server_config()
    if srv is not None:
        origins = getattr(srv, "cors_origins", None)
        if isinstance(origins, list) and origins:
            return origins, None
    return [], _DEFAULT_CORS_ORIGIN_REGEX


_cors_origins, _cors_regex = _get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or [],
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestSizeLimitMiddleware, max_bytes=_get_max_request_body_bytes())

_MAX_TOKENS_CAP = _get_max_tokens_cap()


def _clamp_max_tokens(requested: int | None) -> int | None:
    """Clamp client-requested max_tokens to the server-side cap."""
    if requested is None:
        return None
    return min(requested, _MAX_TOKENS_CAP)


async def _snapshot_loop(interval_seconds: int) -> None:
    """Periodically log system snapshots to the database."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            from ppmlx.memory import get_system_ram_gb
            from ppmlx.db import get_db
            from ppmlx.engine import get_engine
            ram_total = get_system_ram_gb()
            loaded = get_engine().list_loaded()
            uptime = int(time.time() - _start_time)
            get_db().log_system_snapshot(
                memory_total_gb=ram_total,
                memory_used_gb=0.0,  # hard to get used memory without psutil
                loaded_models=loaded,
                uptime_seconds=uptime,
            )
        except Exception:
            pass


def _route_engine(repo_id: str, has_images: bool) -> str:
    """Determine which engine to use: 'text', 'vision', or 'embed'."""
    try:
        from ppmlx.models import is_vision_model, is_embed_model
        if is_embed_model(repo_id):
            return "embed"
        if has_images and is_vision_model(repo_id):
            return "vision"
    except ImportError:
        pass
    return "text"


def _has_images(messages: list[dict]) -> bool:
    """Check if any message contains an image_url content part."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _log_request(_, **kwargs) -> None:
    """Log a request to the DB (best-effort)."""
    try:
        from ppmlx.db import get_db
        get_db().log_request(**kwargs)
    except Exception:
        pass


def _track_usage(event: str, data: dict | None = None, *, context: str = "server") -> None:
    try:
        from ppmlx.analytics import track_async

        track_async(event, data, context=context)
    except Exception:
        pass


def _merge_system_messages(messages: list[dict]) -> list[dict]:
    """Merge all system messages into a single one at the start.

    Many models (e.g. Qwen) only accept one system message and require it
    to be the first message.  When clients like Codex send both an
    ``instructions`` system prompt *and* developer-role messages (also
    mapped to system), we merge them so the model template doesn't error.
    """
    system_parts: list[str] = []
    other: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            if c:
                system_parts.append(c)
        else:
            other.append(msg)
    if not system_parts:
        return messages
    merged = [{"role": "system", "content": "\n\n".join(system_parts)}]
    merged.extend(other)
    return merged


def _inject_tool_awareness(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Prepend a short system hint so the model knows which tools it has.

    Without this, thinking models (Qwen3, GLM-4) hallucinate tool usage
    and burn thousands of tokens reasoning about tools that don't exist.
    With the hint they answer immediately: "I don't have that tool."
    """
    try:
        from ppmlx.config import load_config
        cfg = load_config()
        mode = getattr(getattr(cfg, "tool_awareness", None), "mode", "no_tools_only")
    except Exception:
        mode = "no_tools_only"

    mode = str(mode).strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return messages
    if mode in {"1", "true", "yes"}:
        mode = "all"
    elif mode not in {"all", "no_tools_only"}:
        mode = "no_tools_only"

    if tools and mode == "no_tools_only":
        return messages

    if tools:
        names = sorted({
            t.get("function", {}).get("name") or t.get("name", "")
            for t in tools
        } - {""})
        hint = (
            "You have access ONLY to these tools: "
            + ", ".join(names)
            + ". If the user asks you to use a tool not in this list, "
            "tell them it is not available. Do not hallucinate tool calls."
        )
    else:
        hint = (
            "You do not have access to any external tools "
            "(no web search, no file access, no code execution, etc.). "
            "If the user asks you to use a tool or search the web, "
            "briefly inform them that this capability is not available. "
            "Do not simulate or hallucinate tool usage."
        )

    # Append to existing system message or create one
    if messages and messages[0].get("role") == "system":
        messages = list(messages)
        messages[0] = {
            **messages[0],
            "content": messages[0].get("content", "") + "\n\n" + hint,
        }
    else:
        messages = [{"role": "system", "content": hint}] + list(messages)
    return messages


# ── Tool-call parsing ───────────────────────────────────────────────────

_TOOL_CALL_FALLBACK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TC_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _parse_tool_calls(
    text: str, tokenizer: object = None, tools: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Extract tool-call blocks from model output.

    When *tokenizer* has ``has_tool_calling`` (set by mlx_lm), the
    model-specific parser is used — covers Qwen, GLM-4.7, Mistral,
    Llama, Phi, DeepSeek, Gemma, and others automatically.

    Falls back to basic ``<tool_call>`` JSON regex otherwise.

    Returns *(remaining_text, tool_calls)* where each tool call is
    ``{"name": "...", "arguments": "..."}`` (arguments as a JSON string).
    """
    if tokenizer is not None and getattr(tokenizer, "has_tool_calling", False):
        return _parse_tool_calls_mlx(text, tokenizer, tools)
    return _parse_tool_calls_fallback(text)


def _parse_tool_calls_mlx(
    text: str, tokenizer: object, tools: list[dict] | None,
) -> tuple[str, list[dict]]:
    """Parse tool calls using mlx_lm's model-specific parser."""
    start_tag = tokenizer.tool_call_start
    end_tag = tokenizer.tool_call_end
    parser = tokenizer.tool_parser

    calls: list[dict] = []
    remaining_parts: list[str] = []
    rest = text

    while True:
        s_idx = rest.find(start_tag)
        if s_idx == -1:
            remaining_parts.append(rest)
            break
        remaining_parts.append(rest[:s_idx])
        after_start = rest[s_idx + len(start_tag):]

        if end_tag:
            e_idx = after_start.find(end_tag)
            if e_idx == -1:
                body = after_start
                rest = ""
            else:
                body = after_start[:e_idx]
                rest = after_start[e_idx + len(end_tag):]
        else:
            body = after_start
            rest = ""

        try:
            result = parser(body.strip(), tools=tools)
            if result:
                name = result.get("name", "").strip()
                args = result.get("arguments", {})
                if isinstance(args, dict):
                    args = json.dumps(args)
                elif not isinstance(args, str):
                    args = json.dumps(args)
                calls.append({"name": name, "arguments": args})
        except Exception:
            remaining_parts.append(start_tag + body + (end_tag or ""))

    return "".join(remaining_parts).strip(), calls


def _parse_tool_calls_fallback(text: str) -> tuple[str, list[dict]]:
    """Fallback: extract ``<tool_call>`` JSON blocks (Qwen/Hermes style)."""
    calls: list[dict] = []
    for m in _TOOL_CALL_FALLBACK_RE.finditer(text):
        body = m.group(1).strip()
        jm = _TC_JSON_RE.search(body)
        if jm:
            try:
                data = json.loads(jm.group(0))
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if isinstance(args, dict):
                    args = json.dumps(args)
                elif not isinstance(args, str):
                    args = json.dumps(args)
                calls.append({"name": name, "arguments": args})
            except (json.JSONDecodeError, ValueError):
                pass
    remaining = _TOOL_CALL_FALLBACK_RE.sub("", text).strip()
    return remaining, calls


# Core tools that Codex/Claude Code always need — everything else is optional.
# Matched case-insensitively in _limit_tools.
_CORE_TOOL_NAMES = {
    # Codex tools
    "exec_command", "apply_patch", "write_stdin", "update_plan",
    "request_user_input", "view_image",
    # Anthropic / Claude Code tools
    "bash", "read", "edit", "write", "computer",
    "glob", "grep", "agent", "askuserquestion",
    "notebookedit", "webfetch", "websearch",
}

# Maximum estimated tokens for all tools combined.  Large tool lists
# (Codex sends 66) cause extremely slow prefill on local models
# (e.g. 50s+ for ~20k tool tokens on a 9b model), which makes
# clients timeout.  Default 12000 keeps all tools for most clients.
# Set to 0 in config to disable limiting.
_MAX_TOOLS_TOKENS = 12000


def _get_max_tools_tokens() -> int:
    try:
        from ppmlx.config import load_config
        cfg = load_config()
        return cfg.server.max_tools_tokens
    except Exception:
        return _MAX_TOOLS_TOKENS


def _limit_tools(tools: list[dict] | None) -> list[dict] | None:
    """Trim the tools list to keep prompt prefill fast on local models.

    Prioritises core coding tools and drops MCP / agent tools first.
    """
    if not tools:
        return tools
    max_tokens = _get_max_tools_tokens()
    if max_tokens <= 0:
        return tools  # limiting disabled
    # Estimate tokens per tool (~6 chars per token for structured JSON)
    total = sum(len(json.dumps(t)) for t in tools) // 6
    if total <= max_tokens:
        return tools

    # Filter out non-function tools (e.g. web_search with name=None)
    # and split into core vs non-core (case-insensitive matching)
    core = []
    extra = []
    for t in tools:
        name = t.get("name") or t.get("function", {}).get("name", "")
        if not name:
            continue  # skip tools without a name
        if name.lower() in _CORE_TOOL_NAMES:
            core.append(t)
        else:
            extra.append(t)

    # Start with core, add extras until budget reached
    result = list(core)
    budget = max_tokens - sum(len(json.dumps(t)) for t in result) // 6
    for t in extra:
        cost = len(json.dumps(t)) // 6
        if cost <= budget:
            result.append(t)
            budget -= cost
        if budget <= 0:
            break

    log.info("_limit_tools: %d → %d tools (est %d → %d tokens)",
             len(tools), len(result), total,
             sum(len(json.dumps(t)) for t in result) // 4)
    return result


def _normalize_tool_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI tool_calls message format to plain-text format.

    Qwen's chat template doesn't understand OpenAI's ``tool_calls`` list
    structure on assistant messages.  It expects the tool call to appear as
    ``<tool_call>`` JSON blocks in the content string.  Similarly, ``tool``
    role messages are not understood — they must be re-wrapped.

    This function converts:
    - assistant messages with ``tool_calls`` → content with ``<tool_call>`` blocks
    - ``tool`` role messages → keep as-is (Qwen template maps them to
      ``<tool_response>`` via the ``name`` field)
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        tc_list = msg.get("tool_calls")

        if role == "assistant" and tc_list:
            # Convert tool_calls list to <tool_call> text blocks
            parts: list[str] = []
            content = msg.get("content") or ""
            if content:
                parts.append(content)
            for tc in tc_list:
                fn = tc.get("function", tc)
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args_obj = args
                else:
                    args_obj = args
                parts.append(
                    f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args_obj)}}}\n</tool_call>'
                )
            out.append({"role": "assistant", "content": "\n".join(parts)})
        else:
            out.append(msg)
    return out


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok"}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Estimate token count for a messages request (used by Claude Code)."""
    body = await request.json()
    messages = body.get("messages", [])
    system = body.get("system", "")
    tools = body.get("tools", [])
    # Rough estimate: ~4 chars per token
    total_chars = len(json.dumps(messages)) + len(json.dumps(system)) + len(json.dumps(tools))
    input_tokens = total_chars // 4
    return {"input_tokens": input_tokens}


@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    try:
        from ppmlx.memory import get_system_ram_gb
        ram_gb = get_system_ram_gb()
    except Exception:
        ram_gb = 0.0

    registry_info = {}
    try:
        from ppmlx.registry import registry_meta
        registry_info = registry_meta()
    except Exception:
        pass

    return {
        "status": "ok",
        "version": __version__,
        "loaded_models": loaded,
        "uptime_seconds": int(time.time() - _start_time),
        "system": {
            "memory_total_gb": round(ram_gb, 1),
        },
        "registry": registry_info,
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint — returns JSON stats from the DB."""
    try:
        from ppmlx.db import get_db
        stats = get_db().get_stats()
    except Exception:
        stats = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}

    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    return {**stats, "loaded_models": loaded}


def _model_metadata(model_id: str) -> dict:
    """Build rich model metadata for Codex / OpenAI-compatible clients."""
    meta = {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ppmlx",
        "slug": model_id,
    }
    # Try to pull extra info from registry
    try:
        from ppmlx.registry import registry_lookup
        entry = registry_lookup(model_id)
        if entry:
            meta["owned_by"] = entry.get("lab", "ppmlx")
    except Exception:
        entry = None
    return meta


@app.get("/v1/models")
async def list_models():
    """List available models (local + aliases)."""
    try:
        from ppmlx.models import list_local_models, all_aliases
        local = list_local_models()
        aliases = all_aliases()

        data = []
        seen = set()

        for m in local:
            mid = m.get("alias") or m.get("repo_id", "unknown")
            if mid not in seen:
                seen.add(mid)
                data.append(_model_metadata(mid))

        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                data.append(_model_metadata(alias))

        return {"object": "list", "data": data, "models": data}
    except ImportError:
        return {"object": "list", "data": [], "models": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()
    from ppmlx.schema import ChatCompletionRequest as _CCReq; _CCReq.model_validate(body)

    model_name = body.get("model", "")
    messages = body.get("messages", [])
    # Normalize messages for model compatibility
    tools = _limit_tools(body.get("tools") or None)
    for msg in messages:
        if msg.get("role") == "developer":

            msg["role"] = "system"
        # Ensure content is never None (breaks some chat templates)
        if msg.get("content") is None and not msg.get("tool_calls"):
            msg["content"] = ""
    if tools:
        messages = _normalize_tool_messages(messages)
    messages = _merge_system_messages(messages)
    messages = _inject_tool_awareness(messages, tools)
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = _clamp_max_tokens(body.get("max_tokens"))
    stop = body.get("stop")
    seed = body.get("seed")
    repetition_penalty = body.get("repetition_penalty")
    think = body.get("think")
    reasoning_budget = body.get("reasoning_budget")

    # Load config for thinking defaults
    try:
        from ppmlx.config import load_config
        _cfg = load_config()
    except Exception:
        _cfg = None

    # Map OpenAI reasoning_effort ("low"/"medium"/"high") to reasoning_budget
    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort and reasoning_budget is None and _cfg:
        reasoning_budget = _cfg.thinking.effort_to_budget(str(reasoning_effort))
    if reasoning_effort and think is None:
        think = str(reasoning_effort).lower() != "none"

    # Apply config defaults when client doesn't specify
    if _cfg:
        if reasoning_budget is None and _cfg.thinking.default_reasoning_budget > 0:
            reasoning_budget = _cfg.thinking.default_reasoning_budget
        if think is None and not _cfg.thinking.enabled:
            think = False

    _track_usage(
        "api_chat_completions",
        {
            "stream": stream,
            "tools": bool(tools),
            "messages_count": len(messages),
        },
    )

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    log.info(
        "POST /v1/chat/completions model=%s think=%s budget=%s effort=%s stream=%s tools=%d",
        repo_id, think, reasoning_budget, reasoning_effort, stream,
        len(tools) if tools else 0,
    )

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    request_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    start_ts = time.time()
    created = int(start_ts)

    if stream:
        return _stream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts, tools,
            think=think, reasoning_budget=reasoning_budget,
        )
    else:
        return await _nonstream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts, tools,
            think=think, reasoning_budget=reasoning_budget,
        )


def _stream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts, tools=None,
    think=None, reasoning_budget=None,
):
    """Return streaming SSE response."""
    from fastapi.responses import StreamingResponse

    def _get_tool_call_tags(engine, repo_id):
        """Get model-specific tool call start/end tags for stream filtering."""
        try:
            tokenizer = engine.get_tokenizer(repo_id)
            if getattr(tokenizer, "has_tool_calling", False):
                return tokenizer.tool_call_start, tokenizer.tool_call_end or ""
        except Exception:
            pass
        return "<tool_call>", "</tool_call>"

    def _make_chunk_sse(delta):
        """Build an SSE line for a chat.completion.chunk."""
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        return f"data: {json.dumps(data)}\n\n"

    async def event_generator():
        first_token_ts = None
        is_first_chunk = [True]  # mutable so nested helpers can update
        full_text = ""
        thinking_duration_ms = None
        reasoning_chars_count = 0

        def _delta(key, text):
            """Build a delta dict, adding role on the first chunk."""
            d = {key: text}
            if is_first_chunk[0]:
                d["role"] = "assistant"
                is_first_chunk[0] = False
            return d

        try:
            if engine_type == "text":
                from ppmlx.engine import get_engine
                engine = get_engine()

                # Determine whether to enable thinking
                if think is not None:
                    enable_thinking = think
                elif tools and not reasoning_budget:
                    enable_thinking = False
                else:
                    enable_thinking = True

                # When thinking is enabled and no tools, we parse
                # <think> tags ourselves; otherwise let engine strip them.
                strip_thinking = not enable_thinking or bool(tools)

                extra_kwargs = {}
                if reasoning_budget is not None:
                    extra_kwargs["reasoning_budget"] = reasoning_budget

                gen = engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    strip_thinking=strip_thinking,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **extra_kwargs,
                )

                # When tools are provided, buffer output and filter tool call
                # markup so it doesn't leak to the client as content text.
                if tools:
                    tc_start, tc_end = _get_tool_call_tags(engine, repo_id)
                    buf = ""
                    inside_tc = False

                    async for chunk in _async_iter_sync_gen(gen):
                        if not chunk:
                            yield ": keepalive\n\n"
                            continue
                        full_text += chunk
                        if first_token_ts is None:
                            first_token_ts = time.time()
                        buf += chunk

                        while buf:
                            if inside_tc:
                                if tc_end:
                                    end_idx = buf.find(tc_end)
                                    if end_idx != -1:
                                        buf = buf[end_idx + len(tc_end):]
                                        inside_tc = False
                                        continue
                                    # Partial end tag — keep buffering
                                    partial = False
                                    for i in range(1, min(len(tc_end), len(buf)) + 1):
                                        if buf.endswith(tc_end[:i]):
                                            partial = True
                                            break
                                    if partial:
                                        break  # wait for more data
                                    buf = ""  # no end tag match, discard tool call content
                                else:
                                    buf = ""  # no end tag defined, consume rest
                                break
                            else:
                                start_idx = buf.find(tc_start)
                                if start_idx != -1:
                                    # Yield text before tool call tag
                                    safe = buf[:start_idx]
                                    if safe:
                                        yield _make_chunk_sse(_delta("content", safe))
                                    buf = buf[start_idx + len(tc_start):]
                                    inside_tc = True
                                    continue
                                # Check for partial start tag at end of buffer
                                keep = 0
                                for i in range(1, min(len(tc_start), len(buf)) + 1):
                                    if buf.endswith(tc_start[:i]):
                                        keep = i
                                if keep:
                                    safe = buf[:-keep]
                                    buf = buf[-keep:]
                                else:
                                    safe = buf
                                    buf = ""
                                if safe:
                                    yield _make_chunk_sse(_delta("content", safe))
                                break

                    # Flush remaining buffer (only if outside tool call)
                    if buf and not inside_tc:
                        yield _make_chunk_sse(_delta("content", buf))
                elif enable_thinking and not tools:
                    # No tools, thinking enabled — parse <think> tags
                    # and emit reasoning vs content deltas.
                    inside_think = False
                    buf = ""
                    thinking_start_ts = None
                    budget_exceeded = False
                    has_content = False  # Track if any content was emitted

                    # Detect template-injected thinking (Qwen3, DeepSeek-R1)
                    try:
                        lm = engine._get_or_load(repo_id)
                        prompt = engine._apply_chat_template(lm, messages, enable_thinking=True)
                        if re.search(r"<think>\s*$", prompt):
                            inside_think = True
                            thinking_start_ts = time.time()
                    except Exception:
                        pass

                    # Reasoning budget: chars threshold (rough 4 chars/token estimate)
                    _budget_chars = reasoning_budget * 4 if reasoning_budget else 0

                    async for chunk in _async_iter_sync_gen(gen):
                        if not chunk:
                            yield ": keepalive\n\n"
                            continue
                        full_text += chunk
                        if first_token_ts is None:
                            first_token_ts = time.time()
                        buf += chunk

                        while buf:
                            if inside_think:
                                # Budget exceeded — force end of thinking,
                                # emit remaining buffer as content
                                if _budget_chars and reasoning_chars_count >= _budget_chars:
                                    inside_think = False
                                    budget_exceeded = True
                                    if thinking_start_ts:
                                        thinking_duration_ms = (time.time() - thinking_start_ts) * 1000
                                    # Strip any leftover think tags from buffer
                                    buf = buf.replace("</think>", "").replace("<think>", "")
                                    if buf:
                                        has_content = True
                                        yield _make_chunk_sse(_delta("content", buf))
                                    buf = ""
                                    break
                                # Strip leading <think> tag (engine may emit
                                # it even though template already injected one)
                                if buf.startswith("<think>"):
                                    buf = buf[len("<think>"):]
                                    continue
                                end_idx = buf.find("</think>")
                                if end_idx != -1:
                                    think_chunk = buf[:end_idx]
                                    if think_chunk:
                                        reasoning_chars_count += len(think_chunk)
                                        yield _make_chunk_sse(_delta("reasoning", think_chunk))
                                    buf = buf[end_idx + len("</think>"):]
                                    inside_think = False
                                    if thinking_start_ts:
                                        thinking_duration_ms = (time.time() - thinking_start_ts) * 1000
                                    continue
                                # Check for partial </think> tag at end
                                partial_len = 0
                                for i in range(1, len("</think>")):
                                    if buf.endswith("</think>"[:i]):
                                        partial_len = i
                                        break
                                safe = buf[:len(buf) - partial_len] if partial_len else buf
                                buf = buf[len(safe):] if partial_len else ""
                                if safe:
                                    reasoning_chars_count += len(safe)
                                    yield _make_chunk_sse(_delta("reasoning", safe))
                                break
                            else:
                                if budget_exceeded:
                                    # After budget exceeded, strip any think
                                    # tags and emit everything as content
                                    buf = buf.replace("<think>", "").replace("</think>", "")
                                    if buf:
                                        has_content = True
                                        yield _make_chunk_sse(_delta("content", buf))
                                    buf = ""
                                    break
                                start_idx = buf.find("<think>")
                                if start_idx != -1:
                                    before = buf[:start_idx]
                                    if before:
                                        has_content = True
                                        yield _make_chunk_sse(_delta("content", before))
                                    buf = buf[start_idx + len("<think>"):]
                                    inside_think = True
                                    if not thinking_start_ts:
                                        thinking_start_ts = time.time()
                                    continue
                                # Check for partial <think> tag at end
                                keep = 0
                                for i in range(1, min(len("<think>"), len(buf)) + 1):
                                    if buf.endswith("<think>"[:i]):
                                        keep = i
                                if keep:
                                    safe = buf[:-keep]
                                    buf = buf[-keep:]
                                else:
                                    safe = buf
                                    buf = ""
                                if safe:
                                    has_content = True
                                    yield _make_chunk_sse(_delta("content", safe))
                                break

                    # Flush remaining buffer
                    if buf:
                        key = "reasoning" if inside_think else "content"
                        if inside_think:
                            reasoning_chars_count += len(buf)
                        else:
                            has_content = True
                        yield _make_chunk_sse(_delta(key, buf))

                    # Fallback: if model produced only reasoning with no
                    # content, re-emit reasoning as content so the client
                    # has something to show.
                    if not has_content and reasoning_chars_count > 0:
                        log.warning("Model produced only reasoning, no content — "
                                    "emitting reasoning as content fallback")
                        yield _make_chunk_sse(_delta("content",
                            "[Model produced only reasoning. "
                            "Retry with think=false or lower reasoning_budget.]"))
                else:
                    # No tools, thinking disabled — stream directly
                    async for chunk in _async_iter_sync_gen(gen):
                        if not chunk:
                            yield ": keepalive\n\n"
                            continue
                        full_text += chunk
                        if first_token_ts is None:
                            first_token_ts = time.time()
                        yield _make_chunk_sse(_delta("content", chunk))
            elif engine_type == "vision":
                from ppmlx.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(repo_id, messages, max_tokens=max_tokens or 1024)
                full_text = text
                if first_token_ts is None:
                    first_token_ts = time.time()
                yield _make_chunk_sse(_delta("content", text))
        except Exception:
            log.exception("Chat completion stream error")
            err = {"error": {"message": "Model generation failed", "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n"

        # Parse tool calls if tools were provided
        tokenizer = engine.get_tokenizer(repo_id)
        _, tool_calls = _parse_tool_calls(full_text, tokenizer=tokenizer, tools=tools) if tools else ("", [])

        if tool_calls:
            # Emit tool_calls in streaming format
            tc_list = []
            for i, tc in enumerate(tool_calls):
                call_id = "call_" + uuid.uuid4().hex[:24]
                tc_list.append({
                    "index": i,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
            delta = {"tool_calls": tc_list}
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

        total_dur = (time.time() - start_ts) * 1000
        ttft = (first_token_ts - start_ts) * 1000 if first_token_ts else None
        log_extra = {}
        if thinking_duration_ms is not None:
            log_extra["thinking_duration_ms"] = thinking_duration_ms
        if reasoning_chars_count:
            log_extra["reasoning_chars"] = reasoning_chars_count
        _log_request(
            request,
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias=model_name,
            model_repo=repo_id,
            stream=True,
            total_duration_ms=total_dur,
            time_to_first_token_ms=ttft,
            messages_count=len(messages),
            **log_extra,
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts, tools=None,
    think=None, reasoning_budget=None,
):
    """Return non-streaming JSON response."""
    # Determine thinking mode: explicit param > tool heuristic > default on
    # When a reasoning_budget is set, allow thinking even with tools.
    if think is not None:
        enable_thinking = think
    elif tools and not reasoning_budget:
        enable_thinking = False
    else:
        enable_thinking = True

    try:
        if engine_type == "text":
            from ppmlx.engine import get_engine
            engine = get_engine()
            gen_kwargs = dict(
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=max_tokens,
                seed=seed,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            if reasoning_budget is not None:
                gen_kwargs["reasoning_budget"] = reasoning_budget
            try:
                text, reasoning, prompt_tokens, completion_tokens, *_ = engine.generate(
                    repo_id, messages, **gen_kwargs,
                )
            except TypeError:
                # reasoning_budget not yet supported in engine; retry without it
                gen_kwargs.pop("reasoning_budget", None)
                text, reasoning, prompt_tokens, completion_tokens, *_ = engine.generate(
                    repo_id, messages, **gen_kwargs,
                )
        elif engine_type == "vision":
            from ppmlx.engine_vlm import get_vision_engine
            engine = get_vision_engine()
            text, prompt_tokens, completion_tokens = engine.generate(repo_id, messages)
            reasoning = None
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is an embedding model.")
    except HTTPException:
        raise
    except Exception as exc:
        _log_request(
            request,
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias=model_name,
            model_repo=repo_id,
            stream=False,
            status="error",
            error_message=str(exc),
        )
        log.exception("Chat completion generation failed")
        raise HTTPException(status_code=503, detail="Model generation failed")

    total_dur = (time.time() - start_ts) * 1000

    # Parse tool calls if tools were provided
    tokenizer = engine.get_tokenizer(repo_id)
    remaining_text, tool_calls = _parse_tool_calls(text, tokenizer=tokenizer, tools=tools) if tools else (text, [])

    # Compute reasoning token count for usage details
    try:
        reasoning_tokens_count = len(tokenizer.encode(reasoning)) if reasoning else None
    except Exception:
        reasoning_tokens_count = None

    message: dict = {"role": "assistant", "content": remaining_text or None}
    if reasoning:
        message["reasoning"] = reasoning

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
        message["tool_calls"] = [
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens_count or 0},
        },
    }

    log_kwargs = dict(
        request_id=request_id,
        endpoint="/v1/chat/completions",
        model_alias=model_name,
        model_repo=repo_id,
        stream=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        total_duration_ms=total_dur,
        messages_count=len(messages),
    )
    try:
        _log_request(
            request,
            reasoning_tokens=reasoning_tokens_count,
            thinking_enabled=1 if enable_thinking else 0,
            reasoning_budget=reasoning_budget,
            **log_kwargs,
        )
    except Exception:
        # db.py may not support thinking fields yet; fall back without them
        _log_request(request, **log_kwargs)

    return JSONResponse(response)


@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI-compatible text completions endpoint."""
    body = await request.json()
    from ppmlx.schema import CompletionRequest as _CReq; _CReq.model_validate(body)
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    max_tokens = _clamp_max_tokens(body.get("max_tokens"))
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    _track_usage("api_completions", {"stream": stream})

    messages = [{"role": "user", "content": prompt}]

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    request_id = "cmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())

    try:
        from ppmlx.engine import get_engine
        engine = get_engine()
        text, reasoning, prompt_tokens, completion_tokens, *_ = engine.generate(
            repo_id, messages,
            temperature=0.7 if temperature is None else temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        log.exception("Text completion generation failed")
        raise HTTPException(status_code=503, detail="Model generation failed")

    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint."""
    body = await request.json()
    from ppmlx.schema import EmbeddingRequest as _EReq; _EReq.model_validate(body)
    model_name = body.get("model", "")
    input_text = body.get("input", "")

    if isinstance(input_text, str):
        texts = [input_text]
    else:
        texts = list(input_text)
    _track_usage("api_embeddings", {"batch_size": len(texts)})

    if len(texts) > _MAX_EMBED_INPUTS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many inputs ({len(texts)}). Maximum is {_MAX_EMBED_INPUTS}.",
        )

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    try:
        from ppmlx.engine_embed import get_embed_engine
        engine = get_embed_engine()
        vectors = engine.encode(repo_id, texts)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Embeddings require the 'mlx-embeddings' package. Install with: pip install ppmlx[embeddings]",
        )
    except Exception:
        log.exception("Embedding generation failed")
        raise HTTPException(status_code=503, detail="Embedding generation failed")

    data = [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(vectors)]
    total_tokens = sum(len(t.split()) for t in texts)

    _log_request(
        request,
        request_id="embed-" + uuid.uuid4().hex[:8],
        endpoint="/v1/embeddings",
        model_alias=model_name,
        model_repo=repo_id,
        stream=False,
        prompt_tokens=total_tokens,
        total_tokens=total_tokens,
    )

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── OpenAI Responses API (/v1/responses) ─────────────────────────────


def _responses_input_to_messages(input_data) -> list[dict]:
    """Convert Responses API 'input' field to chat messages list."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                content = item.get("content", "")
                # Handle content array (e.g. [{type: "input_text", text: "..."}])
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text", "")
                            if t:
                                text_parts.append(t)
                    content = "\n".join(text_parts) if text_parts else ""
                messages.append({"role": role, "content": content})
        return messages
    return [{"role": "user", "content": str(input_data)}]


@app.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API endpoint (used by Codex and newer OpenAI tools)."""
    body = await request.json()
    _track_usage(
        "api_responses",
        {
            "stream": bool(body.get("stream", False)),
            "tools": bool(body.get("tools")),
            "instructions": bool(body.get("instructions")),
        },
    )

    model_name = body.get("model", "")
    input_data = body.get("input", "")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = _clamp_max_tokens(body.get("max_output_tokens") or body.get("max_tokens"))
    # instructions field acts as a system prompt
    instructions = body.get("instructions")

    tools = _limit_tools(body.get("tools") or None)
    log.info("POST /v1/responses model=%s stream=%s tools=%d",
             model_name, stream, len(tools) if tools else 0)

    messages = _responses_input_to_messages(input_data)
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    # Merge consecutive system messages into one (some models only allow a
    # single system message at the start, e.g. Qwen).
    messages = _merge_system_messages(messages)

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    resp_id = "resp_" + uuid.uuid4().hex[:24]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if stream:
        return _stream_responses(
            resp_id, msg_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, request, tools,
        )
    else:
        return await _nonstream_responses(
            resp_id, msg_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, request, tools,
        )


def _sse(event: str, data: dict, seq: list[int] | None = None) -> str:
    """Format an SSE event.

    *seq* is a mutable 1-element list used as an auto-incrementing counter
    (pass the same list across calls to get monotonic sequence numbers).
    """
    if "type" not in data:
        data = {**data, "type": event}
    if seq is not None:
        data["sequence_number"] = seq[0]
        seq[0] += 1
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_SENTINEL = object()


async def _async_iter_sync_gen(sync_gen, loop=None):
    """Yield items from a blocking sync generator without blocking the event loop.

    Runs the generator in a background thread and bridges items to the async
    world via an :class:`asyncio.Queue`.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            for item in sync_gen:
                loop.call_soon_threadsafe(q.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    import threading
    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        try:
            item = await asyncio.wait_for(q.get(), timeout=5.0)
        except asyncio.TimeoutError:
            yield ""  # keep-alive: generator still working (e.g. thinking)
            continue
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ── Shared think-tag stream parser ──────────────────────────────────

async def _parse_think_tags(raw_stream):
    """Parse ``<think>...</think>`` tags from a raw token stream.

    Yields tuples describing what was parsed:

    * ``("thinking", chunk)``  — a chunk of reasoning text
    * ``("thinking_done", full_reasoning)`` — thinking block closed
    * ``("text", chunk)``  — a chunk of answer text
    * ``("flush_text", full_text)``  — final flush of any buffered text

    Assumes the model may start *inside* a thinking block (Qwen3 injects
    ``<think>`` into the generation prompt).
    """
    in_thinking = True  # assume starting inside think block
    buf = ""
    reasoning_text = ""
    full_text = ""

    async for chunk in raw_stream:
        buf += chunk
        while buf:
            if not in_thinking:
                think_pos = buf.find("<think>")
                close_pos = buf.find("</think>")
                if think_pos == 0:
                    in_thinking = True
                    buf = buf[len("<think>"):]
                    continue
                elif close_pos == 0:
                    # Closing tag without matching open (template-injected)
                    buf = buf[len("</think>"):]
                    if reasoning_text:
                        yield ("thinking_done", reasoning_text)
                    continue
                else:
                    # Check for partial tag at end of buffer
                    partial = any(
                        buf.endswith(t[:i])
                        for t in ("<think>", "</think>")
                        for i in range(1, len(t))
                    )
                    if partial:
                        break
                    text_chunk = buf[:think_pos] if think_pos > 0 else buf
                    buf = buf[len(text_chunk):]
                    if text_chunk:
                        full_text += text_chunk
                        yield ("text", text_chunk)
                    if not buf:
                        break
            else:
                close_pos = buf.find("</think>")
                if close_pos >= 0:
                    think_chunk = buf[:close_pos]
                    buf = buf[close_pos + len("</think>"):]
                    in_thinking = False
                    if think_chunk:
                        reasoning_text += think_chunk
                        yield ("thinking", think_chunk)
                    yield ("thinking_done", reasoning_text)
                    continue
                else:
                    partial_len = 0
                    for i in range(1, len("</think>")):
                        if buf.endswith("</think>"[:i]):
                            partial_len = i
                            break
                    safe = buf[:len(buf) - partial_len] if partial_len else buf
                    buf = buf[len(safe):] if partial_len else ""
                    if safe:
                        reasoning_text += safe
                        yield ("thinking", safe)
                    break

    # Flush remaining buffer
    if buf:
        if in_thinking:
            reasoning_text += buf
            yield ("thinking", buf)
            yield ("thinking_done", reasoning_text)
        else:
            full_text += buf
            yield ("text", buf)

    yield ("flush_text", full_text)


def _stream_responses(
    resp_id, msg_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, request,
    tools=None,
):
    """Streaming SSE for the Responses API."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        log.info("responses stream start model=%s tools=%d msgs=%d",
                 model_name, len(tools) if tools else 0, len(messages))
        seq = [0]  # mutable counter for sequence_number

        # Build the full response object (matches Ollama's structure)
        resp_obj = {
            "id": resp_id,
            "object": "response",
            "created_at": created,
            "completed_at": None,
            "model": model_name,
            "status": "in_progress",
            "output": [],
            "usage": None,
            "error": None,
            "instructions": None,
            "tools": tools or [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "truncation": "disabled",
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "previous_response_id": None,
            "metadata": {},
            "text": {"format": {"type": "text"}},
        }

        # response.created — wrapped in "response" key (Codex/Ollama format)
        yield _sse("response.created", {"response": resp_obj}, seq)

        # response.in_progress
        yield _sse("response.in_progress", {"response": resp_obj}, seq)

        full_text = ""
        reasoning_text = ""
        start_ts = time.time()
        # Track output index: reasoning (if any) is at 0, message at 0 or 1
        reasoning_idx: int | None = None
        msg_output_idx = 0

        try:
            if engine_type == "text":
                from ppmlx.engine import get_engine
                engine = get_engine()
                gen = engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=max_tokens,
                    tools=tools,
                    strip_thinking=False,  # Handle thinking in server
                    enable_thinking=not bool(tools),  # Disable thinking for tool calls
                )

                # Start with reasoning item (Qwen3 starts inside think block)
                reasoning_idx = 0
                msg_output_idx = 1
                rs_id = "rs_" + uuid.uuid4().hex[:12]
                yield _sse("response.output_item.added", {
                    "output_index": reasoning_idx,
                    "item": {"id": rs_id, "type": "reasoning", "summary": []},
                }, seq)

                raw_stream = _async_iter_sync_gen(gen)
                async for kind, data in _parse_think_tags(raw_stream):
                    if kind == "thinking":
                        yield _sse("response.reasoning_summary_text.delta", {
                            "output_index": reasoning_idx,
                            "delta": data,
                        }, seq)
                    elif kind == "thinking_done":
                        reasoning_text = data
                        yield _sse("response.reasoning_summary_text.done", {
                            "output_index": reasoning_idx,
                            "text": reasoning_text,
                        }, seq)
                        yield _sse("response.output_item.done", {
                            "output_index": reasoning_idx,
                            "item": {"id": rs_id, "type": "reasoning",
                                     "summary": [{"type": "summary_text", "text": reasoning_text}]},
                        }, seq)
                    elif kind == "flush_text":
                        full_text = data

            elif engine_type == "vision":
                from ppmlx.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(
                    repo_id, messages,
                    max_tokens=max_tokens,
                )
                full_text = text
                yield _sse("response.output_text.delta", {
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }, seq)
        except Exception:
            log.exception("responses stream error")
            yield _sse("error", {
                "type": "server_error",
                "message": "Model generation failed",
            }, seq)
            return

        gen_dur = time.time() - start_ts
        log.info("responses generation done in %.1fs, %d chars", gen_dur, len(full_text))

        # Parse tool calls from the model output
        tokenizer = engine.get_tokenizer(repo_id)
        remaining_text, tool_calls = _parse_tool_calls(full_text, tokenizer=tokenizer, tools=tools)
        log.info("parsed %d tool_calls, remaining_text=%d chars",
                 len(tool_calls), len(remaining_text))

        # ── Emit text message (clean, without <tool_call> blocks) ────
        msg_item = {
            "type": "message", "id": msg_id,
            "status": "in_progress", "role": "assistant", "content": [],
        }
        yield _sse("response.output_item.added", {
            "output_index": msg_output_idx, "item": msg_item,
        }, seq)
        yield _sse("response.content_part.added", {
            "output_index": msg_output_idx, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }, seq)
        if remaining_text:
            yield _sse("response.output_text.delta", {
                "output_index": msg_output_idx, "content_index": 0,
                "delta": remaining_text,
            }, seq)
        yield _sse("response.output_text.done", {
            "output_index": msg_output_idx,
            "content_index": 0,
            "text": remaining_text,
        }, seq)
        yield _sse("response.content_part.done", {
            "output_index": msg_output_idx,
            "content_index": 0,
            "part": {"type": "output_text", "text": remaining_text, "annotations": []},
        }, seq)
        done_msg = {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
        }
        yield _sse("response.output_item.done", {
            "output_index": msg_output_idx,
            "item": done_msg,
        }, seq)

        output_items: list[dict] = []
        # Include reasoning item if we emitted one
        if reasoning_idx is not None:
            output_items.append({
                "type": "reasoning",
                "id": rs_id,
                "summary": [{"type": "summary_text", "text": reasoning_text}],
                "encrypted_content": reasoning_text,
            })
        output_items.append(done_msg)
        output_idx = msg_output_idx + 1

        # ── Emit function call items ─────────────────────────────────
        for tc in tool_calls:
            fc_id = "fc_" + uuid.uuid4().hex[:24]
            call_id = "call_" + uuid.uuid4().hex[:24]
            fc_item = {
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": tc["name"],
                "arguments": "",
                "status": "in_progress",
            }
            yield _sse("response.output_item.added", {
                "output_index": output_idx,
                "item": fc_item,
            }, seq)
            yield _sse("response.function_call_arguments.delta", {
                "output_index": output_idx,
                "delta": tc["arguments"],
            }, seq)
            yield _sse("response.function_call_arguments.done", {
                "output_index": output_idx,
                "arguments": tc["arguments"],
            }, seq)
            done_fc = {**fc_item, "arguments": tc["arguments"], "status": "completed"}
            yield _sse("response.output_item.done", {
                "output_index": output_idx,
                "item": done_fc,
            }, seq)
            output_items.append(done_fc)
            output_idx += 1

        # ── Wrap up ──────────────────────────────────────────────────
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages if isinstance(m.get("content"), str))
        completion_tokens = len(full_text.split())
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

        resp_obj["status"] = "completed"
        resp_obj["completed_at"] = int(time.time())
        resp_obj["output"] = output_items
        resp_obj["usage"] = usage
        yield _sse("response.completed", {"response": resp_obj}, seq)
        log.info("responses stream completed, %d output items", len(output_items))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_responses(
    resp_id, msg_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, request,
    tools=None,
):
    """Non-streaming Responses API."""
    try:
        if engine_type == "text":
            from ppmlx.engine import get_engine
            engine = get_engine()
            text, reasoning, prompt_tokens, completion_tokens, *_ = engine.generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=max_tokens,
                tools=tools,
                enable_thinking=not bool(tools),  # Disable thinking for tool calls
            )
        elif engine_type == "vision":
            from ppmlx.engine_vlm import get_vision_engine
            engine = get_vision_engine()
            text, prompt_tokens, completion_tokens = engine.generate(repo_id, messages)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is an embedding model.")
    except HTTPException:
        raise
    except Exception:
        log.exception("Responses generation failed")
        raise HTTPException(status_code=503, detail="Model generation failed")

    tokenizer = engine.get_tokenizer(repo_id)
    remaining_text, tool_calls = _parse_tool_calls(text, tokenizer=tokenizer, tools=tools)

    output: list[dict] = []
    if remaining_text:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
        })
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": "fc_" + uuid.uuid4().hex[:24],
            "call_id": "call_" + uuid.uuid4().hex[:24],
            "name": tc["name"],
            "arguments": tc["arguments"],
            "status": "completed",
        })
    # Fallback: if no tool calls and no remaining text, emit original text
    if not output:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })

    return JSONResponse({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "completed_at": int(time.time()),
        "model": model_name,
        "status": "completed",
        "output": output,
        "error": None,
        "tools": tools or [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "truncation": "disabled",
        "metadata": {},
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })


# ── Anthropic Messages API (/v1/messages) ─────────────────────────────
#
# Claude Code talks to this endpoint when ANTHROPIC_BASE_URL is set.
# Format mirrors the Anthropic Messages API (not OpenAI).


def _anthropic_sse(data: dict) -> str:
    event_type = data.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API endpoint (used by Claude Code)."""
    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = _clamp_max_tokens(body.get("max_tokens"))
    temperature = body.get("temperature", 0.7)
    system_prompt = body.get("system")
    tools = _limit_tools(body.get("tools") or None)

    # Thinking control — Anthropic API uses "thinking" object or budget param
    thinking_cfg = body.get("thinking") or {}
    log.info("POST /v1/messages raw thinking=%s reasoning_effort=%s",
             body.get("thinking"), body.get("reasoning_effort"))
    think_enabled = None
    budget_tokens = None
    if isinstance(thinking_cfg, dict):
        thinking_type = thinking_cfg.get("type")
        if thinking_type == "enabled":
            think_enabled = True
            budget_tokens = thinking_cfg.get("budget_tokens")
        elif thinking_type == "disabled":
            think_enabled = False
        elif thinking_type == "adaptive":
            think_enabled = True
            # adaptive = let model decide, use config budget
        # Legacy format fallback
        if think_enabled is None and "enabled" in thinking_cfg:
            think_enabled = thinking_cfg["enabled"]
        if budget_tokens is None and "budget_tokens" in thinking_cfg:
            budget_tokens = thinking_cfg["budget_tokens"]

    # Load config for thinking defaults
    try:
        from ppmlx.config import load_config
        _cfg = load_config()
    except Exception:
        _cfg = None

    # Also check reasoning_effort (OpenAI-style, some clients send it)
    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort and budget_tokens is None and _cfg:
        budget_tokens = _cfg.thinking.effort_to_budget(str(reasoning_effort))

    # Apply config defaults
    if _cfg:
        if budget_tokens is None and _cfg.thinking.default_reasoning_budget > 0:
            budget_tokens = _cfg.thinking.default_reasoning_budget
        if think_enabled is None and not _cfg.thinking.enabled:
            think_enabled = False

    # Build chat messages
    chat_messages: list[dict] = []
    if system_prompt:
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(
                p.get("text", "") for p in system_prompt if isinstance(p, dict)
            )
        chat_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Anthropic content can be a list of blocks
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_result":
                        # Tool results in Anthropic format
                        tc_content = part.get("content", "")
                        if isinstance(tc_content, list):
                            tc_content = "\n".join(
                                p.get("text", "") for p in tc_content
                                if isinstance(p, dict)
                            )
                        chat_messages.append({"role": "user", "content": tc_content})
                        continue
                    elif part.get("type") == "tool_use":
                        # Previous assistant tool call — convert
                        chat_messages.append({
                            "role": "assistant",
                            "content": f'<tool_call>\n{{"name": "{part.get("name","")}", '
                                       f'"arguments": {json.dumps(part.get("input", {}))}}}\n</tool_call>',
                        })
                        continue
            if text_parts:
                chat_messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            chat_messages.append({"role": role, "content": content})

    chat_messages = _merge_system_messages(chat_messages)

    # Convert Anthropic tools format to OpenAI format for apply_chat_template
    oai_tools = None
    if tools:
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    # Claude Code sends Anthropic model names (claude-haiku-*, claude-sonnet-*, etc.)
    # which don't exist on HuggingFace. Fall back to loaded or default model.
    if repo_id.startswith(("claude-", "anthropic/")):
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
        if loaded:
            fallback = loaded[-1]
        else:
            try:
                from ppmlx.config import load_config
                from ppmlx.models import resolve_alias as _resolve
                fallback = _resolve(load_config().defaults.model)
            except Exception:
                fallback = None
        if fallback:
            log.info("Mapping Anthropic model %s → %s", repo_id, fallback)
            repo_id = fallback
        else:
            log.warning("No model available to map %s to", repo_id)

    msg_id = "msg_" + uuid.uuid4().hex[:24]

    log.info(
        "POST /v1/messages model=%s think=%s budget=%s effort=%s effort_base=%s stream=%s tools=%d",
        model_name, think_enabled, budget_tokens, reasoning_effort,
        _cfg.thinking.effort_base if _cfg else "?", stream, len(tools or []),
    )

    if stream:
        return _stream_anthropic(
            msg_id, model_name, repo_id, chat_messages,
            temperature, max_tokens, oai_tools, tools,
            think=think_enabled, reasoning_budget=budget_tokens,
        )
    else:
        return await _nonstream_anthropic(
            msg_id, model_name, repo_id, chat_messages,
            temperature, max_tokens, oai_tools, tools,
            think=think_enabled, reasoning_budget=budget_tokens,
        )


def _stream_anthropic(
    msg_id, model_name, repo_id, messages,
    temperature, max_tokens, oai_tools, orig_tools,
    think=None, reasoning_budget=None,
):
    from fastapi.responses import StreamingResponse

    async def event_generator():
        log.debug("_stream_anthropic start model=%s think=%s budget=%s",
                  model_name, think, reasoning_budget)
        # message_start
        yield _anthropic_sse({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        full_text = ""
        content_idx = 0
        thinking_started = False
        text_started = False

        try:
            from ppmlx.engine import get_engine
            engine = get_engine()
            # Determine thinking mode
            # When a reasoning_budget is set, allow thinking even with tools
            # (the budget prevents infinite thinking loops).
            if think is not None:
                _enable_thinking = think
            elif oai_tools and not reasoning_budget:
                _enable_thinking = False
            else:
                _enable_thinking = True

            _gen_kwargs = {
                "temperature": 0.7 if temperature is None else temperature,
                "max_tokens": max_tokens,
                "tools": oai_tools,
                "strip_thinking": False,  # We handle thinking/text separation here
                "enable_thinking": _enable_thinking,
            }
            if reasoning_budget is not None:
                _gen_kwargs["reasoning_budget"] = reasoning_budget
            try:
                gen = engine.stream_generate(repo_id, messages, **_gen_kwargs)
            except TypeError:
                _gen_kwargs.pop("reasoning_budget", None)
                gen = engine.stream_generate(repo_id, messages, **_gen_kwargs)

            # Emit thinking block start immediately (Qwen3 starts inside think)
            yield _anthropic_sse({
                "type": "content_block_start",
                "index": content_idx,
                "content_block": {"type": "thinking", "thinking": ""},
            })
            thinking_started = True

            raw_stream = _async_iter_sync_gen(gen)
            async for kind, chunk in _parse_think_tags(raw_stream):
                if kind == "thinking":
                    if not thinking_started:
                        thinking_started = True
                        yield _anthropic_sse({
                            "type": "content_block_start",
                            "index": content_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    yield _anthropic_sse({
                        "type": "content_block_delta",
                        "index": content_idx,
                        "delta": {"type": "thinking_delta", "thinking": chunk},
                    })
                elif kind == "thinking_done":
                    if thinking_started:
                        yield _anthropic_sse({
                            "type": "content_block_stop",
                            "index": content_idx,
                        })
                        content_idx += 1
                        thinking_started = False
                elif kind == "text":
                    if not text_started:
                        text_started = True
                        yield _anthropic_sse({
                            "type": "content_block_start",
                            "index": content_idx,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield _anthropic_sse({
                        "type": "content_block_delta",
                        "index": content_idx,
                        "delta": {"type": "text_delta", "text": chunk},
                    })
                elif kind == "flush_text":
                    full_text = chunk

            # Close thinking block if still open
            if thinking_started:
                yield _anthropic_sse({"type": "content_block_stop", "index": content_idx})
                content_idx += 1

            # Close text block if open
            if text_started:
                yield _anthropic_sse({"type": "content_block_stop", "index": content_idx})
                content_idx += 1

        except Exception:
            log.exception("Anthropic stream error")
            yield _anthropic_sse({
                "type": "error",
                "error": {"type": "server_error", "message": "Model generation failed"},
            })
            return

        log.info("/v1/messages stream done: full_text=%d chars, text_started=%s, "
                 "thinking_started=%s",
                 len(full_text), text_started, thinking_started)

        # Parse tool calls from the collected text output
        tokenizer = engine.get_tokenizer(repo_id)
        remaining_text, tool_calls = _parse_tool_calls(full_text, tokenizer=tokenizer, tools=oai_tools)

        # Emit tool_use blocks
        stop_reason = "end_turn"
        if tool_calls:
            stop_reason = "tool_use"
            for tc in tool_calls:
                tc_id = "toolu_" + uuid.uuid4().hex[:24]
                try:
                    args_obj = json.loads(tc["arguments"])
                except (json.JSONDecodeError, ValueError):
                    args_obj = tc["arguments"]
                yield _anthropic_sse({
                    "type": "content_block_start",
                    "index": content_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_id,
                        "name": tc["name"],
                        "input": {},
                    },
                })
                yield _anthropic_sse({
                    "type": "content_block_delta",
                    "index": content_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(args_obj),
                    },
                })
                yield _anthropic_sse({
                    "type": "content_block_stop",
                    "index": content_idx,
                })
                content_idx += 1

        prompt_tokens = sum(
            len(m.get("content", "").split()) for m in messages
            if isinstance(m.get("content"), str)
        )
        completion_tokens = len(full_text.split())

        yield _anthropic_sse({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        })
        yield _anthropic_sse({"type": "message_stop"})

        # Log request to DB
        try:
            _log_request(
                endpoint="/v1/messages",
                model_alias=model_name,
                model_repo=repo_id,
                stream=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                messages_count=len(messages),
            )
        except Exception:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_anthropic(
    msg_id, model_name, repo_id, messages,
    temperature, max_tokens, oai_tools, orig_tools,
    think=None, reasoning_budget=None,
):
    # Determine thinking mode
    if think is not None:
        enable_thinking = think
    elif oai_tools and not reasoning_budget:
        enable_thinking = False
    else:
        enable_thinking = True

    try:
        from ppmlx.engine import get_engine
        engine = get_engine()
        gen_kwargs = {
            "temperature": 0.7 if temperature is None else temperature,
            "max_tokens": max_tokens,
            "tools": oai_tools,
            "enable_thinking": enable_thinking,
        }
        if reasoning_budget is not None:
            gen_kwargs["reasoning_budget"] = reasoning_budget
        try:
            result = engine.generate(repo_id, messages, **gen_kwargs)
            text, reasoning, prompt_tokens, completion_tokens = result[0], result[1], result[2], result[3]
        except TypeError:
            gen_kwargs.pop("reasoning_budget", None)
            result = engine.generate(repo_id, messages, **gen_kwargs)
            text, reasoning, prompt_tokens, completion_tokens = result[0], result[1], result[2], result[3]
    except Exception:
        log.exception("Anthropic messages generation failed")
        raise HTTPException(status_code=503, detail="Model generation failed")

    tokenizer = engine.get_tokenizer(repo_id)
    remaining_text, tool_calls = _parse_tool_calls(text, tokenizer=tokenizer, tools=oai_tools)

    content: list[dict] = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    if remaining_text:
        content.append({"type": "text", "text": remaining_text})

    stop_reason = "end_turn"
    for tc in tool_calls:
        stop_reason = "tool_use"
        try:
            args_obj = json.loads(tc["arguments"])
        except (json.JSONDecodeError, ValueError):
            args_obj = tc["arguments"]
        content.append({
            "type": "tool_use",
            "id": "toolu_" + uuid.uuid4().hex[:24],
            "name": tc["name"],
            "input": args_obj,
        })

    if not content:
        content.append({"type": "text", "text": text})

    return JSONResponse({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    })


# ── WebSocket transport for Responses API (Codex) ────────────────────


_WS_MAX_MESSAGE_BYTES = 10 * 1024 * 1024  # 10 MB limit per WS message


@app.websocket("/v1/responses")
async def responses_ws(websocket: WebSocket):
    """WebSocket transport for the Responses API (used by Codex CLI)."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > _WS_MAX_MESSAGE_BYTES:
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "invalid_request", "message": "Message too large"},
                })
                continue
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "invalid_request", "message": "Invalid JSON"},
                })
                continue
            msg_type = body.get("type", "")
            if msg_type != "response.create":
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "invalid_request", "message": f"Unknown message type: {msg_type}"},
                })
                continue

            # Extract fields from the response.create message
            response_body = body.get("response", body)
            model_name = response_body.get("model", body.get("model", ""))
            input_data = response_body.get("input", body.get("input", ""))
            temperature = response_body.get("temperature", body.get("temperature", 0.7))
            top_p = response_body.get("top_p", body.get("top_p", 1.0))
            max_tokens = _clamp_max_tokens(
                response_body.get("max_output_tokens")
                or body.get("max_output_tokens")
                or response_body.get("max_tokens")
                or body.get("max_tokens", 4096)
            )
            instructions = response_body.get("instructions", body.get("instructions"))
            tools = response_body.get("tools", body.get("tools")) or None

            messages = _responses_input_to_messages(input_data)
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            messages = _merge_system_messages(messages)

            try:
                from ppmlx.models import resolve_alias
                repo_id = resolve_alias(model_name)
            except Exception:
                repo_id = model_name

            has_imgs = _has_images(messages)
            engine_type = _route_engine(repo_id, has_imgs)

            resp_id = "resp_" + uuid.uuid4().hex[:24]
            msg_id = "msg_" + uuid.uuid4().hex[:24]
            created = int(time.time())

            resp_obj = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "model": model_name,
                "status": "in_progress",
                "output": [],
                "usage": None,
            }
            await websocket.send_json({"type": "response.created", **resp_obj})

            full_text = ""
            try:
                if engine_type == "text":
                    from ppmlx.engine import get_engine
                    engine = get_engine()
                    if tools:
                        text, _, _, _ = engine.generate(
                            repo_id, messages,
                            temperature=0.7 if temperature is None else temperature,
                            top_p=1.0 if top_p is None else top_p,
                            max_tokens=max_tokens,
                            tools=tools,
                            enable_thinking=False,  # Disable thinking for tool calls
                        )
                        full_text = text
                    else:
                        for chunk in engine.stream_generate(
                            repo_id, messages,
                            temperature=0.7 if temperature is None else temperature,
                            top_p=1.0 if top_p is None else top_p,
                            max_tokens=max_tokens,
                            strip_thinking=False,
                        ):
                            full_text += chunk
                elif engine_type == "vision":
                    from ppmlx.engine_vlm import get_vision_engine
                    engine = get_vision_engine()
                    text, _, _ = engine.generate(
                        repo_id, messages,
                        max_tokens=max_tokens,
                    )
                    full_text = text
            except Exception:
                log.exception("WebSocket generation error")
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "server_error", "message": "Model generation failed"},
                })
                continue

            tokenizer = engine.get_tokenizer(repo_id)
            remaining_text, tool_calls = _parse_tool_calls(full_text, tokenizer=tokenizer, tools=tools if tools else None)
            output_items: list[dict] = []
            output_idx = 0

            # Text message
            if remaining_text:
                msg_item = {
                    "type": "message", "id": msg_id,
                    "status": "in_progress", "role": "assistant", "content": [],
                }
                await websocket.send_json({
                    "type": "response.output_item.added",
                    "output_index": output_idx, "item": msg_item,
                })
                await websocket.send_json({
                    "type": "response.content_part.added",
                    "output_index": output_idx, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                await websocket.send_json({
                    "type": "response.output_text.delta",
                    "output_index": output_idx, "content_index": 0,
                    "delta": remaining_text,
                })
                await websocket.send_json({
                    "type": "response.output_text.done",
                    "output_index": output_idx, "content_index": 0,
                    "text": remaining_text,
                })
                await websocket.send_json({
                    "type": "response.content_part.done",
                    "output_index": output_idx, "content_index": 0,
                    "part": {"type": "output_text", "text": remaining_text, "annotations": []},
                })
                done_msg = {
                    "type": "message", "id": msg_id, "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
                }
                await websocket.send_json({
                    "type": "response.output_item.done",
                    "output_index": output_idx, "item": done_msg,
                })
                output_items.append(done_msg)
                output_idx += 1

            # Function calls
            for tc in tool_calls:
                fc_id = "fc_" + uuid.uuid4().hex[:24]
                call_id = "call_" + uuid.uuid4().hex[:24]
                fc_item = {
                    "type": "function_call", "id": fc_id, "call_id": call_id,
                    "name": tc["name"], "arguments": "", "status": "in_progress",
                }
                await websocket.send_json({
                    "type": "response.output_item.added",
                    "output_index": output_idx, "item": fc_item,
                })
                await websocket.send_json({
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_idx, "delta": tc["arguments"],
                })
                await websocket.send_json({
                    "type": "response.function_call_arguments.done",
                    "output_index": output_idx, "arguments": tc["arguments"],
                })
                done_fc = {**fc_item, "arguments": tc["arguments"], "status": "completed"}
                await websocket.send_json({
                    "type": "response.output_item.done",
                    "output_index": output_idx, "item": done_fc,
                })
                output_items.append(done_fc)
                output_idx += 1

            prompt_tokens = sum(
                len(m.get("content", "").split()) for m in messages
                if isinstance(m.get("content"), str)
            )
            completion_tokens = len(full_text.split())
            usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            resp_obj["status"] = "completed"
            resp_obj["output"] = output_items or [{
                "type": "message", "id": msg_id, "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": full_text, "annotations": []}],
            }]
            resp_obj["usage"] = usage
            await websocket.send_json({"type": "response.completed", **resp_obj})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
