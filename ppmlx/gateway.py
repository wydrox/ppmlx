"""Smart API Gateway — route between local ppmlx models and cloud providers.

Provides a single OpenAI-compatible endpoint that intelligently routes
requests to local MLX models or cloud providers (OpenAI, Anthropic) based
on configurable pattern-matching rules.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ppmlx.schema import make_request_id, now_ts

log = logging.getLogger("ppmlx.gateway")


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class RouteConfig:
    """A single routing rule: model pattern -> backend."""

    pattern: str  # glob pattern, e.g. "gpt-*", "claude-*", "*"
    backend: str  # "local", "openai", "anthropic"
    api_key_env: str = ""  # env var name for API key
    base_url: str = ""  # override base URL for the backend
    fallback: str = ""  # fallback backend on error/timeout
    timeout: float = 30.0  # request timeout in seconds


@dataclass
class GatewayConfig:
    """Gateway configuration parsed from config.toml [gateway] section."""

    port: int = 6768
    host: str = "127.0.0.1"
    local_server: str = "http://127.0.0.1:6767"  # ppmlx server URL
    routes: list[RouteConfig] = field(default_factory=list)


def load_gateway_config() -> GatewayConfig:
    """Load gateway config from ~/.ppmlx/config.toml [gateway] section."""
    import tomllib
    from ppmlx.config import get_ppmlx_dir

    gw = GatewayConfig()

    toml_path = get_ppmlx_dir() / "config.toml"
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return gw

    gw_data = data.get("gateway", {})
    if not gw_data:
        return gw

    if "port" in gw_data:
        gw.port = int(gw_data["port"])
    if "host" in gw_data:
        gw.host = str(gw_data["host"])
    if "local_server" in gw_data:
        gw.local_server = str(gw_data["local_server"])

    for route_data in gw_data.get("routes", []):
        gw.routes.append(
            RouteConfig(
                pattern=str(route_data.get("pattern", "*")),
                backend=str(route_data.get("backend", "local")),
                api_key_env=str(route_data.get("api_key_env", "")),
                base_url=str(route_data.get("base_url", "")),
                fallback=str(route_data.get("fallback", "")),
                timeout=float(route_data.get("timeout", 30.0)),
            )
        )

    return gw


# ── Cloud provider clients ───────────────────────────────────────────────

_DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
_DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1"


def _get_api_key(env_var: str) -> str | None:
    """Retrieve an API key from an environment variable."""
    if not env_var:
        return None
    return os.environ.get(env_var)


async def _proxy_to_local(
    client: httpx.AsyncClient,
    local_server: str,
    path: str,
    body: dict,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    """Forward a request to the local ppmlx server."""
    url = f"{local_server.rstrip('/')}{path}"
    if stream:
        req = client.build_request("POST", url, json=body, timeout=120.0)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            _relay_sse(resp),
            media_type="text/event-stream",
            headers={"X-ppmlx-backend": "local"},
        )
    else:
        resp = await client.post(url, json=body, timeout=120.0)
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
            headers={"X-ppmlx-backend": "local"},
        )


async def _relay_sse(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Relay SSE chunks from an upstream response."""
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()


async def _proxy_to_openai(
    client: httpx.AsyncClient,
    body: dict,
    api_key: str,
    base_url: str,
    timeout: float,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    """Forward a chat completion request to the OpenAI API."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if stream:
        req = client.build_request(
            "POST", url, json=body, headers=headers, timeout=timeout
        )
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            _relay_sse(resp),
            media_type="text/event-stream",
            headers={"X-ppmlx-backend": "openai"},
        )
    else:
        resp = await client.post(url, json=body, headers=headers, timeout=timeout)
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
            headers={"X-ppmlx-backend": "openai"},
        )


def _openai_to_anthropic_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI message format to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    Anthropic uses a separate `system` parameter instead of a system message.
    """
    system_prompt: str | None = None
    anthropic_msgs: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if system_prompt:
                system_prompt += "\n\n" + (content or "")
            else:
                system_prompt = content or ""
        elif role == "assistant":
            anthropic_msgs.append({"role": "assistant", "content": content or ""})
        else:
            # user, tool, developer -> user
            anthropic_msgs.append({"role": "user", "content": content or ""})

    return system_prompt, anthropic_msgs


def _anthropic_to_openai_response(
    anthropic_resp: dict, model: str
) -> dict:
    """Convert an Anthropic response to OpenAI chat completion format."""
    content_blocks = anthropic_resp.get("content", [])
    text_parts = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif isinstance(block, str):
            text_parts.append(block)
    text = "".join(text_parts)

    usage = anthropic_resp.get("usage", {})
    return {
        "id": make_request_id(),
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _map_anthropic_stop(
                    anthropic_resp.get("stop_reason", "end_turn")
                ),
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        },
    }


_ANTHROPIC_STOP_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _map_anthropic_stop(stop_reason: str) -> str:
    """Map Anthropic stop reason to OpenAI finish_reason."""
    return _ANTHROPIC_STOP_MAP.get(stop_reason, "stop")


async def _proxy_to_anthropic(
    client: httpx.AsyncClient,
    body: dict,
    api_key: str,
    base_url: str,
    timeout: float,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    """Forward a chat completion request to the Anthropic API.

    Translates OpenAI format to Anthropic Messages API format and back.
    Streaming translates Anthropic SSE events to OpenAI SSE format.
    """
    messages = body.get("messages", [])
    system_prompt, anthropic_msgs = _openai_to_anthropic_messages(messages)

    if not anthropic_msgs:
        anthropic_msgs = [{"role": "user", "content": ""}]

    anthropic_body: dict[str, Any] = {
        "model": body.get("model", "claude-sonnet-4-20250514"),
        "messages": anthropic_msgs,
        "max_tokens": body.get("max_tokens") or 4096,
    }
    if system_prompt:
        anthropic_body["system"] = system_prompt
    if body.get("temperature") is not None:
        anthropic_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        anthropic_body["top_p"] = body["top_p"]
    if body.get("stop"):
        stop = body["stop"]
        if isinstance(stop, str):
            stop = [stop]
        anthropic_body["stop_sequences"] = stop

    url = f"{base_url.rstrip('/')}/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    if stream:
        anthropic_body["stream"] = True
        req = client.build_request(
            "POST", url, json=anthropic_body, headers=headers, timeout=timeout
        )
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            _translate_anthropic_stream(resp, body.get("model", "")),
            media_type="text/event-stream",
            headers={"X-ppmlx-backend": "anthropic"},
        )
    else:
        resp = await client.post(
            url, json=anthropic_body, headers=headers, timeout=timeout
        )
        if resp.status_code != 200:
            return JSONResponse(
                content=resp.json(),
                status_code=resp.status_code,
                headers={"X-ppmlx-backend": "anthropic"},
            )
        openai_resp = _anthropic_to_openai_response(resp.json(), body.get("model", ""))
        return JSONResponse(
            content=openai_resp,
            status_code=200,
            headers={"X-ppmlx-backend": "anthropic"},
        )


async def _translate_anthropic_stream(
    resp: httpx.Response, model: str
) -> AsyncIterator[bytes]:
    """Translate Anthropic SSE stream to OpenAI SSE format."""
    request_id = make_request_id()
    created = now_ts()

    def _sse_chunk(delta: dict, finish_reason: str | None) -> bytes:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    try:
        buffer = ""
        async for raw_chunk in resp.aiter_text():
            buffer += raw_chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        return
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        if text:
                            yield _sse_chunk({"content": text}, None)
                    elif event_type == "message_stop":
                        yield _sse_chunk({}, "stop")
                        yield b"data: [DONE]\n\n"
                        return
    finally:
        await resp.aclose()


# ── Router ───────────────────────────────────────────────────────────────


def match_route(model: str, routes: list[RouteConfig]) -> RouteConfig | None:
    """Find the first route whose pattern matches the model name."""
    for route in routes:
        if fnmatch.fnmatch(model, route.pattern):
            return route
    return None


def resolve_backend(
    route: RouteConfig,
) -> tuple[str, str, str | None]:
    """Resolve a route to (backend_name, base_url, api_key).

    Returns the backend type, the effective base URL, and the API key (if any).
    """
    backend = route.backend
    api_key = _get_api_key(route.api_key_env)

    if backend == "openai":
        base_url = route.base_url or _DEFAULT_OPENAI_URL
        return backend, base_url, api_key
    elif backend == "anthropic":
        base_url = route.base_url or _DEFAULT_ANTHROPIC_URL
        return backend, base_url, api_key
    else:
        # "local" or unknown -> local
        return "local", "", None


def build_routing_table(config: GatewayConfig) -> list[dict[str, str]]:
    """Build a human-readable routing table for display."""
    rows = []
    for route in config.routes:
        backend, base_url, api_key = resolve_backend(route)
        has_key = "yes" if api_key else "no"
        fallback = route.fallback or "none"
        rows.append(
            {
                "pattern": route.pattern,
                "backend": backend,
                "base_url": base_url or "(local ppmlx)",
                "api_key": has_key,
                "fallback": fallback,
                "timeout": f"{route.timeout}s",
            }
        )
    return rows


# ── FastAPI gateway app ──────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def gateway_lifespan(app: FastAPI):
    """Manage the httpx client lifecycle."""
    global _http_client
    _http_client = httpx.AsyncClient(follow_redirects=True)
    yield
    await _http_client.aclose()
    _http_client = None


def create_gateway_app(config: GatewayConfig | None = None) -> FastAPI:
    """Create the gateway FastAPI application."""
    if config is None:
        config = load_gateway_config()

    gateway_app = FastAPI(
        title="ppmlx gateway",
        description="Smart API gateway routing between local and cloud LLMs",
        lifespan=gateway_lifespan,
    )

    gateway_app.state.gateway_config = config

    gateway_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @gateway_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "ppmlx-gateway",
            "routes": len(config.routes),
        }

    @gateway_app.get("/v1/models")
    async def list_models():
        """List models from all configured backends."""
        models: list[dict] = []

        # Local models
        if _http_client:
            try:
                resp = await _http_client.get(
                    f"{config.local_server.rstrip('/')}/v1/models", timeout=5.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        m["owned_by"] = "ppmlx-local"
                        models.append(m)
            except Exception:
                pass

        # Add well-known cloud model stubs from routes
        for route in config.routes:
            if route.backend in ("openai", "anthropic"):
                models.append(
                    {
                        "id": route.pattern,
                        "object": "model",
                        "created": now_ts(),
                        "owned_by": route.backend,
                    }
                )

        return {"object": "list", "data": models}

    @gateway_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Route chat completions to the appropriate backend."""
        body = await request.json()
        model_name = body.get("model", "")
        stream = body.get("stream", False)

        route = match_route(model_name, config.routes)
        if route is None:
            # Default to local
            route = RouteConfig(pattern="*", backend="local")

        return await _dispatch(config, route, body, stream)

    @gateway_app.post("/v1/completions")
    async def completions(request: Request):
        """Route text completions to the appropriate backend."""
        body = await request.json()
        model_name = body.get("model", "")
        stream = body.get("stream", False)

        route = match_route(model_name, config.routes)
        if route is None:
            route = RouteConfig(pattern="*", backend="local")

        backend, base_url, api_key = resolve_backend(route)
        if _http_client is None:
            raise HTTPException(status_code=503, detail="Gateway is not ready")

        if backend == "local":
            return await _proxy_to_local(
                _http_client, config.local_server, "/v1/completions", body, stream
            )
        elif backend == "openai" and api_key:
            # OpenAI supports /v1/completions
            url = f"{base_url.rstrip('/')}/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = await _http_client.post(
                url, json=body, headers=headers, timeout=route.timeout
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        else:
            return JSONResponse(
                content={
                    "error": {
                        "message": f"Backend '{backend}' does not support /v1/completions or is missing API key",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )

    @gateway_app.post("/v1/embeddings")
    async def embeddings(request: Request):
        """Route embedding requests — always to local."""
        body = await request.json()
        if _http_client is None:
            raise HTTPException(status_code=503, detail="Gateway is not ready")
        return await _proxy_to_local(
            _http_client, config.local_server, "/v1/embeddings", body, False
        )

    @gateway_app.get("/gateway/routes")
    async def get_routes():
        """Show the current routing table."""
        return {"routes": build_routing_table(config)}

    return gateway_app


async def _dispatch(
    config: GatewayConfig,
    route: RouteConfig,
    body: dict,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    """Dispatch a request to the resolved backend, with fallback on error."""
    assert _http_client is not None

    backend, base_url, api_key = resolve_backend(route)

    try:
        return await _call_backend(
            config, backend, base_url, api_key, route.timeout, body, stream
        )
    except Exception as exc:
        log.warning("Backend %s failed: %s", backend, exc)

        # Try fallback if configured
        if route.fallback:
            fallback_route = RouteConfig(
                pattern=route.pattern,
                backend=route.fallback,
                api_key_env=_infer_api_key_env(route.fallback),
                timeout=route.timeout,
            )
            fb_backend, fb_url, fb_key = resolve_backend(fallback_route)
            try:
                log.info("Falling back to %s", fb_backend)
                return await _call_backend(
                    config, fb_backend, fb_url, fb_key, route.timeout, body, stream
                )
            except Exception as fb_exc:
                log.error("Fallback %s also failed: %s", fb_backend, fb_exc)
                return JSONResponse(
                    content={
                        "error": {
                            "message": f"All backends failed. Primary ({backend}): {exc}. Fallback ({fb_backend}): {fb_exc}",
                            "type": "server_error",
                        }
                    },
                    status_code=502,
                )

        return JSONResponse(
            content={
                "error": {
                    "message": f"Backend '{backend}' failed: {exc}",
                    "type": "server_error",
                }
            },
            status_code=502,
        )


async def _call_backend(
    config: GatewayConfig,
    backend: str,
    base_url: str,
    api_key: str | None,
    timeout: float,
    body: dict,
    stream: bool,
) -> JSONResponse | StreamingResponse:
    """Call a specific backend."""
    assert _http_client is not None

    if backend == "local":
        return await _proxy_to_local(
            _http_client, config.local_server, "/v1/chat/completions", body, stream
        )
    elif backend == "openai":
        if not api_key:
            raise ValueError("OpenAI API key not set (check api_key_env in config)")
        return await _proxy_to_openai(
            _http_client, body, api_key, base_url, timeout, stream
        )
    elif backend == "anthropic":
        if not api_key:
            raise ValueError(
                "Anthropic API key not set (check api_key_env in config)"
            )
        return await _proxy_to_anthropic(
            _http_client, body, api_key, base_url, timeout, stream
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _infer_api_key_env(backend: str) -> str:
    """Infer the default API key env var for a backend."""
    defaults = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return defaults.get(backend, "")
