from __future__ import annotations
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, cast

if TYPE_CHECKING:
    from ppmlx.config import KVCacheConfig

log = logging.getLogger("ppmlx.engine")


def _patch_rotating_kv_cache():
    """Fix RotatingKVCache._temporal_order crash when prompt > max_size.

    See https://github.com/ml-explore/mlx-lm/pull/1104
    """
    try:
        from mlx_lm.models.cache import RotatingKVCache

        _orig_temporal_order = RotatingKVCache._temporal_order
        _cache_trim_warned = set()

        def _safe_temporal_order(self, v):
            result = _orig_temporal_order(self, v)
            # Trim to max_size if the result is larger (prompt exceeded window)
            if result.shape[2] > self.max_size:
                excess = result.shape[2] - self.max_size
                if id(self) not in _cache_trim_warned:
                    _cache_trim_warned.add(id(self))
                    log.warning(
                        "Prompt exceeds sliding window (%d > %d) — "
                        "trimming %d tokens from KV cache. "
                        "Response quality may be degraded.",
                        result.shape[2], self.max_size, excess,
                    )
                result = result[..., -self.max_size:, :]
            return result

        # ``_temporal_order`` is a method on the third-party class.  Use
        # ``setattr`` so type checkers do not treat this runtime patch as an
        # invalid assignment to a declared method.
        setattr(RotatingKVCache, "_temporal_order", _safe_temporal_order)
    except Exception:
        pass


_patch_rotating_kv_cache()

_turboquant_patched = False


def _patch_turboquant_cache(kv_cfg: "KVCacheConfig") -> None:
    """Monkey-patch make_prompt_cache to wrap KVCache with TurboQuant compression."""
    global _turboquant_patched
    if _turboquant_patched:
        return
    try:
        from mlx_lm.models.cache import make_prompt_cache as _orig_make, KVCache
        from ppmlx.turboquant import TurboQuantCache
        import mlx_lm.models.cache as cache_mod

        bits, qjl, qjl_dim = kv_cfg.bits, kv_cfg.qjl, kv_cfg.qjl_dim

        def _turbo_make(model, max_kv_size=None):
            caches = _orig_make(model, max_kv_size)
            return [
                TurboQuantCache.wrap(c, bits=bits, qjl=qjl, qjl_dim=qjl_dim)
                if isinstance(c, KVCache) else c
                for c in caches
            ]

        cache_mod.make_prompt_cache = _turbo_make
        _turboquant_patched = True
        log.info("TurboQuant KV-cache compression enabled (%d-bit, QJL=%s)", bits, qjl)
    except Exception as exc:
        log.warning("Failed to patch TurboQuant cache: %s", exc)


@dataclass
class LoadedModel:
    """A model that has been loaded into memory."""
    repo_id: str
    model: Any          # mlx_lm model object
    tokenizer: Any      # mlx_lm tokenizer object
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.monotonic)


class GenerateResult(NamedTuple):
    """Result of a non-streaming generation.

    Five fields; callers that previously unpacked four should use
    ``text, reasoning, pt, ct, *_ = engine.generate(...)`` or named access.
    """
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0


_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CHANNEL_THOUGHT_PATTERN = re.compile(r"<\|channel>thought\n?(.*?)<channel\|>", re.DOTALL)

# Sanity cap: never auto-set max_tokens above this even on huge-context models.
# 4096 matches GPT-4o / Claude defaults; clients can always request more explicitly.
_MAX_AUTO_TOKENS = 4_096

# Thinking models get a higher default budget so reasoning doesn't exhaust all tokens.
_MAX_AUTO_TOKENS_THINKING = 8_192


def _context_size(lm: "LoadedModel") -> int:
    """Return the model's context window size from its tokenizer."""
    tok = lm.tokenizer
    inner = getattr(tok, "tokenizer", tok)
    for obj in (tok, inner):
        val = getattr(obj, "model_max_length", None)
        if val and isinstance(val, int) and val < 10 ** 9:
            return val
    return 4096  # conservative fallback


def _is_thinking_model(tokenizer: Any) -> bool:
    """Return True if the tokenizer's chat template contains thinking markers."""
    template = getattr(tokenizer, "chat_template", None)
    if template and isinstance(template, str):
        return "<think>" in template or "<|channel>thought" in template or "<|think|>" in template
    return False


def _is_gemma4_model(tokenizer: Any) -> bool:
    """Return True if the tokenizer uses Gemma-4 channel-style tokens."""
    template = getattr(tokenizer, "chat_template", None)
    if template and isinstance(template, str):
        return "<|channel>" in template and "<channel|>" in template
    return False


# Gemma-4 channel token patterns (longest first for prefix matching)
_GEMMA4_CHANNEL_PREFIXES = ["<|channel>thought", "<|channel>on", "<channel|>"]


class _Gemma4Normalizer:
    """Stateful converter from Gemma-4 channel tokens to ``<think>``/``</think>``.

    Works at **two levels** simultaneously:

    1. **Token-ID based** (primary) — Intercepts ``<|channel>`` (100),
       ``<channel|>`` (101), ``thought`` (45518), ``on`` (498) token IDs from
       ``GenerationResponse.token`` to track thinking state.  These tokens are
       invisible to the detokenizer (stripped as special tokens), so we inject
       ``<think>``/``</think>`` markers synthetically.

    2. **Text-based** (fallback) — If channel tokens DO leak into decoded text
       (some detokenizer configurations), strips them and converts to standard
       think tags.
    """

    # Gemma-4 special token IDs
    _CHANNEL_START = 100   # <|channel>
    _CHANNEL_END = 101     # <channel|>
    _THOUGHT = 45518       # "thought" (after <|channel>)
    _ON = 498              # "on" (after <|channel>)

    def __init__(self) -> None:
        self._in_thought = False
        self._pending_channel = False
        self._text_buf = ""  # for text-based fallback

    def feed(self, text: str, token_id: int | None = None) -> str:
        """Feed a text chunk and optional token ID, return normalized output.

        Call with both ``text`` and ``token_id`` from ``GenerationResponse``
        for best results.  Falls back to text-only normalization when
        ``token_id`` is None.
        """
        marker = ""
        suppress = False

        if token_id is not None:
            if token_id == self._CHANNEL_START:
                self._pending_channel = True
                suppress = True
            elif self._pending_channel:
                self._pending_channel = False
                if token_id == self._THOUGHT:
                    self._in_thought = True
                    # Don't emit <think> — callers handle thinking-start
                    # state (engine sets inside_think, server uses assume_thinking)
                    suppress = True
                elif token_id == self._ON:
                    if self._in_thought:
                        self._in_thought = False
                        marker = "</think>"
                    suppress = True
                # else: false alarm — not a channel sequence; don't suppress
            elif token_id == self._CHANNEL_END:
                if self._in_thought:
                    self._in_thought = False
                    marker = "</think>"
                suppress = True

        if suppress:
            # Token-ID based: don't yield the original text for channel tokens
            return marker

        # Text-based fallback: strip any channel text that leaked through
        self._text_buf += text
        cleaned = self._strip_channel_text()
        return marker + cleaned

    def flush(self) -> str:
        """Flush any remaining buffered data (call at end of stream)."""
        out = self._text_buf
        self._text_buf = ""
        return out

    def _strip_channel_text(self) -> str:
        """Strip channel-token text and inject think markers (fallback path)."""
        buf = self._text_buf
        result: list[str] = []
        while buf:
            if buf[0] == "<":
                if buf.startswith("<|channel>thought"):
                    buf = buf[len("<|channel>thought"):]
                    if buf.startswith("\n"):
                        buf = buf[1:]
                    self._in_thought = True
                    result.append("<think>")
                    continue
                if buf.startswith("<|channel>on"):
                    buf = buf[len("<|channel>on"):]
                    if buf.startswith("\n"):
                        buf = buf[1:]
                    if self._in_thought:
                        self._in_thought = False
                        result.append("</think>")
                    continue
                if buf.startswith("<channel|>"):
                    buf = buf[len("<channel|>"):]
                    if buf.startswith("\n"):
                        buf = buf[1:]
                    if self._in_thought:
                        self._in_thought = False
                        result.append("</think>")
                    continue
                # Partial prefix — wait for more data
                if any(p.startswith(buf) for p in _GEMMA4_CHANNEL_PREFIXES):
                    break
                result.append(buf[0])
                buf = buf[1:]
            else:
                next_lt = buf.find("<")
                if next_lt == -1:
                    result.append(buf)
                    buf = ""
                else:
                    result.append(buf[:next_lt])
                    buf = buf[next_lt:]
        self._text_buf = buf
        return "".join(result)


def _auto_max_tokens(lm: "LoadedModel") -> int:
    """Choose a default max_tokens based on model context size and type."""
    cap = _MAX_AUTO_TOKENS_THINKING if _is_thinking_model(lm.tokenizer) else _MAX_AUTO_TOKENS
    return min(_context_size(lm) // 2, cap)


def _strip_thinking(text: str) -> tuple[str, str | None]:
    """
    Strip thinking blocks from text.
    Returns (answer_text, reasoning_text | None).

    Handles three cases:
    1. Standard: ``<think>reasoning</think>answer``
    2. Template-injected: output starts *inside* a think block (Qwen3 adds
       ``<think>\\n`` to the generation prompt, so the model output begins with
       the reasoning directly followed by ``</think>``).
    3. Channel markers: ``<|channel>thought ... <channel|>answer``.
    """
    channel_match = _CHANNEL_THOUGHT_PATTERN.search(text)
    if channel_match:
        reasoning = channel_match.group(1).strip()
        answer = _CHANNEL_THOUGHT_PATTERN.sub("", text).strip()
        return answer, (reasoning or None)

    # Fallback: no opening channel marker but a closing <channel|> exists.
    channel_end_idx = text.find("<channel|>")
    if channel_end_idx != -1:
        reasoning = text[:channel_end_idx].strip()
        answer = text[channel_end_idx + len("<channel|>"):].strip()
        return answer, (reasoning or None)

    match = _THINK_PATTERN.search(text)
    if match:
        reasoning = match.group(1).strip()
        answer = _THINK_PATTERN.sub("", text).strip()
        return answer, reasoning
    # Fallback: no opening <think> but a closing </think> exists — the model
    # started inside a think block injected by the chat template.
    end_idx = text.find("</think>")
    if end_idx != -1:
        reasoning = text[:end_idx].strip()
        answer = text[end_idx + len("</think>"):].strip()
        return answer, (reasoning or None)
    return text, None


def _apply_seed(seed: int | None) -> None:
    """
    Seed MLX's global RNG for (best-effort) deterministic sampling.

    mlx-lm 0.31.x removed the ``seed`` keyword from ``generate`` /
    ``generate_step``; seeding ``mlx.core.random`` directly is the supported
    way to make sampling reproducible.
    """
    if seed is None:
        return
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except Exception as exc:  # pragma: no cover - mlx missing/unseedable
        log.warning("Could not seed mlx RNG (seed=%s): %s", seed, exc)


def _resolve_model_path(repo_id: str) -> str:
    """Resolve a repo_id to a local path if available, otherwise return the repo_id."""
    try:
        from ppmlx.models import resolve_model_path
        # Interactive CLI/serve flows resolve user-selected registry ids and
        # may fall back to HuggingFace; the Agent IR provider layer is the
        # fail-closed LOCAL data-path boundary, not this legacy wrapper.
        return resolve_model_path(repo_id, allow_download=True)
    except ImportError:
        return repo_id


def _can_retry_gemma4_non_strict(path: str, exc: ValueError) -> bool:
    """Return true for the known MLX Gemma 4 extra-weight mismatch.

    Some converted Gemma 4 MLX checkpoints include explicit KV weights for
    layers that the current ``mlx-lm`` Gemma 4 implementation treats as
    KV-shared. Strict loading rejects those unused extra tensors before the
    model can run. Keep the fallback narrow so unrelated shape/config errors
    still fail loudly.
    """
    message = str(exc)
    if "parameters not in model" not in message:
        return False
    model_path = Path(path).expanduser()
    if not model_path.exists() or not model_path.is_dir():
        return False
    config_path = model_path / "config.json"
    try:
        with open(config_path, "rb") as f:
            config = json.load(f)
    except Exception:
        return False
    model_type = str(config.get("model_type") or "").lower()
    text_model_type = str((config.get("text_config") or {}).get("model_type") or "").lower()
    return model_type == "gemma4" or text_model_type == "gemma4_text"


def _load_local_model_non_strict(path: str):
    """Load a local model path with strict weight matching disabled."""
    from mlx_lm.utils import load_model, load_tokenizer

    model_path = Path(path).expanduser()
    model, config = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


class TextEngine:
    """
    Wraps mlx-lm for text generation with LRU model caching.

    Thread-safe: uses a lock for model loading/unloading and a separate
    lock to serialize GPU inference (MLX Metal crashes on concurrent generates).
    """

    _REAPER_INTERVAL = 30  # seconds between TTL reaper sweeps

    def __init__(self, max_loaded: int = 2, ttl_seconds: int = 0,
                 prompt_cache_limit: int = 4,
                 kv_cache: "KVCacheConfig | None" = None):
        from ppmlx.config import KVCacheConfig
        self._max_loaded = max_loaded
        self._ttl_seconds = ttl_seconds
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        kv_cfg = kv_cache or KVCacheConfig()
        if kv_cfg.quantize == "turboquant":
            _patch_turboquant_cache(kv_cfg)
        # Prompt KV-cache reuse (skip prefill for repeated prompt prefixes)
        from ppmlx.prompt_cache import PromptCacheStore
        self._prompt_cache = PromptCacheStore(max_entries=prompt_cache_limit)
        if self._ttl_seconds > 0:
            self._start_reaper()

    def _start_reaper(self) -> None:
        """Start the background TTL reaper thread."""
        self._reaper_stop.clear()
        t = threading.Thread(target=self._ttl_reaper, daemon=True, name="ppmlx-ttl-reaper")
        t.start()
        self._reaper_thread = t

    def _ttl_reaper(self) -> None:
        """Background loop that unloads models idle longer than ttl_seconds."""
        while not self._reaper_stop.wait(timeout=self._REAPER_INTERVAL):
            self._reap_expired()

    def _reap_expired(self) -> list[str]:
        """Single sweep: unload models idle longer than ttl_seconds. Returns expired IDs."""
        if self._ttl_seconds <= 0:
            return []
        now = time.monotonic()
        expired: list[str] = []
        with self._lock:
            for repo_id, lm in list(self._models.items()):
                if (now - lm.last_used_at) >= self._ttl_seconds:
                    expired.append(repo_id)
            for repo_id in expired:
                del self._models[repo_id]
        for repo_id in expired:
            self._emit_event("unload", repo_id)
        return expired

    def stop_reaper(self) -> None:
        """Signal the reaper thread to stop and wait for it to finish."""
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5)
            self._reaper_thread = None

    @staticmethod
    def _ensure_chat_template(tokenizer: Any, repo_id: str) -> None:
        """Fetch chat_template from transformers if missing from tokenizer files.

        Also infers the tool parser when the chat template is added late
        (mlx-lm only runs inference at load time, before this fallback).
        """
        if not getattr(tokenizer, "chat_template", None):
            try:
                from transformers import AutoTokenizer as _AT
                ref_tok = _AT.from_pretrained(repo_id, trust_remote_code=True)
                tmpl = getattr(ref_tok, "chat_template", None)
                if tmpl:
                    tokenizer.chat_template = tmpl
            except Exception:
                log.debug("Could not fetch chat template for %s", repo_id)

        # Infer tool parser if template is present but parser is missing.
        if (
            getattr(tokenizer, "chat_template", None)
            and not getattr(tokenizer, "has_tool_calling", False)
        ):
            try:
                from mlx_lm.tokenizer_utils import _infer_tool_parser
                import importlib

                parser_type = _infer_tool_parser(tokenizer.chat_template)
                if parser_type is not None:
                    tool_mod = importlib.import_module(
                        f"mlx_lm.tool_parsers.{parser_type}"
                    )
                    tokenizer._tool_parser = tool_mod.parse_tool_call
                    tokenizer._tool_call_start = tool_mod.tool_call_start
                    tokenizer._tool_call_end = tool_mod.tool_call_end
                    log.info(
                        "Late-inferred tool parser %r for %s",
                        parser_type, repo_id,
                    )
            except Exception:
                log.debug("Tool parser inference failed for %s", repo_id)

    def _load_impl(self, repo_id: str) -> LoadedModel:
        """Actually load a model using mlx_lm.load. Called under lock."""
        from mlx_lm import load as mlx_load
        path = _resolve_model_path(repo_id)
        try:
            model, tokenizer = cast(tuple[Any, Any], mlx_load(path))
        except ValueError as exc:
            if not _can_retry_gemma4_non_strict(path, exc):
                raise
            log.debug(
                "Gemma 4 checkpoint has extra weights not represented by this mlx-lm model class; "
                "retrying load with strict=False for %s",
                repo_id,
            )
            model, tokenizer = _load_local_model_non_strict(path)
        return LoadedModel(repo_id=repo_id, model=model, tokenizer=tokenizer)

    def load(self, repo_id: str) -> LoadedModel:
        """
        Load a model into the LRU cache.
        Evicts least-recently-used model if cache is full.
        """
        evicted_ids: list[str] = []
        with self._lock:
            if repo_id in self._models:
                self._models.move_to_end(repo_id)
                lm = self._models[repo_id]
                lm.last_used = time.time()
                lm.last_used_at = time.monotonic()
                return lm

            # Evict LRU if at capacity
            while len(self._models) >= self._max_loaded:
                evicted_id, _ = self._models.popitem(last=False)
                evicted_ids.append(evicted_id)

            lm = self._load_impl(repo_id)
            self._models[repo_id] = lm

        for evicted_id in evicted_ids:
            self._emit_event("unload", evicted_id)
        self._emit_event("load", repo_id)
        return lm

    def _emit_event(self, event: str, repo_id: str) -> None:
        """Log model lifecycle event to DB (best-effort)."""
        try:
            from ppmlx.db import get_db
            get_db().log_model_event(event=event, model_repo=repo_id)
        except Exception:
            pass

    def _apply_chat_template(
        self, lm: LoadedModel, messages: list[dict],
        enable_thinking: bool = True, tools: list[dict] | None = None,
        strict_tools: bool = False,
    ) -> str:
        """Apply the model's chat template to produce a prompt string."""
        tokenizer = lm.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = tools
            # Pass enable_thinking explicitly — some templates (e.g. Gemma-4)
            # default to False and need this to activate thinking mode.
            try:
                return tokenizer.apply_chat_template(
                    messages, **kwargs, enable_thinking=enable_thinking
                )
            except TypeError:
                pass  # model doesn't support the flag — fall through
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Tokenizer doesn't support 'tools' kwarg — retry without it
                if tools and "tools" in kwargs:
                    if strict_tools:
                        raise RuntimeError(
                            "The selected local model cannot accept tool definitions"
                        ) from None
                    log.warning(
                        "Tokenizer for %s does not support 'tools' kwarg — "
                        "%d tools dropped from chat template. "
                        "Tool calling may not work with this model.",
                        lm.repo_id, len(tools),
                    )
                    del kwargs["tools"]
                    return tokenizer.apply_chat_template(messages, **kwargs)
                raise
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    _PREFILL_CHUNK_SIZE = 2048

    def _apply_prompt_cache(
        self, repo_id: str, lm: LoadedModel, prompt: str,
        kwargs: dict[str, Any], *, draft_model: str | None = None,
    ) -> int | None:
        """Modify *kwargs* in-place to use a cached KV-cache prefix.

        Returns the prompt token count (to avoid redundant re-tokenization),
        or ``None`` if caching was skipped.

        Must be called under ``self._generate_lock``.
        Skipped when caching is disabled, speculative decoding is active,
        or MLX cache utilities aren't available.
        """
        if not self._prompt_cache.is_enabled or draft_model is not None:
            return None
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        except ImportError:
            return None

        prompt_tokens_list = lm.tokenizer.encode(prompt)
        cache, num_cached = self._prefill_with_cache(
            repo_id, lm, prompt_tokens_list,
        )
        if cache is None:
            return len(prompt_tokens_list)

        # After _prefill_with_cache, the cache covers ALL prompt tokens.
        # generate_step requires ≥1 prompt token, so trim the last one
        # from the cache and pass it as the prompt seed.
        if can_trim_prompt_cache(cache):
            trim_prompt_cache(cache, 1)
        kwargs["prompt"] = mx.array([prompt_tokens_list[-1]])
        kwargs["prompt_cache"] = cache
        if num_cached > 0:
            log.info(
                "Prompt cache: skipped prefill for %d/%d tokens",
                num_cached, len(prompt_tokens_list),
            )
        return len(prompt_tokens_list)

    def _prefill_with_cache(
        self, repo_id: str, lm: LoadedModel, prompt_tokens: list[int],
    ) -> tuple[list[Any] | None, int]:
        """Prefill the prompt, reusing cached KV-cache prefixes when possible.

        Returns ``(prompt_cache, num_cached_tokens)`` or ``(None, 0)`` if
        MLX cache utilities aren't available.  The cache is fully prefilled
        up to ``prompt_tokens`` and ready for generation.  A deep-copy is
        stored in ``self._prompt_cache`` for future reuse.

        Must be called under ``self._generate_lock`` (uses GPU).
        """
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError:
            return None, 0

        cached_kv, remaining = self._prompt_cache.find_prefix(repo_id, prompt_tokens)

        if cached_kv is not None:
            num_cached = len(prompt_tokens) - len(remaining)
            # Prefill remaining tokens into the existing cache
            if remaining:
                arr = mx.array(remaining)[None]
                for s in range(0, arr.shape[1], self._PREFILL_CHUNK_SIZE):
                    lm.model(arr[:, s : s + self._PREFILL_CHUNK_SIZE], cache=cached_kv)
                mx.eval([c.state for c in cached_kv])
        else:
            num_cached = 0
            cached_kv = make_prompt_cache(lm.model)
            arr = mx.array(prompt_tokens)[None]
            for s in range(0, arr.shape[1], self._PREFILL_CHUNK_SIZE):
                lm.model(arr[:, s : s + self._PREFILL_CHUNK_SIZE], cache=cached_kv)
            mx.eval([c.state for c in cached_kv])

        # Store the fully-prefilled prompt cache (deep-copies internally)
        self._prompt_cache.store(repo_id, prompt_tokens, cached_kv)

        return cached_kv, num_cached

    def _build_mlx_kwargs(
        self, prompt: str, max_tokens: int,
        temperature: float, top_p: float,
        seed: int | None, repetition_penalty: float | None,
        draft_model: str | None, num_draft_tokens: int,
    ) -> dict[str, Any]:
        """Build shared kwargs dict for mlx-lm generate/stream calls."""
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            sampler = None

        kwargs: dict[str, Any] = {"prompt": prompt, "max_tokens": max_tokens}
        if sampler is not None:
            kwargs["sampler"] = sampler
        _apply_seed(seed)
        if repetition_penalty is not None and repetition_penalty != 1.0:
            try:
                from mlx_lm.sample_utils import make_logits_processors
                kwargs["logits_processors"] = make_logits_processors(
                    repetition_penalty=repetition_penalty,
                )
            except (ImportError, TypeError):
                pass
        if draft_model is not None:
            draft_lm = self.load(draft_model)
            kwargs["draft_model"] = draft_lm.model
            kwargs["num_draft_tokens"] = num_draft_tokens
        return kwargs

    def generate(
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
        reasoning_budget: int | None = None,
        strict_tools: bool = False,
        draft_model: str | None = None,
        num_draft_tokens: int = 5,
    ) -> GenerateResult:
        """
        Generate a response.
        Returns a ``GenerateResult`` (backward-compatible with 4-tuple unpacking).
        """
        from mlx_lm import generate as mlx_generate

        lm = self.load(repo_id)
        if max_tokens is None:
            max_tokens = _auto_max_tokens(lm)
        prompt = self._apply_chat_template(lm, messages, enable_thinking=enable_thinking, tools=tools, strict_tools=strict_tools)
        kwargs = self._build_mlx_kwargs(
            prompt, max_tokens, temperature, top_p,
            seed, repetition_penalty, draft_model, num_draft_tokens,
        )
        kwargs["verbose"] = False

        with self._generate_lock:
            cached_prompt_len = self._apply_prompt_cache(
                repo_id, lm, prompt, kwargs, draft_model=draft_model,
            )
            text = mlx_generate(lm.model, lm.tokenizer, **kwargs)

        # Normalize gemma4 channel tokens to standard think tags
        if _is_gemma4_model(lm.tokenizer):
            g4 = _Gemma4Normalizer()
            text = g4.feed(text) + g4.flush()

        # Estimate token counts (reuse cached count when available)
        try:
            prompt_tokens = cached_prompt_len if cached_prompt_len is not None else len(lm.tokenizer.encode(prompt))
            completion_tokens = len(lm.tokenizer.encode(text))
        except Exception:
            prompt_tokens = cached_prompt_len if cached_prompt_len is not None else len(prompt.split())
            completion_tokens = len(text.split())

        reasoning_tokens = 0
        if strip_thinking:
            text, reasoning = _strip_thinking(text)
            if reasoning is not None:
                try:
                    reasoning_tokens = len(lm.tokenizer.encode(reasoning))
                except Exception:
                    reasoning_tokens = len(reasoning.split())

            # If the model spent all tokens on thinking and returned an
            # empty answer, retry once with thinking disabled.
            if text == "" and reasoning is not None and enable_thinking:
                log.info("Empty answer after thinking — retrying with enable_thinking=False")
                retry_prompt = self._apply_chat_template(
                    lm,
                    messages,
                    enable_thinking=False,
                    tools=tools,
                    strict_tools=strict_tools,
                )
                # Build fresh kwargs — the original may carry a stale prompt_cache
                retry_kwargs = self._build_mlx_kwargs(
                    retry_prompt, max_tokens, temperature, top_p,
                    seed, repetition_penalty, draft_model, num_draft_tokens,
                )
                retry_kwargs["verbose"] = False
                with self._generate_lock:
                    text = mlx_generate(lm.model, lm.tokenizer, **retry_kwargs)
                try:
                    completion_tokens = len(lm.tokenizer.encode(text))
                except Exception:
                    completion_tokens = len(text.split())
                text, reasoning = _strip_thinking(text)
                reasoning_tokens = 0
        else:
            reasoning = None

        return GenerateResult(text, reasoning, prompt_tokens, completion_tokens, reasoning_tokens)

    def stream_generate(
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
        reasoning_budget: int | None = None,
        strict_tools: bool = False,
        draft_model: str | None = None,
        num_draft_tokens: int = 5,
    ) -> Iterator[str]:
        """
        Stream token-by-token generation.
        Yields text chunks. When strip_thinking=True (default), <think>...</think>
        blocks are silently consumed and not yielded.
        """
        from mlx_lm import stream_generate as mlx_stream

        lm = self.load(repo_id)
        if max_tokens is None:
            max_tokens = _auto_max_tokens(lm)
        prompt = self._apply_chat_template(lm, messages, enable_thinking=enable_thinking, tools=tools, strict_tools=strict_tools)
        kwargs = self._build_mlx_kwargs(
            prompt, max_tokens, temperature, top_p,
            seed, repetition_penalty, draft_model, num_draft_tokens,
        )

        _gemma4 = _is_gemma4_model(lm.tokenizer)

        if not strip_thinking:
            g4 = _Gemma4Normalizer() if _gemma4 else None
            with self._generate_lock:
                self._apply_prompt_cache(
                    repo_id, lm, prompt, kwargs, draft_model=draft_model,
                )
                for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
                    text = response if isinstance(response, str) else response.text
                    if g4:
                        token_id = getattr(response, "token", None)
                        normalized = g4.feed(text, token_id=token_id)
                        if normalized:
                            yield normalized
                    else:
                        yield text
            if g4:
                remainder = g4.flush()
                if remainder:
                    yield remainder
            return

        # State machine to strip thinking markers from streamed tokens.
        # Supports Qwen-style <think>...</think> and Gemma-style
        # <|channel>thought ... <channel|> markers.
        _THINK_PASSTHROUGH_TOKENS = 10_000  # chars before assuming model never closes the marker
        start_tags = ("<think>", "<|channel>thought")
        end_tags = ("</think>", "<channel|>")
        all_tags = start_tags + end_tags

        inside_think = bool(re.search(r"(<think>|<\|channel>thought)\s*$", prompt))
        buf = ""
        think_chars = 0  # chars suppressed while inside_think=True
        g4 = _Gemma4Normalizer() if _gemma4 else None

        def _partial_suffix_len() -> int:
            for tag in all_tags:
                for i in range(min(len(tag) - 1, len(buf)), 0, -1):
                    if buf.endswith(tag[:i]):
                        return i
            return 0

        self._generate_lock.acquire()
        try:
            self._apply_prompt_cache(
                repo_id, lm, prompt, kwargs, draft_model=draft_model,
            )
            for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
                chunk = response if isinstance(response, str) else response.text
                if g4:
                    token_id = getattr(response, "token", None)
                    normalized = g4.feed(chunk, token_id=token_id)
                    if not normalized:
                        continue
                    buf += normalized
                else:
                    buf += chunk

                while buf:
                    if inside_think:
                        end_positions = [(buf.find(tag), tag) for tag in end_tags if buf.find(tag) >= 0]
                        first_end = min(end_positions, default=(-1, ""), key=lambda x: x[0])
                        if first_end[0] >= 0:
                            # Properly closed — discard thinking content
                            think_chars = 0
                            buf = buf[first_end[0] + len(first_end[1]):]
                            inside_think = False
                            continue

                        keep = _partial_suffix_len()
                        suppressed = buf[:-keep] if keep else buf
                        think_chars += len(suppressed)
                        buf = buf[-keep:] if keep else ""

                        # If a reasoning_budget was set and we've exceeded it
                        # (rough char-to-token estimate: 4 chars per token), force
                        # the end of the thinking phase and start yielding text.
                        if reasoning_budget is not None and think_chars > reasoning_budget * 4:
                            inside_think = False
                            think_chars = 0
                            continue

                        # If we've suppressed too much without a close marker, this
                        # model doesn't use proper think-marker pairs — yield as plain text.
                        if think_chars > _THINK_PASSTHROUGH_TOKENS:
                            inside_think = False
                            think_chars = 0
                            if suppressed:
                                yield suppressed
                            continue
                        break
                    else:
                        start_positions = [(buf.find(tag), tag) for tag in start_tags if buf.find(tag) >= 0]
                        first_start = min(start_positions, default=(-1, ""), key=lambda x: x[0])
                        if first_start[0] >= 0:
                            # Yield everything before the marker
                            if first_start[0] > 0:
                                yield buf[:first_start[0]]
                            buf = buf[first_start[0] + len(first_start[1]):]
                            if first_start[1] == "<|channel>thought" and buf.startswith("\n"):
                                buf = buf[1:]
                            inside_think = True
                            continue

                        keep = _partial_suffix_len()
                        if keep:
                            safe = buf[:-keep]
                            if safe:
                                yield safe
                            buf = buf[-keep:]
                        else:
                            if buf:
                                yield buf
                            buf = ""
                        break
            # Flush remaining gemma4 normalization buffer
            if g4:
                remainder = g4.flush()
                if remainder:
                    buf += remainder
            # Flush any remaining buffer outside think blocks
            if buf and not inside_think:
                yield buf
        finally:
            self._generate_lock.release()

    def get_tokenizer(self, repo_id: str) -> Any:
        """Return the tokenizer for a model (loads if needed)."""
        return self.load(repo_id).tokenizer

    def unload(self, repo_id: str) -> bool:
        """Unload a specific model from cache. Returns True if it was loaded."""
        with self._lock:
            if repo_id not in self._models:
                return False
            del self._models[repo_id]
        self._prompt_cache.clear(repo_id)
        self._emit_event("unload", repo_id)
        return True

    def unload_all(self) -> None:
        """Unload all models from cache."""
        with self._lock:
            repo_ids = list(self._models.keys())
            self._models.clear()
        self._prompt_cache.clear()
        for repo_id in repo_ids:
            self._emit_event("unload", repo_id)

    def list_loaded(self) -> list[str]:
        """Return list of currently loaded model repo IDs."""
        with self._lock:
            return list(self._models.keys())

    def list_loaded_info(self) -> list[dict]:
        """Return detailed info for each loaded model (for ps command)."""
        now = time.monotonic()
        with self._lock:
            result = []
            for repo_id, lm in self._models.items():
                idle_seconds = now - lm.last_used_at
                info: dict = {
                    "repo_id": repo_id,
                    "loaded_at": lm.loaded_at,
                    "last_used": lm.last_used,
                    "idle_seconds": round(idle_seconds, 1),
                }
                if self._ttl_seconds > 0:
                    remaining = max(0, self._ttl_seconds - idle_seconds)
                    info["ttl_remaining_seconds"] = round(remaining, 1)
                result.append(info)
            return result


_engine_instance: TextEngine | None = None
_engine_lock = threading.Lock()


def get_engine(max_loaded: int = 2, ttl_seconds: int = 0,
               prompt_cache_limit: int = 4,
               kv_cache: "KVCacheConfig | None" = None) -> TextEngine:
    """Return the module-level singleton TextEngine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = TextEngine(
                    max_loaded=max_loaded,
                    ttl_seconds=ttl_seconds,
                    prompt_cache_limit=prompt_cache_limit,
                    kv_cache=kv_cache,
                )
    return _engine_instance


def reset_engine() -> None:
    """Reset singleton (for testing)."""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.stop_reaper()
        _engine_instance.unload_all()
    _engine_instance = None
