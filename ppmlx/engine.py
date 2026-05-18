from __future__ import annotations
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, NamedTuple

log = logging.getLogger("ppmlx.engine")


@dataclass
class LoadedModel:
    """A model that has been loaded into memory."""
    repo_id: str
    model: Any          # mlx_lm model object
    tokenizer: Any      # mlx_lm tokenizer object
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


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
        return "<think>" in template or "<|channel>thought" in template
    return False


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


def _resolve_model_path(repo_id: str) -> str:
    """Resolve a repo_id to a local path if available, otherwise return the repo_id."""
    try:
        from ppmlx.models import resolve_model_path
        return resolve_model_path(repo_id)
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

    Thread-safe: uses a lock for model loading/unloading.
    Multiple concurrent requests for the same loaded model are fine.
    """

    def __init__(self, max_loaded: int = 2):
        self._max_loaded = max_loaded
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()

    def _load_impl(self, repo_id: str) -> LoadedModel:
        """Actually load a model using mlx_lm.load. Called under lock."""
        from mlx_lm import load as mlx_load
        path = _resolve_model_path(repo_id)
        try:
            model, tokenizer = mlx_load(path)
        except ValueError as exc:
            if not _can_retry_gemma4_non_strict(path, exc):
                raise
            log.warning(
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

    def _get_or_load(self, repo_id: str) -> LoadedModel:
        """Get from cache or load; moves to end of LRU."""
        return self.load(repo_id)

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
            if not enable_thinking:
                try:
                    return tokenizer.apply_chat_template(
                        messages, **kwargs, enable_thinking=False
                    )
                except TypeError:
                    pass  # model doesn't support the flag — fall through
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Tokenizer doesn't support 'tools' kwarg — retry without it
                if tools and "tools" in kwargs:
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
    ) -> GenerateResult:
        """
        Generate a response.
        Returns a ``GenerateResult`` (backward-compatible with 4-tuple unpacking).

        reasoning_text is populated for <think>...</think> models.
        prompt_tokens and completion_tokens are estimates (token count from encode).
        enable_thinking=False suppresses the thinking phase for models that support it (e.g. Qwen3).
        max_tokens=None means 50% of the model's context window (capped at _MAX_AUTO_TOKENS,
        or _MAX_AUTO_TOKENS_THINKING for thinking models).
        reasoning_budget limits how many tokens the model may spend on reasoning.
        """
        from mlx_lm import generate as mlx_generate

        lm = self._get_or_load(repo_id)
        if max_tokens is None:
            max_tokens = _auto_max_tokens(lm)
        prompt = self._apply_chat_template(lm, messages, enable_thinking=enable_thinking, tools=tools)

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            sampler = None
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": False,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if seed is not None:
            kwargs["seed"] = seed
        if repetition_penalty is not None and repetition_penalty != 1.0:
            try:
                from mlx_lm.sample_utils import make_logits_processors
                kwargs["logits_processors"] = make_logits_processors(
                    repetition_penalty=repetition_penalty,
                )
            except (ImportError, TypeError):
                pass

        text = mlx_generate(lm.model, lm.tokenizer, **kwargs)

        # Estimate token counts
        try:
            prompt_tokens = len(lm.tokenizer.encode(prompt))
            completion_tokens = len(lm.tokenizer.encode(text))
        except Exception:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(text.split())

        reasoning_tokens = 0
        if strip_thinking:
            text, reasoning = _strip_thinking(text)
            if reasoning is not None:
                try:
                    reasoning_tokens = len(lm.tokenizer.encode(reasoning))
                except Exception:
                    reasoning_tokens = len(reasoning.split())

            # Bug fix: if the model spent all tokens on thinking and returned an
            # empty answer, retry once with thinking disabled.
            if text == "" and reasoning is not None and enable_thinking:
                log.info("Empty answer after thinking — retrying with enable_thinking=False")
                retry_prompt = self._apply_chat_template(
                    lm, messages, enable_thinking=False, tools=tools,
                )
                retry_kwargs = {**kwargs, "prompt": retry_prompt}
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
    ) -> Iterator[str]:
        """
        Stream token-by-token generation.
        Yields text chunks. When strip_thinking=True (default), <think>...</think>
        blocks are silently consumed and not yielded.
        enable_thinking=False suppresses the thinking phase for models that support it (e.g. Qwen3).
        max_tokens=None means 50% of the model's context window (capped at _MAX_AUTO_TOKENS,
        or _MAX_AUTO_TOKENS_THINKING for thinking models).
        reasoning_budget limits how many tokens (approximate) the model may spend on reasoning.
        """
        from mlx_lm import stream_generate as mlx_stream

        lm = self._get_or_load(repo_id)
        if max_tokens is None:
            max_tokens = _auto_max_tokens(lm)
        prompt = self._apply_chat_template(lm, messages, enable_thinking=enable_thinking, tools=tools)

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            sampler = None
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if seed is not None:
            kwargs["seed"] = seed
        if repetition_penalty is not None and repetition_penalty != 1.0:
            try:
                from mlx_lm.sample_utils import make_logits_processors
                kwargs["logits_processors"] = make_logits_processors(
                    repetition_penalty=repetition_penalty,
                )
            except (ImportError, TypeError):
                pass

        if not strip_thinking:
            for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
                if hasattr(response, "text"):
                    yield response.text
                elif isinstance(response, str):
                    yield response
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

        def _partial_suffix_len() -> int:
            for tag in all_tags:
                for i in range(min(len(tag) - 1, len(buf)), 0, -1):
                    if buf.endswith(tag[:i]):
                        return i
            return 0

        for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
            chunk = response.text if hasattr(response, "text") else response
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

        # Flush any remaining buffer outside think blocks
        if buf and not inside_think:
            yield buf

    def get_tokenizer(self, repo_id: str) -> Any:
        """Return the tokenizer for a model (loads if needed)."""
        return self._get_or_load(repo_id).tokenizer

    def unload(self, repo_id: str) -> bool:
        """Unload a specific model from cache. Returns True if it was loaded."""
        with self._lock:
            if repo_id not in self._models:
                return False
            del self._models[repo_id]
        self._emit_event("unload", repo_id)
        return True

    def unload_all(self) -> None:
        """Unload all models from cache."""
        with self._lock:
            repo_ids = list(self._models.keys())
            self._models.clear()
        for repo_id in repo_ids:
            self._emit_event("unload", repo_id)

    def list_loaded(self) -> list[str]:
        """Return list of currently loaded model repo IDs."""
        with self._lock:
            return list(self._models.keys())


_engine_instance: TextEngine | None = None
_engine_lock = threading.Lock()


def get_engine(max_loaded: int = 2) -> TextEngine:
    """Return the module-level singleton TextEngine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = TextEngine(max_loaded=max_loaded)
    return _engine_instance


def reset_engine() -> None:
    """Reset singleton (for testing)."""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.unload_all()
    _engine_instance = None
