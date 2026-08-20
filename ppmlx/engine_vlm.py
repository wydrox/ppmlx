from __future__ import annotations
import base64
import tempfile
import threading
from pathlib import Path
from typing import Any


def _resolve_model_path(repo_id: str) -> str:
    """Resolve alias to local path if available."""
    try:
        from ppmlx.models import resolve_model_path
        return resolve_model_path(repo_id)
    except ImportError:
        return repo_id


class VisionEngine:
    """Wraps mlx-vlm for multimodal (image+text) generation."""

    def __init__(self):
        self._models: dict[str, tuple[Any, Any]] = {}  # repo_id → (model, processor)
        self._lock = threading.Lock()

    def load(self, repo_id: str) -> None:
        """Load a vision model using mlx_vlm.load()."""
        if repo_id in self._models:
            return
        path = _resolve_model_path(repo_id)
        from mlx_vlm import load as vlm_load
        with self._lock:
            if repo_id not in self._models:
                try:
                    model, processor = vlm_load(path)
                except ImportError as e:
                    msg = str(e)
                    if "torchvision" in msg or "torch" in msg:
                        raise RuntimeError(
                            f"Vision processor for '{repo_id}' requires PyTorch, but this may be "
                            f"an unnecessary dependency — mlx-vlm should handle vision natively on Apple Silicon.\n"
                            f"Workaround: pip install torch torchvision\n"
                            f"If this is a text-only model, send requests without images to use the MLX text engine instead."
                        ) from e
                    raise
                self._models[repo_id] = (model, processor)

    def _extract_images(
        self, messages: list[dict], *, allow_local_paths: bool = False,
    ) -> list[str | bytes]:
        """
        Extract image references from message content.
        Returns local paths or decoded image bytes.

        When ``allow_local_paths`` is False (default, for API requests),
        ``file://`` URLs and bare filesystem paths are rejected to prevent
        local file read attacks.  The CLI REPL sets ``allow_local_paths=True``.
        """
        images: list[str | bytes] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        if url.startswith("data:image/"):
                            try:
                                _, data = url.split(",", 1)
                                images.append(base64.b64decode(data, validate=True))
                            except Exception:
                                pass
                        elif url.startswith("file://"):
                            if allow_local_paths:
                                images.append(url[7:])
                            # else: silently skip — do not expose local files
                        elif url and (url.startswith("/") or url.startswith("~")):
                            if allow_local_paths:
                                images.append(str(Path(url).expanduser()))
                            # else: silently skip
                        elif url and (url.startswith("http://") or url.startswith("https://")):
                            raise ValueError("Remote image URLs are not supported")
                        # else: skip unknown schemes
        return images

    def generate(
        self,
        repo_id: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> tuple[str, int, int]:
        """
        Generate a response for a vision request.
        Returns (text, prompt_tokens, completion_tokens).
        """
        from mlx_vlm import generate as vlm_generate

        self.load(repo_id)
        model, processor = self._models[repo_id]

        images = self._extract_images(messages)

        text_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
        prompt = "\n".join(text_parts)

        with tempfile.TemporaryDirectory(prefix="ppmlx-vlm-") as temp_dir:
            image = images[0] if images else None
            if isinstance(image, bytes):
                image_path = Path(temp_dir) / "image.jpg"
                image_path.write_bytes(image)
                image = str(image_path)
            output = vlm_generate(
                model,
                processor,
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

        prompt_tokens = len(prompt.split())
        completion_tokens = len(str(output).split())

        return str(output), prompt_tokens, completion_tokens

    def get_tokenizer(self, repo_id: str) -> Any:
        """Return the tokenizer that belongs to a loaded vision processor."""
        self.load(repo_id)
        _, processor = self._models[repo_id]
        return getattr(processor, "tokenizer", processor)

    def list_loaded(self) -> list[str]:
        return list(self._models.keys())

    def unload_all(self) -> None:
        with self._lock:
            self._models.clear()


_vlm_engine: VisionEngine | None = None
_vlm_lock = threading.Lock()


def get_vision_engine() -> VisionEngine:
    global _vlm_engine
    if _vlm_engine is None:
        with _vlm_lock:
            if _vlm_engine is None:
                _vlm_engine = VisionEngine()
    return _vlm_engine
