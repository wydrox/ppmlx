from __future__ import annotations
import math
import threading
from typing import Any, cast


def _resolve_model_path(repo_id: str) -> str:
    try:
        from ppmlx.models import resolve_model_path
        return resolve_model_path(repo_id)
    except ImportError:
        return repo_id


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class EmbedEngine:
    """Wraps mlx-embeddings for text embedding."""

    def __init__(self):
        self._models: dict[str, tuple[Any, Any]] = {}  # repo_id → (model, tokenizer)
        self._lock = threading.Lock()

    def load(self, repo_id: str) -> None:
        """Load an embedding model."""
        if repo_id in self._models:
            return
        path = _resolve_model_path(repo_id)
        try:
            from mlx_embeddings.utils import load as embed_load
        except ImportError:
            raise ImportError(
                "Embeddings require the 'mlx-embeddings' package.\n"
                "Install it with: uv tool install ppmlx[embeddings]  (or: pip install ppmlx[embeddings])"
            )
        with self._lock:
            if repo_id not in self._models:
                model, tokenizer = embed_load(path)
                self._models[repo_id] = (model, tokenizer)

    def encode(
        self,
        repo_id: str,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Encode a list of texts into embedding vectors.
        Returns list of float vectors, optionally L2-normalized.
        """
        self.load(repo_id)
        model, tokenizer = self._models[repo_id]

        inputs = tokenizer(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,
        )

        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

        # mlx-embeddings models may return last_hidden_state (needs pooling),
        # pooler_output, or the embedding tensor directly.
        import mlx.core as mx

        if hasattr(outputs, "last_hidden_state"):
            # Mean pooling
            hidden = outputs.last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is not None:
                mask_expanded = mx.expand_dims(mask, -1).astype(hidden.dtype)
                summed = mx.sum(hidden * mask_expanded, axis=1)
                counts = mx.sum(mask_expanded, axis=1)
                embeddings = summed / counts
            else:
                embeddings = mx.mean(hidden, axis=1)
        elif hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            # Assume outputs is directly the embedding tensor
            embeddings = outputs

        result: list[list[float]] = []
        for i in range(len(texts)):
            vec: list[float] = cast(list[float], embeddings[i].tolist())
            if normalize:
                vec = _l2_normalize(vec)
            result.append(vec)
        return result

    def list_loaded(self) -> list[str]:
        return list(self._models.keys())

    def unload_all(self) -> None:
        with self._lock:
            self._models.clear()


_embed_engine: EmbedEngine | None = None
_embed_lock = threading.Lock()


def get_embed_engine() -> EmbedEngine:
    global _embed_engine
    if _embed_engine is None:
        with _embed_lock:
            if _embed_engine is None:
                _embed_engine = EmbedEngine()
    return _embed_engine
