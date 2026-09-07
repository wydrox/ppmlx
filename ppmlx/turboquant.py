"""TurboQuant KV-cache compression via PolarQuant + QJL.

Implements the two-stage compression algorithm from:
    TurboQuant: Redefining AI Efficiency with Extreme Compression
    (ICLR 2026, arxiv 2504.19874)

Stage 1 — PolarQuant: random orthogonal rotation + uniform quantization of
    normalized components.  Captures magnitude and direction in 2-4 bits.
Stage 2 — QJL (Quantized Johnson-Lindenstrauss): 1-bit sign projection of
    residual errors for bias-free correction.

Usage: wrap an mlx_lm ``KVCache`` with ``TurboQuantCache.wrap(cache, bits=3)``
to transparently compress keys and values stored in the cache.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("ppmlx.turboquant")


def _row_norms(x: Any, keepdims: bool = True) -> Any:
    """Per-row L2 norms, clamped to avoid division by zero."""
    import mlx.core as mx
    n = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=keepdims))
    return mx.maximum(n, 1e-8)


# ---------------------------------------------------------------------------
# Lazy-initialized random matrices (deterministic, keyed by dimension)
# ---------------------------------------------------------------------------
_rotation_matrices: dict[int, Any] = {}
_jl_matrices: dict[tuple[int, int], Any] = {}
_SEED = 31415  # fixed seed for reproducibility


def _get_rotation_matrix(head_dim: int) -> Any:
    """Return a deterministic orthogonal rotation matrix for *head_dim*."""
    if head_dim not in _rotation_matrices:
        import mlx.core as mx
        key = mx.random.key(_SEED)
        G = mx.random.normal(shape=(head_dim, head_dim), key=key)
        # QR decomposition must run on CPU in MLX (not yet supported on GPU)
        try:
            Q, _R = mx.linalg.qr(G, stream=mx.Device(mx.cpu))
        except (TypeError, AttributeError):
            # Fallback for test stubs or older MLX versions
            Q, _R = mx.linalg.qr(G)
        mx.eval(Q)
        _rotation_matrices[head_dim] = Q
    return _rotation_matrices[head_dim]


def _get_jl_matrix(head_dim: int, proj_dim: int) -> Any:
    """Return a deterministic Rademacher JL projection matrix."""
    k = (head_dim, proj_dim)
    if k not in _jl_matrices:
        import mlx.core as mx
        key = mx.random.key(_SEED + head_dim + proj_dim)
        # Rademacher: ±1 / sqrt(proj_dim)
        bits = mx.random.bernoulli(shape=(head_dim, proj_dim), key=key)
        P = mx.where(bits, 1.0, -1.0) / math.sqrt(proj_dim)
        P = P.astype(mx.float16)
        mx.eval(P)
        _jl_matrices[k] = P
    return _jl_matrices[k]


# ---------------------------------------------------------------------------
# PolarQuant — Stage 1
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PolarQuantData:
    """Compressed representation from PolarQuant."""
    quantized: Any      # mx.array uint8 (N, head_dim)
    norms: Any          # mx.array float16 (N,)
    scale: float
    zero_point: float
    bits: int
    batch_dims: tuple[int, ...]  # e.g. (B, n_heads) — dims before seq
    head_dim: int
    seq_len: int          # number of sequence tokens compressed


def polar_quantize(vectors: Any, R: Any, bits: int = 3) -> PolarQuantData:
    """Quantize vectors via rotation + uniform component quantization.

    Args:
        vectors: (B, n_heads, seq_len, head_dim) — raw key or value vectors.
        R: (head_dim, head_dim) orthogonal rotation matrix.
        bits: quantization depth (2, 3, or 4).

    Returns:
        PolarQuantData with compressed representation.
    """
    import mlx.core as mx

    orig_shape = vectors.shape
    head_dim = orig_shape[-1]
    seq_len = orig_shape[-2]
    batch_dims = orig_shape[:-2]  # (B, n_heads)

    # Flatten to 2-D for batch processing
    flat = vectors.reshape(-1, head_dim).astype(mx.float32)

    # 1. Rotate
    rotated = flat @ R  # (N, head_dim)

    # 2. Compute per-vector norms
    norms = _row_norms(rotated)  # (N, 1)

    # 3. Normalize
    normalized = rotated / norms  # values in [-1, 1]

    # 4. Uniform quantization of normalized components
    n_levels = (1 << bits) - 1  # e.g. 7 for 3-bit
    # Map [-1, 1] -> [0, n_levels]
    scale = 2.0 / n_levels
    zero_point = -1.0
    quantized = mx.round((normalized - zero_point) / scale)
    quantized = mx.clip(quantized, 0, n_levels).astype(mx.uint8)

    norms_f16 = norms.squeeze(-1).astype(mx.float16)

    return PolarQuantData(
        quantized=quantized,
        norms=norms_f16,
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        batch_dims=tuple(batch_dims),
        head_dim=head_dim,
        seq_len=seq_len,
    )


def polar_dequantize(data: PolarQuantData, R: Any) -> Any:
    """Reconstruct vectors from PolarQuant compressed data.

    Returns tensor with shape (*batch_dims, seq_len, head_dim).
    """
    import mlx.core as mx

    quantized = data.quantized.astype(mx.float32)

    # Reverse quantization: [0, n_levels] -> [-1, 1]
    normalized = quantized * data.scale + data.zero_point

    # Restore norms
    norms = data.norms.astype(mx.float32).reshape(-1, 1)  # (N, 1)
    rotated = normalized * norms

    # Inverse rotation (R is orthogonal, so R^-1 = R^T)
    flat = rotated @ R.T

    out_shape = (*data.batch_dims, data.seq_len, data.head_dim)
    return flat.reshape(out_shape).astype(mx.float16)


# ---------------------------------------------------------------------------
# QJL — Stage 2 (1-bit error correction)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QJLData:
    """Packed sign bits from JL projection of residuals."""
    signs: Any        # mx.array uint8 — packed sign bits
    norms: Any        # mx.array float16 — per-vector residual norms
    proj_dim: int
    n_vectors: int


def qjl_compress(residuals: Any, P: Any) -> QJLData:
    """Project residuals with JL matrix and store signs only.

    Args:
        residuals: (..., head_dim) — reconstruction error from PolarQuant.
        P: (head_dim, proj_dim) — JL projection matrix.

    Returns:
        QJLData with packed sign bits and residual norms.
    """
    import mlx.core as mx

    head_dim = residuals.shape[-1]
    flat = residuals.reshape(-1, head_dim).astype(mx.float32)
    n_vectors = flat.shape[0]

    # Store residual norms for magnitude reconstruction
    norms = _row_norms(flat)  # (N, 1)

    # Project
    projected = flat @ P  # (N, proj_dim)

    # Pack signs: 1 for positive, 0 for negative
    positive = (projected > 0.0).astype(mx.uint8)

    return QJLData(
        signs=positive, norms=norms.squeeze(-1).astype(mx.float16),
        proj_dim=P.shape[1], n_vectors=n_vectors,
    )


def qjl_decompress(data: QJLData, P: Any) -> Any:
    """Approximate residual reconstruction from packed sign bits."""
    import mlx.core as mx

    # Unpack: 0 -> -1, 1 -> +1
    signs = data.signs.astype(mx.float32) * 2.0 - 1.0  # (N, proj_dim)

    # Reconstruct direction: signs @ P^T
    direction = signs @ P.T  # (N, head_dim)

    # Normalize direction and scale by stored residual norms
    direction = direction / _row_norms(direction)

    norms = data.norms.astype(mx.float32).reshape(-1, 1)
    reconstructed = direction * norms

    return reconstructed


# ---------------------------------------------------------------------------
# Combined compress / decompress
# ---------------------------------------------------------------------------

def compress(vectors: Any, bits: int = 3, qjl: bool = True,
             qjl_dim: int = 0) -> tuple[PolarQuantData, QJLData | None]:
    """Full TurboQuant compression pipeline."""
    head_dim = vectors.shape[-1]
    R = _get_rotation_matrix(head_dim)

    pq = polar_quantize(vectors, R, bits=bits)

    jq = None
    if qjl:
        # Compute residual
        reconstructed = polar_dequantize(pq, R)
        residual = vectors.astype(reconstructed.dtype) - reconstructed
        proj_dim = qjl_dim if qjl_dim > 0 else max(head_dim // 2, 16)
        P = _get_jl_matrix(head_dim, proj_dim)
        jq = qjl_compress(residual, P)

    return pq, jq


def decompress(pq: PolarQuantData, jq: QJLData | None) -> Any:
    """Full TurboQuant decompression pipeline."""
    import mlx.core as mx

    R = _get_rotation_matrix(pq.head_dim)

    result = polar_dequantize(pq, R)

    if jq is not None:
        P = _get_jl_matrix(pq.head_dim, jq.proj_dim)
        correction = qjl_decompress(jq, P)
        flat = result.reshape(-1, pq.head_dim) + correction.astype(mx.float16)
        out_shape = (*pq.batch_dims, pq.seq_len, pq.head_dim)
        result = flat.reshape(out_shape)

    return result


# ---------------------------------------------------------------------------
# TurboQuantCache — drop-in KVCache wrapper
# ---------------------------------------------------------------------------

# Minimum tokens before compression kicks in (single-token generation
# steps are kept uncompressed in a "recent" buffer).
_COMPRESS_THRESHOLD = 64


class TurboQuantCache:
    """Drop-in wrapper around an mlx_lm ``KVCache`` that transparently
    compresses stored keys and values with TurboQuant.

    The wrapper delegates attribute access to the inner cache, but intercepts
    ``update_and_fetch`` to compress older tokens.
    """

    def __init__(
        self,
        inner: Any,
        bits: int = 3,
        qjl: bool = True,
        qjl_dim: int = 0,
    ) -> None:
        self._inner = inner
        # Named tq_bits/tq_qjl to avoid collision with QuantizedKVCache.bits
        # (mlx_lm's scaled_dot_product_attention checks hasattr(cache, "bits"))
        self.tq_bits = bits
        self.tq_qjl = qjl
        self.tq_qjl_dim = qjl_dim

        # Compressed chunks (appended per compression round, avoids O(n²) merge)
        self._k_chunks: list[tuple[PolarQuantData, QJLData | None]] = []
        self._v_chunks: list[tuple[PolarQuantData, QJLData | None]] = []

        # Cached decompressed output of compressed chunks (invalidated on new compression)
        self._k_decompressed: Any | None = None
        self._v_decompressed: Any | None = None

        # Recent (uncompressed) buffer — not yet worth compressing
        self._k_recent: Any | None = None  # mx.array
        self._v_recent: Any | None = None  # mx.array

        self._total_len = 0  # total tokens stored (compressed + recent)

    # -- core cache protocol -----------------------------------------------

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        """Store new KV and return the full (decompressed) KV sequence."""
        import mlx.core as mx

        # Append to recent buffer
        if self._k_recent is None:
            self._k_recent = keys
            self._v_recent = values
        else:
            assert self._v_recent is not None
            self._k_recent = mx.concatenate([self._k_recent, keys], axis=2)
            self._v_recent = mx.concatenate([self._v_recent, values], axis=2)

        self._total_len += keys.shape[2]

        # Compress recent buffer when it exceeds threshold
        recent_len = self._k_recent.shape[2]
        if recent_len >= _COMPRESS_THRESHOLD:
            self._compress_recent()

        # Build full decompressed output
        return self._fetch_all()

    def _compress_recent(self) -> None:
        """Move recent buffer into compressed storage."""
        import mlx.core as mx

        k_pq, k_jq = compress(self._k_recent, self.tq_bits, self.tq_qjl, self.tq_qjl_dim)
        v_pq, v_jq = compress(self._v_recent, self.tq_bits, self.tq_qjl, self.tq_qjl_dim)

        self._k_chunks.append((k_pq, k_jq))
        self._v_chunks.append((v_pq, v_jq))

        # Invalidate decompressed cache — will be rebuilt on next _fetch_all
        self._k_decompressed = None
        self._v_decompressed = None

        self._k_recent = None
        self._v_recent = None
        mx.eval([k_pq.quantized, k_pq.norms, v_pq.quantized, v_pq.norms])

    def _fetch_all(self) -> tuple[Any, Any]:
        """Return fully decompressed keys and values."""
        import mlx.core as mx

        parts_k: list[Any] = []
        parts_v: list[Any] = []

        if self._k_chunks:
            # Reuse cached decompressed output if available
            if self._k_decompressed is None:
                dk = [decompress(pq, jq) for pq, jq in self._k_chunks]
                dv = [decompress(pq, jq) for pq, jq in self._v_chunks]
                self._k_decompressed = mx.concatenate(dk, axis=2) if len(dk) > 1 else dk[0]
                self._v_decompressed = mx.concatenate(dv, axis=2) if len(dv) > 1 else dv[0]
            parts_k.append(self._k_decompressed)
            parts_v.append(self._v_decompressed)

        if self._k_recent is not None:
            parts_k.append(self._k_recent)
            parts_v.append(self._v_recent)

        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    # -- cache metadata protocol (delegated to inner) ----------------------

    @property
    def offset(self) -> int:
        return self._total_len

    @offset.setter
    def offset(self, v: int) -> None:
        # mlx_lm sets offset externally; we track length internally via _total_len
        pass

    @property
    def state(self) -> Any:
        """Return state tuple for deep-copy / serialization."""
        return (
            self._k_chunks, self._v_chunks,
            self._k_recent, self._v_recent,
            self._total_len,
            self.tq_bits, self.tq_qjl, self.tq_qjl_dim,
        )

    @state.setter
    def state(self, v: Any) -> None:
        (
            self._k_chunks, self._v_chunks,
            self._k_recent, self._v_recent,
            self._total_len,
            self.tq_bits, self.tq_qjl, self.tq_qjl_dim,
        ) = v
        self._k_decompressed = None
        self._v_decompressed = None

    @property
    def meta_state(self) -> Any:
        return getattr(self._inner, "meta_state", None)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Trim the last *n* tokens from the cache."""
        if n <= 0:
            return 0
        if self._k_recent is not None and self._k_recent.shape[2] >= n:
            assert self._v_recent is not None
            self._k_recent = self._k_recent[:, :, :-n, :]
            self._v_recent = self._v_recent[:, :, :-n, :]
            if self._k_recent.shape[2] == 0:
                self._k_recent = None
                self._v_recent = None
            self._total_len -= n
            return n
        # Cannot trim into compressed region easily — fall back to 0
        return 0

    @property
    def nbytes(self) -> int:
        """Estimated memory usage of compressed data."""
        total = 0
        for chunks in (self._k_chunks, self._v_chunks):
            for pq, jq in chunks:
                total += getattr(pq.quantized, "nbytes", 0)
                total += getattr(pq.norms, "nbytes", 0)
                if jq is not None:
                    total += getattr(jq.signs, "nbytes", 0)
        for arr in (self._k_recent, self._v_recent):
            if arr is not None:
                total += getattr(arr, "nbytes", 0)
        return total

    def empty(self) -> bool:
        return not self._k_chunks and self._k_recent is None

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to inner cache."""
        return getattr(self._inner, name)

    def __deepcopy__(self, memo: dict) -> "TurboQuantCache":
        new = TurboQuantCache.__new__(TurboQuantCache)
        new._inner = copy.deepcopy(self._inner, memo)
        new.tq_bits = self.tq_bits
        new.tq_qjl = self.tq_qjl
        new.tq_qjl_dim = self.tq_qjl_dim
        new._k_chunks = copy.deepcopy(self._k_chunks, memo)
        new._v_chunks = copy.deepcopy(self._v_chunks, memo)
        new._k_decompressed = None  # will be rebuilt on demand
        new._v_decompressed = None
        new._k_recent = copy.deepcopy(self._k_recent, memo)
        new._v_recent = copy.deepcopy(self._v_recent, memo)
        new._total_len = self._total_len
        return new

    @classmethod
    def wrap(
        cls,
        cache: Any,
        bits: int = 3,
        qjl: bool = True,
        qjl_dim: int = 0,
    ) -> "TurboQuantCache":
        """Wrap an existing KVCache.  Returns *cache* unchanged for
        RotatingKVCache (incompatible sliding-window semantics).
        """
        try:
            from mlx_lm.models.cache import RotatingKVCache
            if isinstance(cache, RotatingKVCache):
                return cache  # type: ignore[return-value]
        except ImportError:
            pass
        return cls(inner=cache, bits=bits, qjl=qjl, qjl_dim=qjl_dim)
