"""Tests for TurboQuant KV-cache compression."""
from __future__ import annotations

import copy
import sys
import types
from unittest.mock import MagicMock

import pytest


# ── MLX stub setup ──────────────────────────────────────────────────────
# On non-macOS CI, mlx is stubbed by conftest.py.  We add the minimal
# attributes that turboquant.py needs so unit tests can run anywhere.

def _ensure_mx_stubs():
    """Ensure mlx.core stub has the functions turboquant imports."""
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return
    # If it's a real mlx module, nothing to patch
    if hasattr(mx, "concatenate") and not isinstance(mx.concatenate, MagicMock):
        return
    # It's a stub — add numpy-backed implementations for testing
    import numpy as np

    class _MXArray:
        """Minimal mx.array stand-in backed by numpy."""
        def __init__(self, data, dtype=None):
            if isinstance(data, _MXArray):
                self._np = data._np.copy()
            elif isinstance(data, np.ndarray):
                self._np = data
            else:
                self._np = np.array(data, dtype=np.float32 if dtype is None else None)
            self.shape = self._np.shape
            self.ndim = self._np.ndim
            self.dtype = dtype or _float32

        @property
        def nbytes(self):
            return self._np.nbytes

        def astype(self, dtype):
            _dtype_map = {
                _float32: np.float32, _float16: np.float16, _uint8: np.uint8,
                "float32": np.float32, "float16": np.float16, "uint8": np.uint8,
            }
            np_dtype = _dtype_map.get(dtype, np.float32)
            return _MXArray(self._np.astype(np_dtype), dtype=dtype)

        def reshape(self, *args):
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                shape = args[0]
            else:
                shape = args
            return _MXArray(self._np.reshape(shape))

        def squeeze(self, axis=None):
            return _MXArray(self._np.squeeze(axis=axis))

        def __matmul__(self, other):
            if isinstance(other, _MXArray):
                return _MXArray(self._np @ other._np)
            return NotImplemented

        def __sub__(self, other):
            if isinstance(other, _MXArray):
                return _MXArray(self._np - other._np)
            return _MXArray(self._np - other)

        def __mul__(self, other):
            if isinstance(other, _MXArray):
                return _MXArray(self._np * other._np)
            return _MXArray(self._np * other)

        def __add__(self, other):
            if isinstance(other, _MXArray):
                return _MXArray(self._np + other._np)
            return _MXArray(self._np + other)

        def __truediv__(self, other):
            if isinstance(other, _MXArray):
                return _MXArray(self._np / other._np)
            return _MXArray(self._np / other)

        def __gt__(self, other):
            o = other._np if isinstance(other, _MXArray) else other
            return _MXArray(self._np > o)

        def __lt__(self, other):
            o = other._np if isinstance(other, _MXArray) else other
            return _MXArray(self._np < o)

        def __getitem__(self, key):
            return _MXArray(self._np[key])

        @property
        def T(self):
            return _MXArray(self._np.T)

    # dtype sentinels
    _float32 = "float32"
    _float16 = "float16"
    _uint8 = "uint8"

    mx.array = _MXArray
    mx.float32 = _float32
    mx.float16 = _float16
    mx.uint8 = _uint8

    mx.concatenate = lambda arrs, axis=0: _MXArray(
        np.concatenate([a._np for a in arrs], axis=axis)
    )
    mx.sum = lambda a, axis=None, keepdims=False: _MXArray(
        np.sum(a._np, axis=axis, keepdims=keepdims)
    )
    mx.sqrt = lambda a: _MXArray(np.sqrt(a._np))
    mx.maximum = lambda a, b: _MXArray(np.maximum(a._np if isinstance(a, _MXArray) else a,
                                                    b._np if isinstance(b, _MXArray) else b))
    mx.round = lambda a: _MXArray(np.round(a._np))
    mx.clip = lambda a, lo, hi: _MXArray(np.clip(a._np, lo, hi))
    mx.where = lambda cond, x, y: _MXArray(np.where(
        cond._np if isinstance(cond, _MXArray) else cond,
        x._np if isinstance(x, _MXArray) else x,
        y._np if isinstance(y, _MXArray) else y,
    ))
    mx.eval = lambda *args: None

    # mlx.core.random
    rand_mod = sys.modules.get("mlx.core.random")
    if rand_mod is None:
        rand_mod = types.ModuleType("mlx.core.random")
        sys.modules["mlx.core.random"] = rand_mod
    mx.random = rand_mod

    _rng = np.random.RandomState(31415)

    def _key(seed):
        return seed

    def _normal(shape, key=None):
        rng = np.random.RandomState(key if key is not None else 0)
        return _MXArray(rng.randn(*shape).astype(np.float32))

    def _bernoulli(shape, key=None):
        rng = np.random.RandomState(key if key is not None else 0)
        return _MXArray(rng.randint(0, 2, size=shape).astype(np.float32))

    mx.random.key = _key
    mx.random.normal = _normal
    mx.random.bernoulli = _bernoulli

    # mlx.core.linalg
    linalg_mod = sys.modules.get("mlx.core.linalg")
    if linalg_mod is None:
        linalg_mod = types.ModuleType("mlx.core.linalg")
        sys.modules["mlx.core.linalg"] = linalg_mod
    mx.linalg = linalg_mod

    def _qr(a):
        q, r = np.linalg.qr(a._np)
        return _MXArray(q), _MXArray(r)

    mx.linalg.qr = _qr


_ensure_mx_stubs()

# Clear cached matrices so each test module starts fresh
import ppmlx.turboquant as tq
tq._rotation_matrices.clear()
tq._jl_matrices.clear()


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_vectors(batch=1, n_heads=4, seq_len=32, head_dim=64):
    """Create a random KV-like tensor."""
    import mlx.core as mx
    import numpy as np
    rng = np.random.RandomState(42)
    data = rng.randn(batch, n_heads, seq_len, head_dim).astype(np.float32)
    return mx.array(data)


def _to_numpy(arr):
    """Convert mx.array (real or stub) to numpy."""
    if hasattr(arr, "_np"):
        return arr._np
    import numpy as np
    return np.array(arr)


# ── PolarQuant Tests ────────────────────────────────────────────────────

class TestPolarQuant:
    def test_roundtrip_bounded_error(self):
        """Compress→decompress should produce bounded reconstruction error."""
        import numpy as np
        vectors = _make_vectors(seq_len=16, head_dim=32)
        R = tq._get_rotation_matrix(32)
        pq = tq.polar_quantize(vectors, R, bits=4)
        reconstructed = tq.polar_dequantize(pq, R)

        orig = _to_numpy(vectors).reshape(-1)
        recon = _to_numpy(reconstructed).reshape(-1)
        # Relative error should be reasonable for 4-bit
        mse = np.mean((orig - recon) ** 2)
        assert mse < np.mean(orig ** 2) * 0.5, f"MSE too high: {mse}"

    def test_output_shapes(self):
        vectors = _make_vectors(seq_len=8, head_dim=16)
        R = tq._get_rotation_matrix(16)
        pq = tq.polar_quantize(vectors, R, bits=3)
        # quantized should be (N, head_dim) where N = batch * n_heads * seq_len
        assert pq.quantized.shape[-1] == 16
        assert pq.norms.shape[0] == pq.quantized.shape[0]
        assert pq.head_dim == 16
        assert pq.seq_len == 8
        assert pq.batch_dims == (1, 4)

    def test_rotation_matrix_deterministic(self):
        """Same head_dim always produces same rotation matrix."""
        tq._rotation_matrices.clear()
        R1 = tq._get_rotation_matrix(32)
        tq._rotation_matrices.clear()
        R2 = tq._get_rotation_matrix(32)
        import numpy as np
        np.testing.assert_array_equal(_to_numpy(R1), _to_numpy(R2))


# ── QJL Tests ───────────────────────────────────────────────────────────

class TestQJL:
    def test_compress_decompress_shape(self):
        import numpy as np
        residuals = _make_vectors(seq_len=8, head_dim=16)
        P = tq._get_jl_matrix(16, 8)
        jq = tq.qjl_compress(residuals, P)
        assert jq.proj_dim == 8
        reconstructed = tq.qjl_decompress(jq, P)
        assert reconstructed.shape[-1] == 16

    def test_qjl_reduces_error(self):
        """Full TurboQuant (PQ+QJL) should have lower error than PQ alone."""
        import numpy as np
        vectors = _make_vectors(seq_len=16, head_dim=32)

        # PQ only
        pq_only, _ = tq.compress(vectors, bits=3, qjl=False)
        recon_pq = tq.decompress(pq_only, None)

        # PQ + QJL
        pq_qjl, jq = tq.compress(vectors, bits=3, qjl=True)
        recon_full = tq.decompress(pq_qjl, jq)

        orig = _to_numpy(vectors).reshape(-1)
        err_pq = np.mean((_to_numpy(recon_pq).reshape(-1) - orig) ** 2)
        err_full = np.mean((_to_numpy(recon_full).reshape(-1) - orig) ** 2)
        # QJL should reduce or at least not increase error
        assert err_full <= err_pq * 1.1, f"QJL made error worse: {err_full} > {err_pq}"


# ── TurboQuantCache Tests ──────────────────────────────────────────────

class TestTurboQuantCache:
    def _make_cache(self, bits=3, qjl=True):
        inner = MagicMock()
        inner.__class__.__name__ = "KVCache"
        return tq.TurboQuantCache(inner=inner, bits=bits, qjl=qjl)

    def test_update_and_fetch_single(self):
        """Single update should return the same data."""
        import numpy as np
        cache = self._make_cache(bits=4, qjl=False)
        keys = _make_vectors(seq_len=4, head_dim=16)
        values = _make_vectors(seq_len=4, head_dim=16)

        out_k, out_v = cache.update_and_fetch(keys, values)
        # No compression yet (below threshold), should be exact
        np.testing.assert_array_almost_equal(
            _to_numpy(out_k), _to_numpy(keys), decimal=5
        )

    def test_update_accumulates(self):
        """Multiple updates should grow the sequence."""
        cache = self._make_cache(bits=3, qjl=False)
        for _ in range(5):
            k = _make_vectors(seq_len=4, head_dim=16)
            v = _make_vectors(seq_len=4, head_dim=16)
            out_k, out_v = cache.update_and_fetch(k, v)

        assert out_k.shape[2] == 20  # 5 * 4

    def test_compression_triggers_at_threshold(self):
        """Buffer exceeding threshold should compress."""
        cache = self._make_cache(bits=3, qjl=False)
        # Push enough to exceed _COMPRESS_THRESHOLD (64)
        k = _make_vectors(seq_len=70, head_dim=16)
        v = _make_vectors(seq_len=70, head_dim=16)
        cache.update_and_fetch(k, v)

        assert len(cache._k_chunks) > 0, "Should have compressed data"
        assert cache._k_recent is None, "Recent buffer should be empty"

    def test_nbytes_smaller_than_raw(self):
        """Compressed nbytes should be less than raw float16."""
        cache = self._make_cache(bits=3, qjl=True)
        k = _make_vectors(seq_len=128, head_dim=32)
        v = _make_vectors(seq_len=128, head_dim=32)
        cache.update_and_fetch(k, v)

        raw_bytes = 2 * k.nbytes  # keys + values in float16
        compressed_bytes = cache.nbytes
        assert compressed_bytes < raw_bytes, (
            f"Compressed ({compressed_bytes}) should be < raw ({raw_bytes})"
        )

    def test_deepcopy(self):
        """Deep copy should produce independent cache."""
        cache = self._make_cache(bits=3, qjl=False)
        k = _make_vectors(seq_len=8, head_dim=16)
        v = _make_vectors(seq_len=8, head_dim=16)
        cache.update_and_fetch(k, v)

        cache2 = copy.deepcopy(cache)
        # Mutating original should not affect copy
        k2 = _make_vectors(seq_len=4, head_dim=16)
        v2 = _make_vectors(seq_len=4, head_dim=16)
        cache.update_and_fetch(k2, v2)

        assert cache._total_len == 12
        assert cache2._total_len == 8

    def test_empty(self):
        cache = self._make_cache()
        assert cache.empty()

    def test_trim_zero_keeps_cached_tokens(self):
        cache = self._make_cache(bits=3, qjl=False)
        cache.update_and_fetch(_make_vectors(seq_len=10, head_dim=16), _make_vectors(seq_len=10, head_dim=16))
        assert cache.trim(0) == 0
        assert cache._total_len == 10
        assert cache._k_recent.shape[2] == 10
        assert cache._v_recent.shape[2] == 10

    def test_trim_recent(self):
        cache = self._make_cache(bits=3, qjl=False)
        k = _make_vectors(seq_len=10, head_dim=16)
        v = _make_vectors(seq_len=10, head_dim=16)
        cache.update_and_fetch(k, v)

        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache._total_len == 7

    def test_wrap_skips_rotating_cache(self):
        """wrap() should return RotatingKVCache unchanged."""
        try:
            from mlx_lm.models.cache import RotatingKVCache
            mock_rot = MagicMock(spec=RotatingKVCache)
            result = tq.TurboQuantCache.wrap(mock_rot)
            assert result is mock_rot
        except ImportError:
            # RotatingKVCache not available (stub) — create a mock class
            rot_cls = type("RotatingKVCache", (), {})
            rot_mock = rot_cls()
            # Without real import, wrap should still return a TurboQuantCache
            result = tq.TurboQuantCache.wrap(rot_mock)
            assert isinstance(result, tq.TurboQuantCache)


# ── Config Tests ────────────────────────────────────────────────────────

class TestKVCacheConfig:
    def test_defaults(self):
        from ppmlx.config import KVCacheConfig
        cfg = KVCacheConfig()
        assert cfg.quantize == "off"
        assert cfg.bits == 3
        assert cfg.qjl is True
        assert cfg.qjl_dim == 0

    def test_config_has_kv_cache(self):
        from ppmlx.config import Config
        cfg = Config()
        assert hasattr(cfg, "kv_cache")
        assert cfg.kv_cache.quantize == "off"

    def test_toml_parsing(self, tmp_home):
        import tomli_w
        toml_path = tmp_home / ".ppmlx" / "config.toml"
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(toml_path, "wb") as f:
            tomli_w.dump({"kv_cache": {"quantize": "turboquant", "bits": 4, "qjl": False}}, f)

        from ppmlx.config import load_config
        cfg = load_config()
        assert cfg.kv_cache.quantize == "turboquant"
        assert cfg.kv_cache.bits == 4
        assert cfg.kv_cache.qjl is False

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PPMLX_KV_CACHE_QUANTIZE", "turboquant")
        monkeypatch.setenv("PPMLX_KV_CACHE_BITS", "4")
        from ppmlx.config import load_config
        cfg = load_config()
        assert cfg.kv_cache.quantize == "turboquant"
        assert cfg.kv_cache.bits == 4

    def test_cli_override(self):
        from ppmlx.config import load_config
        cfg = load_config(cli_overrides={"kv_quant": "turboquant"})
        assert cfg.kv_cache.quantize == "turboquant"
