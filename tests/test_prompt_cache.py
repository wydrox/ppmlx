"""Tests for ppmlx.prompt_cache — LRU prompt KV-cache store."""
from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_fake_cache(n_layers: int = 2) -> list:
    """Create a fake KV-cache (list of mock layer caches)."""
    layers = []
    for _ in range(n_layers):
        layer = MagicMock()
        layer.state = MagicMock()
        layer.is_trimmable = MagicMock(return_value=True)
        layer.trim = MagicMock(return_value=1)
        layers.append(layer)
    return layers


# ── Tests ───────────────────────────────────────────────────────────────

class TestPromptCacheStore:
    def test_empty_cache_returns_none(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        cache, remaining = store.find_prefix("model/a", [1, 2, 3])
        assert cache is None
        assert remaining == [1, 2, 3]

    def test_store_and_find_exact(self, monkeypatch):
        from ppmlx.prompt_cache import PromptCacheStore
        # Mock mlx-lm cache functions for the exact-hit trim path
        import types
        fake_cache_mod = types.ModuleType("mlx_lm.models.cache")
        fake_cache_mod.can_trim_prompt_cache = lambda c: True
        fake_cache_mod.trim_prompt_cache = lambda c, n: n
        monkeypatch.setitem(
            __import__("sys").modules, "mlx_lm.models.cache", fake_cache_mod,
        )

        store = PromptCacheStore(max_entries=4)
        fake = _make_fake_cache()
        store.store("model/a", [1, 2, 3], fake)

        # Exact match — should trim by 1 and return last token
        cache, remaining = store.find_prefix("model/a", [1, 2, 3])
        assert cache is not None
        assert remaining == [3]  # last token re-added after trim

    def test_store_and_find_prefix(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        fake = _make_fake_cache()
        store.store("model/a", [1, 2, 3], fake)

        # Longer prompt with cached prefix
        cache, remaining = store.find_prefix("model/a", [1, 2, 3, 4, 5])
        assert cache is not None
        assert remaining == [4, 5]

    def test_no_prefix_match(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        fake = _make_fake_cache()
        store.store("model/a", [1, 2, 3], fake)

        # Different prefix — no match
        cache, remaining = store.find_prefix("model/a", [4, 5, 6])
        assert cache is None
        assert remaining == [4, 5, 6]

    def test_longest_prefix_wins(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        store.store("model/a", [1, 2], _make_fake_cache())
        store.store("model/a", [1, 2, 3, 4], _make_fake_cache())

        cache, remaining = store.find_prefix("model/a", [1, 2, 3, 4, 5, 6])
        assert cache is not None
        assert remaining == [5, 6]

    def test_different_models_isolated(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        store.store("model/a", [1, 2, 3], _make_fake_cache())

        # Different model — no match
        cache, remaining = store.find_prefix("model/b", [1, 2, 3])
        assert cache is None
        assert remaining == [1, 2, 3]

    def test_lru_eviction(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=2)
        store.store("model/a", [1, 2], _make_fake_cache())
        store.store("model/a", [3, 4], _make_fake_cache())
        assert len(store) == 2

        # Adding a third should evict the first
        store.store("model/a", [5, 6], _make_fake_cache())
        assert len(store) == 2

        # First entry should be gone
        cache, _ = store.find_prefix("model/a", [1, 2, 7])
        assert cache is None

        # Second and third should still be there
        cache, _ = store.find_prefix("model/a", [3, 4, 7])
        assert cache is not None
        cache, _ = store.find_prefix("model/a", [5, 6, 7])
        assert cache is not None

    def test_find_updates_lru_order(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=2)
        store.store("model/a", [1, 2], _make_fake_cache())
        store.store("model/a", [3, 4], _make_fake_cache())

        # Access the first entry — moves it to end (most recently used)
        store.find_prefix("model/a", [1, 2, 7])

        # Now adding a third should evict [3,4] (LRU), not [1,2]
        store.store("model/a", [5, 6], _make_fake_cache())
        cache, _ = store.find_prefix("model/a", [1, 2, 7])
        assert cache is not None
        cache, _ = store.find_prefix("model/a", [3, 4, 7])
        assert cache is None

    def test_clear_all(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        store.store("model/a", [1, 2], _make_fake_cache())
        store.store("model/b", [3, 4], _make_fake_cache())
        assert len(store) == 2

        store.clear()
        assert len(store) == 0

    def test_clear_by_model(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        store.store("model/a", [1, 2], _make_fake_cache())
        store.store("model/b", [3, 4], _make_fake_cache())

        store.clear("model/a")
        assert len(store) == 1
        cache, _ = store.find_prefix("model/b", [3, 4, 5])
        assert cache is not None

    def test_deep_copy_isolation(self):
        """Modifying a returned cache should not affect the stored copy."""
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        original = _make_fake_cache()
        store.store("model/a", [1, 2, 3], original)

        cache1, _ = store.find_prefix("model/a", [1, 2, 3, 4])
        cache1.append("MUTATED")

        cache2, _ = store.find_prefix("model/a", [1, 2, 3, 5])
        assert "MUTATED" not in cache2

    def test_stats(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        store.store("model/a", [1, 2, 3], _make_fake_cache())
        store.store("model/b", [4, 5], _make_fake_cache())

        stats = store.stats()
        assert stats["entries"] == 2
        assert stats["max_entries"] == 4
        assert len(stats["models"]) == 2

    def test_disabled_when_limit_zero(self):
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=0)
        store.store("model/a", [1, 2, 3], _make_fake_cache())
        assert len(store) == 0

    def test_partial_prefix_no_match(self):
        """A cached [1,2,3] should NOT match a prompt [1,3,2]."""
        from ppmlx.prompt_cache import PromptCacheStore
        store = PromptCacheStore(max_entries=4)
        store.store("model/a", [1, 2, 3], _make_fake_cache())

        cache, remaining = store.find_prefix("model/a", [1, 3, 2, 4])
        assert cache is None
        assert remaining == [1, 3, 2, 4]
