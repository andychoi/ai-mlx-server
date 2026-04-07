"""Unit tests for ModelCache LRU eviction policy. No MLX required."""
import sys
import os
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cache import ModelCache


class TestModelCacheLRU(unittest.TestCase):
    """Test LRU eviction with max_models limit."""

    def test_max_models_evicts_oldest(self):
        """Put 5 entries into a cache of size 3; oldest 2 should be evicted."""
        cache = ModelCache(max_models=3)
        for i in range(5):
            cache.put(f"model_{i}", f"value_{i}")
        self.assertEqual(len(cache), 3)
        # model_0 and model_1 should have been evicted
        self.assertIsNone(cache.get("model_0"))
        self.assertIsNone(cache.get("model_1"))
        # model_2, model_3, model_4 should still be present
        self.assertEqual(cache.get("model_2"), "value_2")
        self.assertEqual(cache.get("model_3"), "value_3")
        self.assertEqual(cache.get("model_4"), "value_4")

    def test_get_bumps_lru_position(self):
        """Access entry 1 before adding entry 4 — entry 2 should be evicted, not entry 1."""
        cache = ModelCache(max_models=3)
        cache.put("model_0", "v0")
        cache.put("model_1", "v1")
        cache.put("model_2", "v2")
        # Access model_0 — it moves to end (most recently used)
        self.assertEqual(cache.get("model_0"), "v0")
        # Now add model_3 — cache is over limit; model_1 is LRU
        cache.put("model_3", "v3")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get("model_1"))   # evicted
        self.assertEqual(cache.get("model_0"), "v0")  # still present
        self.assertEqual(cache.get("model_2"), "v2")  # still present
        self.assertEqual(cache.get("model_3"), "v3")  # still present

    def test_pinned_models_survive_eviction(self):
        """Pinned models must not be evicted even when over the limit."""
        cache = ModelCache(max_models=2)
        cache.put("pinned_model", "vp", pinned=True)
        cache.put("model_a", "va")
        # Adding model_b forces eviction — pinned_model must survive
        cache.put("model_b", "vb")
        self.assertIsNotNone(cache.get("pinned_model"))
        # Total may exceed max_models if all non-pinned are gone
        self.assertGreaterEqual(len(cache), 1)

    def test_stats_returns_correct_count(self):
        """stats() should return an entry for each resident model."""
        cache = ModelCache(max_models=5)
        cache.put("m1", "v1", role="chat")
        cache.put("m2", "v2", role="embedding")
        cache.put("m3", "v3")
        stats = cache.stats()
        self.assertEqual(len(stats), 3)
        ids = {s["id"] for s in stats}
        self.assertEqual(ids, {"m1", "m2", "m3"})
        roles = {s["id"]: s["role"] for s in stats}
        self.assertEqual(roles["m1"], "chat")
        self.assertEqual(roles["m2"], "embedding")

    def test_max_bytes_evicts(self):
        """Byte-based eviction: exceed max_bytes triggers LRU removal."""
        # max 100 bytes; each entry is 60 bytes
        cache = ModelCache(max_bytes=100)
        cache.put("big_0", "v0", est_bytes=60)
        cache.put("big_1", "v1", est_bytes=60)
        # total = 120 > 100 → big_0 should be evicted
        self.assertEqual(len(cache), 1)
        self.assertIsNone(cache.get("big_0"))
        self.assertEqual(cache.get("big_1"), "v1")

    def test_put_updates_existing_entry(self):
        """Re-putting an existing key updates value without adding a new entry."""
        cache = ModelCache(max_models=3)
        cache.put("m1", "old")
        cache.put("m1", "new")
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get("m1"), "new")

    def test_on_evict_callback_fires(self):
        """on_evict is called with the evicted key when LRU eviction occurs."""
        evicted = []
        cache = ModelCache(max_models=2, on_evict=evicted.append)
        cache.put("m0", "v0")
        cache.put("m1", "v1")
        cache.put("m2", "v2")   # triggers eviction of m0
        self.assertEqual(evicted, ["m0"])
        cache.put("m3", "v3")   # triggers eviction of m1
        self.assertEqual(evicted, ["m0", "m1"])

    def test_on_evict_callback_can_access_cache(self):
        """Callback must not deadlock when it reads from the same cache."""
        accessed = []

        def callback(key):
            # Simulate _tear_down_worker calling cache.get() or stats()
            accessed.append(cache.stats())  # would deadlock if lock is held

        cache = ModelCache(max_models=1, on_evict=callback)
        cache.put("m0", "v0")
        cache.put("m1", "v1")  # evicts m0 → callback fires → calls cache.stats()
        self.assertEqual(len(accessed), 1)  # callback was called


if __name__ == "__main__":
    unittest.main()
