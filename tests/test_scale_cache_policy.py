"""Check memory bounds, scan behavior and actual cache replacement decisions."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm.scale_cache_policy import CachePolicy


class PolicyTests(unittest.TestCase):
    def test_disabled_and_oversized_bypass_without_evicting(self):
        off = CachePolicy(1024, False)
        off.register("a", 1024)
        for _ in range(3):
            self.assertIsNone(off.lookup("a").offset)
        self.assertEqual(off.report()["hits"], 0)
        self.assertEqual(off.report()["decoded_bytes"], 3072)
        cache = CachePolicy(1024, policy="lru")
        cache.register("small", 256)
        cache.register("big", 2048)
        cache.lookup("small")
        self.assertEqual(cache.lookup("big").evicted, ())
        self.assertTrue(cache.lookup("small").hit)

    def test_retain_survives_full_layer_scan(self):
        cache = CachePolicy(512, policy="retain")
        for key in "abcd":
            cache.register(key, 256)
        for _ in range(3):
            for key in "abcd":
                cache.lookup(key)
        self.assertEqual(cache.report()["hits"], 4)
        self.assertEqual(list(cache.entries), ["a", "b"])
        self.assertEqual(cache.report()["bypasses"], 6)

    def test_lru_honors_reuse_and_coalesces_free_blocks(self):
        cache = CachePolicy(1024, policy="lru")
        for key, size in (("a", 256), ("b", 256), ("c", 512), ("d", 768)):
            cache.register(key, size)
        for key in "abca":
            cache.lookup(key)
        result = cache.lookup("d")
        self.assertEqual(result.evicted, ("b", "c"))
        self.assertEqual(result.offset, 256)
        self.assertTrue(cache.lookup("a").hit)

    def test_randomized_arena_never_overlaps_or_exceeds_budget(self):
        rng = random.Random(815)
        for mode in ("lru", "retain"):
            cache = CachePolicy(4096, policy=mode)
            for i in range(24):
                cache.register(str(i), rng.randrange(1, 6145))
            for _ in range(2000):
                key = str(rng.randrange(24))
                before = key in cache.entries
                decision = cache.lookup(key)
                self.assertEqual(decision.hit, before)
                ranges = sorted([(o, n) for o, n, _ in cache.entries.values()] + cache.free)
                cursor = 0
                for offset, length in ranges:
                    self.assertEqual(offset, cursor)
                    cursor += length
                self.assertEqual(cursor, 4096)
                self.assertLessEqual(cache.report()["occupied_bytes"], 4096)

    def test_reset_clear_and_invalid_input(self):
        cache = CachePolicy(512)
        cache.register("a", 5)
        with self.assertRaises(ValueError):
            cache.register("a", 5)
        with self.assertRaises(KeyError):
            cache.lookup("absent")
        self.assertEqual(cache.report()["lookups"], 0)
        cache.lookup("a")
        cache.reset_stats()
        self.assertTrue(cache.lookup("a").hit)
        cache.clear()
        self.assertEqual(cache.report()["resident_entries"], 0)
        self.assertFalse(cache.lookup("a").hit)
        for args in ((-1,), (1,), (512, True, "random"), (512, 1)):
            with self.assertRaises(ValueError):
                CachePolicy(*args)


if __name__ == "__main__":
    unittest.main()
