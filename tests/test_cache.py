"""Unit tests for backend.data.cache."""

import os
import shutil
import tempfile
import time
import unittest

# Point cache at a temp dir before importing
_TEMP_DIR = tempfile.mkdtemp(prefix="ts_cache_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache


class TestCache(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    def test_make_key(self):
        k = cache.make_key("fotmob", "player", "123")
        self.assertEqual(k, "fotmob:player:123")

    def test_set_and_get(self):
        cache.set("test:a", {"foo": 1})
        val = cache.get("test:a")
        self.assertEqual(val, {"foo": 1})

    def test_get_missing_returns_none(self):
        self.assertIsNone(cache.get("nonexistent_key_xyz"))

    def test_max_age_fresh(self):
        cache.set("test:fresh", 42)
        val = cache.get("test:fresh", max_age=60)
        self.assertEqual(val, 42)

    def test_max_age_expired(self):
        cache.set("test:old", 99)
        # Manually expire by overwriting timestamp
        c = cache._get_cache()
        c.set("test:old", (time.time() - 120, 99))
        val = cache.get("test:old", max_age=60)
        self.assertIsNone(val)

    def test_invalidate(self):
        cache.set("test:del", "bye")
        self.assertTrue(cache.invalidate("test:del"))
        self.assertIsNone(cache.get("test:del"))

    def test_invalidate_missing(self):
        self.assertFalse(cache.invalidate("never_existed_xyz"))

    def test_clear_namespace(self):
        cache.set("ns1:a", 1)
        cache.set("ns1:b", 2)
        cache.set("ns2:c", 3)
        count = cache.clear_namespace("ns1")
        self.assertEqual(count, 2)
        self.assertIsNone(cache.get("ns1:a"))
        self.assertIsNone(cache.get("ns1:b"))
        self.assertEqual(cache.get("ns2:c"), 3)


if __name__ == "__main__":
    unittest.main()
