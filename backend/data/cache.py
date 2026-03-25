"""diskcache layer — all external calls go through here."""

import os
import time
from typing import Any, Optional

import diskcache

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "cache",
)

_cache: Optional[diskcache.Cache] = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        cache_dir = os.environ.get("CACHE_DIR", _DEFAULT_CACHE_DIR)
        _cache = diskcache.Cache(cache_dir)
    return _cache


def make_key(namespace: str, *parts: str) -> str:
    """Build a namespaced cache key."""
    return f"{namespace}:" + ":".join(str(p) for p in parts)


def get(key: str, max_age: Optional[int] = None) -> Any:
    """Return cached value or None if missing / expired.

    Parameters
    ----------
    key : str
        Cache key (use ``make_key`` to build).
    max_age : int, optional
        Maximum age in seconds.  If the stored timestamp is older than
        ``now - max_age`` the entry is treated as expired and ``None``
        is returned.
    """
    cache = _get_cache()
    entry = cache.get(key)
    if entry is None:
        return None
    stored_time, value = entry
    if max_age is not None and (time.time() - stored_time) > max_age:
        return None
    return value


def set(key: str, value: Any) -> None:  # noqa: A001
    """Store *value* under *key* with a timestamp."""
    cache = _get_cache()
    cache.set(key, (time.time(), value))


def invalidate(key: str) -> bool:
    """Delete a single key. Returns True if the key existed."""
    cache = _get_cache()
    return cache.delete(key)


def clear_namespace(namespace: str) -> int:
    """Delete all keys that start with *namespace:*. Returns count deleted."""
    cache = _get_cache()
    count = 0
    for key in list(cache):
        if isinstance(key, str) and key.startswith(f"{namespace}:"):
            cache.delete(key)
            count += 1
    return count


def close() -> None:
    """Close the underlying cache (useful in tests)."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None
