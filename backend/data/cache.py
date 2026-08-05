"""diskcache layer — all external calls go through here.

The cache is unbounded by default in diskcache, which let it reach 1 GB across
4,567 files with no way to reclaim space short of deleting the directory. A
size limit and eviction policy are set at construction, and
:func:`prune_expired` removes entries that no live TTL can ever accept.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import diskcache

_log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "cache",
)

# Cap the cache at 2 GB. diskcache evicts least-recently-used entries once the
# limit is passed, so a long training run cannot fill the disk. Override with
# CACHE_SIZE_LIMIT_MB when a bigger working set is genuinely needed.
_DEFAULT_SIZE_LIMIT_MB = 2048

# The longest TTL any caller asks for: squad profiles and REEP aliases at 7
# days. Entries older than this can never satisfy a read, so they are safe to
# delete. Kept here rather than in each client so pruning has one source of
# truth; raise it if a longer TTL is introduced.
MAX_TTL_SECONDS = 604_800  # 7 days

_cache: Optional[diskcache.Cache] = None


def _size_limit_bytes() -> int:
    raw = os.environ.get("CACHE_SIZE_LIMIT_MB")
    try:
        mb = int(raw) if raw else _DEFAULT_SIZE_LIMIT_MB
    except ValueError:
        mb = _DEFAULT_SIZE_LIMIT_MB
    return max(64, mb) * 1024 * 1024


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        cache_dir = os.environ.get("CACHE_DIR", _DEFAULT_CACHE_DIR)
        _cache = diskcache.Cache(
            cache_dir,
            size_limit=_size_limit_bytes(),
            eviction_policy="least-recently-used",
        )
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


def clear() -> None:
    """Delete all cached data."""
    cache = _get_cache()
    cache.clear()


def prune_expired(max_age: int = MAX_TTL_SECONDS) -> int:
    """Delete entries older than any TTL a caller could still accept.

    Every read passes its own ``max_age``, so nothing is evicted on read — an
    entry only disappears when the LRU limit is hit. This removes entries that
    are older than the longest TTL in use and therefore cannot satisfy any
    read, reclaiming space without affecting behaviour.

    Returns the number of entries deleted.
    """
    cache = _get_cache()
    cutoff = time.time() - max_age
    deleted = 0
    for key in list(cache):
        entry = cache.get(key)
        if not isinstance(entry, tuple) or len(entry) != 2:
            continue
        stored_time, _ = entry
        try:
            if float(stored_time) < cutoff:
                cache.delete(key)
                deleted += 1
        except (TypeError, ValueError):
            continue
    if deleted:
        _log.info("Cache prune removed %d expired entries", deleted)
    return deleted


def stats() -> Dict[str, Any]:
    """Return cache size, entry count and configured limit.

    Used by the Diagnostics page so cache growth is visible rather than
    something you discover when the disk fills.
    """
    cache = _get_cache()
    limit = _size_limit_bytes()
    try:
        volume = int(cache.volume())
        count = len(cache)
    except Exception:  # pragma: no cover - defensive
        return {"error": "cache unavailable"}
    return {
        "entries": count,
        "bytes": volume,
        "mb": round(volume / (1024 * 1024), 1),
        "limit_mb": round(limit / (1024 * 1024)),
        "pct_of_limit": round(volume / limit * 100, 1) if limit else 0.0,
        "eviction_policy": "least-recently-used",
    }


def namespace_breakdown() -> Dict[str, int]:
    """Count entries per namespace, largest first."""
    cache = _get_cache()
    counts: Dict[str, int] = {}
    for key in list(cache):
        if not isinstance(key, str):
            continue
        namespace = key.split(":", 1)[0] if ":" in key else "(none)"
        counts[namespace] = counts.get(namespace, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def close() -> None:
    """Close the underlying cache (useful in tests)."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None
