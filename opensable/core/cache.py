"""
Open-Sable Caching System

Multi-layer caching with memory, disk, and optional Redis support.
Automatic TTL, LRU eviction, and cache warming.
"""

import asyncio
import logging
from typing import Any, Optional, Callable
from pathlib import Path
import pickle
import hashlib
from collections import OrderedDict
import time

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with metadata"""

    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl  # seconds
        self.hits = 0
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def access(self) -> Any:
        """Access value and update metadata"""
        self.hits += 1
        self.last_accessed = time.time()
        return self.value


class MemoryCache:
    """In-memory LRU cache"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                return None

            if entry.is_expired():
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            return entry.access()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        async with self.lock:
            entry = CacheEntry(value, ttl)

            # Remove if exists
            if key in self.cache:
                del self.cache[key]

            # Add to end
            self.cache[key] = entry

            # Evict oldest if over size
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove first (oldest)

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self):
        """Clear all cache"""
        async with self.lock:
            self.cache.clear()

    async def cleanup(self):
        """Remove expired entries"""
        async with self.lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

            for key in expired_keys:
                del self.cache[key]

            return len(expired_keys)

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_hits = sum(entry.hits for entry in self.cache.values())

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
        }


class DiskCache:
    """Persistent disk cache"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get file path for key"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        path = self._get_path(key)

        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            entry = data["entry"]

            if entry.is_expired():
                path.unlink()
                return None

            return entry.access()

        except Exception as e:
            logger.error(f"Error reading cache {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in disk cache"""
        path = self._get_path(key)

        try:
            entry = CacheEntry(value, ttl)
            data = {"key": key, "entry": entry}

            with open(path, "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Error writing cache {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from disk cache"""
        path = self._get_path(key)

        if path.exists():
            path.unlink()
            return True

        return False

    async def clear(self):
        """Clear all disk cache"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()

    async def cleanup(self):
        """Remove expired entries"""
        expired_count = 0

        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)

                if data["entry"].is_expired():
                    cache_file.unlink()
                    expired_count += 1

            except Exception as e:
                logger.error(f"Error checking cache file {cache_file}: {e}")

        return expired_count


class MultiLayerCache:
    """Multi-layer cache with memory and disk tiers"""

    def __init__(self, config: Config):
        self.config = config

        # Memory cache (L1)
        self.memory_cache = MemoryCache(max_size=getattr(config, "cache_memory_size", 1000))

        # Disk cache (L2)
        cache_dir = Path.home() / ".opensable" / "cache"
        self.disk_cache = DiskCache(cache_dir)

        # Configuration
        self.use_disk = getattr(config, "cache_use_disk", True)
        self.default_ttl = getattr(config, "cache_default_ttl", 3600)  # 1 hour

        # Stats
        self.stats = {"hits": 0, "misses": 0, "sets": 0}

        # Cleanup task
        self._cleanup_task = None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks L1 then L2)"""
        # Check memory cache (L1)
        value = await self.memory_cache.get(key)

        if value is not None:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit (L1): {key}")
            return value

        # Check disk cache (L2) if enabled
        if self.use_disk:
            value = await self.disk_cache.get(key)

            if value is not None:
                self.stats["hits"] += 1
                logger.debug(f"Cache hit (L2): {key}")

                # Promote to L1
                await self.memory_cache.set(key, value)

                return value

        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in all cache layers"""
        if ttl is None:
            ttl = self.default_ttl

        # Set in L1 (memory)
        await self.memory_cache.set(key, value, ttl)

        # Set in L2 (disk) if enabled
        if self.use_disk:
            await self.disk_cache.set(key, value, ttl)

        self.stats["sets"] += 1
        logger.debug(f"Cache set: {key} (TTL: {ttl}s)")

    async def delete(self, key: str) -> bool:
        """Delete key from all cache layers"""
        deleted = False

        if await self.memory_cache.delete(key):
            deleted = True

        if self.use_disk and await self.disk_cache.delete(key):
            deleted = True

        return deleted

    async def clear(self):
        """Clear all cache layers"""
        await self.memory_cache.clear()

        if self.use_disk:
            await self.disk_cache.clear()

        logger.info("Cache cleared")

    async def get_or_compute(
        self, key: str, compute_fn: Callable, ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and cache"""
        # Try to get from cache
        value = await self.get(key)

        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(compute_fn):
            value = await compute_fn()
        else:
            value = compute_fn()

        # Cache result
        await self.set(key, value, ttl)

        return value

    async def start(self):
        """Start cache cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache system started")

    async def stop(self):
        """Stop cache cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Cache system stopped")

    async def _cleanup_loop(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                logger.debug("Running cache cleanup...")

                memory_cleaned = await self.memory_cache.cleanup()

                disk_cleaned = 0
                if self.use_disk:
                    disk_cleaned = await self.disk_cache.cleanup()

                if memory_cleaned > 0 or disk_cleaned > 0:
                    logger.info(f"Cache cleanup: {memory_cleaned} memory, {disk_cleaned} disk")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {**self.stats, "hit_rate": hit_rate, "memory_cache": self.memory_cache.get_stats()}


# Decorator for caching function results
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator to cache function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Build cache key from function name and args
            key_parts = [key_prefix, func.__name__]

            # Add args to key
            if args:
                key_parts.append(str(hash(args)))
            if kwargs:
                key_parts.append(str(hash(tuple(sorted(kwargs.items())))))

            cache_key = ":".join(key_parts)

            # Get cache instance (assumes it's available globally or passed)
            # This is a simplified version - in production, inject cache properly
            from opensable.core.config import load_config

            cache = MultiLayerCache(load_config())

            # Try to get from cache
            result = await cache.get(cache_key)

            if result is not None:
                return result

            # Compute result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache result
            await cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    cache = MultiLayerCache(config)

    async def test():
        await cache.start()

        # Set values
        await cache.set("user:123:name", "Alice", ttl=60)
        await cache.set("user:123:email", "alice@example.com", ttl=60)

        # Get values
        name = await cache.get("user:123:name")
        print(f"Name: {name}")

        # Get or compute
        def expensive_computation():
            time.sleep(1)
            return "Result of expensive computation"

        result = await cache.get_or_compute("computation:result", expensive_computation, ttl=300)
        print(f"Computation result: {result}")

        # Stats
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")

        await cache.stop()

    asyncio.run(test())
