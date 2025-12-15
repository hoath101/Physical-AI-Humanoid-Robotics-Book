import time
import asyncio
from typing import Any, Optional, Dict
from functools import wraps

class SimpleCache:
    """
    A simple in-memory cache with TTL (Time To Live) functionality.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl: int = 300) -> None:  # Default TTL: 5 minutes
        """
        Set a value in the cache with a TTL.
        """
        expiration_time = time.time() + ttl
        self._cache[key] = {
            'value': value,
            'expiration': expiration_time
        }

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache if it exists and hasn't expired.
        """
        if key in self._cache:
            cached_item = self._cache[key]
            if time.time() < cached_item['expiration']:
                return cached_item['value']
            else:
                # Remove expired item
                del self._cache[key]
        return None

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        self._cache.clear()

    def cleanup_expired(self) -> None:
        """
        Remove all expired items from the cache.
        """
        current_time = time.time()
        expired_keys = [
            key for key, value in self._cache.items()
            if current_time >= value['expiration']
        ]
        for key in expired_keys:
            del self._cache[key]

# Global cache instance
cache = SimpleCache()

def cached(ttl: int = 300):
    """
    Decorator to cache function results.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator

def sync_cached(ttl: int = 300):
    """
    Decorator to cache synchronous function results.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator