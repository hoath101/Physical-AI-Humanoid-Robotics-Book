"""
Cache service for the Physical AI & Humanoid Robotics Book RAG Chatbot.
Implements caching for query responses to improve performance and reduce API calls.
"""

import asyncio
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis.asyncio as redis
from config.ingestion_config import get_config_value

# Global cache service instance
cache_service = None


class CacheService:
    """
    Asynchronous cache service for storing and retrieving query responses.
    Uses Redis as the backend cache with fallback to in-memory cache.
    """

    def __init__(self):
        self.redis_client = None
        self.in_memory_cache = {}
        self.cache_ttl = int(get_config_value('CACHE_TTL_SECONDS', 3600))  # Default 1 hour
        self.max_memory_items = int(get_config_value('CACHE_MAX_MEMORY_ITEMS', 1000))

    async def initialize(self):
        """Initialize the cache service, attempting to connect to Redis first."""
        try:
            # Try to connect to Redis
            redis_url = get_config_value('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = await redis.from_url(redis_url, decode_responses=True)
            # Test the connection
            await self.redis_client.ping()
            print("Cache service initialized with Redis backend")
        except Exception as e:
            print(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self.redis_client = None

    async def close(self):
        """Close the cache service connections."""
        if self.redis_client:
            await self.redis_client.close()

    def _generate_cache_key(self, question: str, selected_text: Optional[str], book_id: str) -> str:
        """
        Generate a unique cache key for the given query parameters.

        Args:
            question: The user's question
            selected_text: Optional selected text for context
            book_id: ID of the book being queried

        Returns:
            str: A unique cache key
        """
        cache_input = f"{question}||{selected_text or ''}||{book_id}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    async def get_query_response(self, question: str, selected_text: Optional[str], book_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached query response.

        Args:
            question: The user's question
            selected_text: Optional selected text for context
            book_id: ID of the book being queried

        Returns:
            Cached response dict if found, None otherwise
        """
        cache_key = self._generate_cache_key(question, selected_text, book_id)

        # Try Redis first if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                print(f"Error retrieving from Redis cache: {e}")

        # Fallback to in-memory cache
        if cache_key in self.in_memory_cache:
            cached_item = self.in_memory_cache[cache_key]
            # Check if cache item has expired
            if datetime.now() < cached_item['expires_at']:
                return cached_item['data']
            else:
                # Remove expired item
                del self.in_memory_cache[cache_key]

        return None

    async def set_query_response(self, question: str, selected_text: Optional[str], book_id: str, response: Dict[str, Any]):
        """
        Store a query response in cache.

        Args:
            question: The user's question
            selected_text: Optional selected text for context
            book_id: ID of the book being queried
            response: The response to cache
        """
        cache_key = self._generate_cache_key(question, selected_text, book_id)
        expires_at = datetime.now() + timedelta(seconds=self.cache_ttl)

        # Store in Redis if available
        if self.redis_client:
            try:
                cached_data = json.dumps(response)
                await self.redis_client.setex(cache_key, self.cache_ttl, cached_data)
            except Exception as e:
                print(f"Error storing in Redis cache: {e}")

        # Store in in-memory cache
        self.in_memory_cache[cache_key] = {
            'data': response,
            'expires_at': expires_at
        }

        # Manage memory usage
        if len(self.in_memory_cache) > self.max_memory_items:
            # Remove oldest items (simple approach)
            oldest_keys = sorted(
                self.in_memory_cache.keys(),
                key=lambda k: self.in_memory_cache[k]['expires_at']
            )[:100]  # Remove 100 oldest items

            for key in oldest_keys:
                if key in self.in_memory_cache:
                    del self.in_memory_cache[key]

    async def delete_query_response(self, question: str, selected_text: Optional[str], book_id: str):
        """
        Remove a cached query response.

        Args:
            question: The user's question
            selected_text: Optional selected text for context
            book_id: ID of the book being queried
        """
        cache_key = self._generate_cache_key(question, selected_text, book_id)

        # Remove from Redis if available
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
            except Exception as e:
                print(f"Error deleting from Redis cache: {e}")

        # Remove from in-memory cache
        if cache_key in self.in_memory_cache:
            del self.in_memory_cache[cache_key]

    async def clear_cache(self):
        """Clear all cached items."""
        if self.redis_client:
            try:
                # This will clear the entire Redis database
                await self.redis_client.flushdb()
            except Exception as e:
                print(f"Error clearing Redis cache: {e}")

        self.in_memory_cache.clear()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        stats = {
            'in_memory_items': len(self.in_memory_cache),
            'cache_ttl_seconds': self.cache_ttl,
            'max_memory_items': self.max_memory_items,
            'backend': 'redis' if self.redis_client else 'memory_only'
        }

        if self.redis_client:
            try:
                # Get Redis info
                info = await self.redis_client.info()
                stats['redis_connected'] = True
                stats['redis_used_memory'] = info.get('used_memory_human', 'N/A')
                stats['redis_total_commands'] = info.get('total_commands_processed', 0)
            except Exception as e:
                stats['redis_connected'] = False
                stats['redis_error'] = str(e)

        return stats


async def init_cache():
    """Initialize the global cache service."""
    global cache_service
    cache_service = CacheService()
    await cache_service.initialize()


async def close_cache():
    """Close the global cache service."""
    global cache_service
    if cache_service:
        await cache_service.close()


# Additional utility functions for cache management
async def get_cache_response(question: str, selected_text: Optional[str], book_id: str):
    """Get a cached response."""
    global cache_service
    if cache_service:
        return await cache_service.get_query_response(question, selected_text, book_id)
    return None


async def set_cache_response(question: str, selected_text: Optional[str], book_id: str, response: Dict[str, Any]):
    """Set a cached response."""
    global cache_service
    if cache_service:
        await cache_service.set_query_response(question, selected_text, book_id, response)


async def clear_all_cache():
    """Clear all cache."""
    global cache_service
    if cache_service:
        await cache_service.clear_cache()