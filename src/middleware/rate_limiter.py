"""
Redis-backed sliding window rate limiter.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger("superai.ratelimit")


class RateLimiter:
    """
    Sliding window rate limiter.
    Uses in-memory store with optional Redis backend for distributed deployments.
    """

    def __init__(self, redis_client: Optional[Any] = None):
        self._settings = get_settings()
        self._redis = redis_client
        self._local_store: Dict[str, list] = {}

    async def check(
        self,
        key: str,
        limit: Optional[int] = None,
        window_seconds: int = 60,
    ) -> bool:
        """
        Check if request is within rate limit.
        Returns True if allowed, raises HTTPException if exceeded.
        """
        max_requests = limit or self._settings.rate_limit_per_minute
        now = time.time()

        if self._redis:
            return await self._check_redis(key, max_requests, window_seconds, now)
        return self._check_local(key, max_requests, window_seconds, now)

    def _check_local(
        self, key: str, limit: int, window: int, now: float
    ) -> bool:
        """In-memory sliding window check."""
        if key not in self._local_store:
            self._local_store[key] = []

        # Remove expired entries
        cutoff = now - window
        self._local_store[key] = [
            ts for ts in self._local_store[key] if ts > cutoff
        ]

        if len(self._local_store[key]) >= limit:
            retry_after = int(self._local_store[key][0] + window - now) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after}s",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(self._local_store[key][0] + window)),
                },
            )

        self._local_store[key].append(now)
        return True

    async def _check_redis(
        self, key: str, limit: int, window: int, now: float
    ) -> bool:
        """Redis sliding window check using sorted sets."""
        redis_key = f"superai:ratelimit:{key}"
        cutoff = now - window

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(redis_key, 0, cutoff)
        pipe.zcard(redis_key)
        pipe.zadd(redis_key, {str(now): now})
        pipe.expire(redis_key, window + 1)
        results = await pipe.execute()

        current_count = results[1]

        if current_count >= limit:
            # Get oldest entry for retry-after calculation
            oldest = await self._redis.zrange(redis_key, 0, 0, withscores=True)
            retry_after = int(oldest[0][1] + window - now) + 1 if oldest else window
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after}s",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        remaining = max(0, limit - current_count - 1)
        return True

    async def get_usage(self, key: str, window_seconds: int = 60) -> Dict[str, int]:
        """Get current rate limit usage for a key."""
        now = time.time()
        cutoff = now - window_seconds

        if self._redis:
            redis_key = f"superai:ratelimit:{key}"
            count = await self._redis.zcount(redis_key, cutoff, now)
        else:
            entries = self._local_store.get(key, [])
            count = sum(1 for ts in entries if ts > cutoff)

        limit = self._settings.rate_limit_per_minute
        return {
            "used": count,
            "limit": limit,
            "remaining": max(0, limit - count),
            "window_seconds": window_seconds,
        }

    def cleanup(self) -> None:
        """Remove expired entries from local store."""
        now = time.time()
        cutoff = now - 120  # 2 minute cleanup window
        keys_to_remove = []
        for key, timestamps in self._local_store.items():
            self._local_store[key] = [ts for ts in timestamps if ts > cutoff]
            if not self._local_store[key]:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._local_store[key]