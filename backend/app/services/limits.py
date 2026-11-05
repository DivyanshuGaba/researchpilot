"""Rate limits and daily token budgets. Redis is the shared store; if it is down the API stays
up and falls back to a conservative in-process limiter (architecture.md section 13)."""

import logging
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)


class Limiter:
    def __init__(self, redis_url: str | None) -> None:
        self._url = redis_url
        self._redis: Any = None
        self._local: dict[str, tuple[float, int]] = {}  # key -> (window_start or expiry, count)
        self.redis_ok = redis_url is not None

    def _client(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            assert self._url is not None
            self._redis = aioredis.from_url(
                self._url, socket_timeout=0.5, socket_connect_timeout=0.5
            )
        return self._redis

    def _local_hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        now = time.monotonic()
        start, count = self._local.get(key, (now, 0))
        if now - start >= window_s:
            start, count = now, 0
        count += 1
        self._local[key] = (start, count)
        return count <= limit, max(1, int(window_s - (now - start)))

    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        """Count one event. Returns (allowed, retry_after_seconds)."""
        if self._url is None:
            return self._local_hit(key, limit, window_s)
        try:
            r = self._client()
            n = await r.incr(key)
            if n == 1:
                await r.expire(key, window_s)
            ttl = await r.ttl(key)
            self.redis_ok = True
            return n <= limit, max(int(ttl), 1)
        except Exception:  # noqa: BLE001 - Redis down: degrade, do not fail the request
            if self.redis_ok:
                log.warning("rate limiter: Redis unavailable, using in-process fallback")
            self.redis_ok = False
            return self._local_hit(key, max(1, limit // 2), window_s)

    @staticmethod
    def _day_key(user_id: uuid.UUID) -> str:
        return f"tokens:{user_id}:{datetime.now(UTC):%Y%m%d}"

    async def add_tokens(self, user_id: uuid.UUID, n: int) -> None:
        if n <= 0:
            return
        key = self._day_key(user_id)
        if self._url is not None:
            try:
                r = self._client()
                await r.incrby(key, n)
                await r.expire(key, 2 * 86400)
                return
            except Exception:  # noqa: BLE001
                self.redis_ok = False
        start, count = self._local.get(key, (time.monotonic(), 0))
        self._local[key] = (start, count + n)

    async def tokens_used(self, user_id: uuid.UUID) -> int:
        key = self._day_key(user_id)
        if self._url is not None:
            try:
                value = await self._client().get(key)
                return int(value or 0)
            except Exception:  # noqa: BLE001
                self.redis_ok = False
        return self._local.get(key, (0.0, 0))[1]


@lru_cache
def get_limiter() -> Limiter:
    return Limiter(get_settings().redis_url)
