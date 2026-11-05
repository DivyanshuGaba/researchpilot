import os

# Integration tests talk to the compose Postgres on host port 5433 unless overridden.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://rp_app:rp_app@localhost:5433/researchpilot"
)


import pytest

from app.services.limits import Limiter


@pytest.fixture(autouse=True)
def _local_limiter(monkeypatch):
    """Every test gets a fresh in-process limiter: no Redis dependency, no shared counters."""
    limiter = Limiter(None)
    monkeypatch.setattr("app.api.deps.get_limiter", lambda: limiter)
    return limiter


@pytest.fixture(autouse=True)
def _no_response_cache(monkeypatch):
    """Caching is off unless a test asks for it, so repeated identical questions in a test
    exercise the pipeline instead of being served from Redis."""
    from app.services.cache import RedisCache

    monkeypatch.setattr(
        "app.api.deps._shared_cache", lambda url, ttl, enabled: RedisCache(url, ttl, False)
    )
