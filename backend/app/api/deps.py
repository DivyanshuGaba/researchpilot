import uuid
from collections.abc import Callable, Iterator
from functools import lru_cache

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import tenant_session
from app.services.auth import AuthError, decode_access_token
from app.services.cache import CachedEmbedder, RedisCache
from app.services.chat_service import Deps, SessionOpener, StoreFactory
from app.services.embeddings import Embedder, get_embedder
from app.services.generation.llm_client import AnthropicClient, LLMClient, MeteredLLM
from app.services.limits import Limiter, get_limiter
from app.services.retrieval.reranker import Reranker, get_reranker
from app.services.vectorstore import PgVectorStore


def settings_dep() -> Settings:
    return get_settings()


def limiter_dep() -> Limiter:
    return get_limiter()


_bearer = HTTPBearer(auto_error=False)


def current_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(settings_dep),
) -> uuid.UUID:
    """The signed-in user, from a short-lived JWT access token (Authorization: Bearer)."""
    if creds is None:
        raise HTTPException(401, "not signed in", headers={"WWW-Authenticate": "Bearer"})
    try:
        return decode_access_token(creds.credentials, settings)
    except AuthError as exc:
        raise HTTPException(
            401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
        ) from exc


def db_session(user_id: uuid.UUID = Depends(current_user_id)) -> Iterator[Session]:
    with tenant_session(user_id) as session:
        yield session


def session_opener_dep() -> SessionOpener:
    return tenant_session


def store_factory_dep() -> StoreFactory:
    return PgVectorStore


@lru_cache
def _shared_cache(url: str, ttl: int, enabled: bool) -> RedisCache:
    return RedisCache(url, ttl, enabled)


def cache_dep(settings: Settings = Depends(settings_dep)) -> RedisCache:
    return _shared_cache(settings.redis_url, settings.cache_ttl_seconds, settings.cache_enabled)


def embedder_dep(
    settings: Settings = Depends(settings_dep), cache: RedisCache = Depends(cache_dep)
) -> Embedder:
    base = get_embedder(settings)
    return CachedEmbedder(base, cache) if cache.enabled else base


@lru_cache
def _shared_client(api_key: str, max_concurrency: int) -> AnthropicClient:
    """One client (and one concurrency limit) per process, shared by every request."""
    return AnthropicClient(api_key, max_concurrency=max_concurrency)


def llm_dep(settings: Settings = Depends(settings_dep)) -> LLMClient:
    if not settings.anthropic_api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured")
    return _shared_client(settings.anthropic_api_key, settings.llm_max_concurrency)


EnqueueFn = Callable[[str, str], None]


def enqueue_dep() -> EnqueueFn:
    from app.workers.tasks import ingest_document

    return lambda version_id, user_id: ingest_document.delay(version_id, user_id)


def reranker_dep(settings: Settings = Depends(settings_dep)) -> Reranker | None:
    return get_reranker(settings)


def chat_deps(
    open_session: SessionOpener = Depends(session_opener_dep),
    make_store: StoreFactory = Depends(store_factory_dep),
    embedder: Embedder = Depends(embedder_dep),
    reranker: Reranker | None = Depends(reranker_dep),
    llm: LLMClient = Depends(llm_dep),
    settings: Settings = Depends(settings_dep),
    limiter: Limiter = Depends(limiter_dep),
    cache: RedisCache = Depends(cache_dep),
) -> Deps:
    # A fresh metered wrapper per request, so usage and cost are per request.
    return Deps(open_session, make_store, embedder, reranker, MeteredLLM(llm), settings, limiter, cache)


async def enforce_rate_limit(
    user_id: uuid.UUID = Depends(current_user_id),
    settings: Settings = Depends(settings_dep),
    limiter: Limiter = Depends(limiter_dep),
) -> None:
    """Per-user request rate for endpoints that call the model, plus the daily token budget."""
    ok, retry = await limiter.hit(f"rl:{user_id}", settings.rate_limit_per_minute, 60)
    if not ok:
        raise HTTPException(
            429, "too many requests, slow down", headers={"Retry-After": str(retry)}
        )
    if settings.daily_token_budget and await limiter.tokens_used(user_id) >= settings.daily_token_budget:
        raise HTTPException(429, "daily token budget reached, try again tomorrow")
