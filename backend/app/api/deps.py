import uuid
from collections.abc import Callable, Iterator

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import tenant_session
from app.services.chat_service import Deps, SessionOpener, StoreFactory
from app.services.embeddings import Embedder, get_embedder
from app.services.generation.llm_client import AnthropicClient, LLMClient
from app.services.retrieval.reranker import Reranker, get_reranker
from app.services.vectorstore import PgVectorStore


def current_user_id(x_user_id: str = Header(...)) -> uuid.UUID:
    """Phase 1 dev auth: the caller states its user id. Replaced by JWT in Phase 5."""
    try:
        return uuid.UUID(x_user_id)
    except ValueError as exc:
        raise HTTPException(401, "invalid X-User-Id") from exc


def db_session(user_id: uuid.UUID = Depends(current_user_id)) -> Iterator[Session]:
    with tenant_session(user_id) as session:
        yield session


def session_opener_dep() -> SessionOpener:
    return tenant_session


def store_factory_dep() -> StoreFactory:
    return PgVectorStore


def settings_dep() -> Settings:
    return get_settings()


def embedder_dep(settings: Settings = Depends(settings_dep)) -> Embedder:
    return get_embedder(settings)


def llm_dep(settings: Settings = Depends(settings_dep)) -> LLMClient:
    if not settings.anthropic_api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured")
    return AnthropicClient(settings.anthropic_api_key)


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
) -> Deps:
    return Deps(open_session, make_store, embedder, reranker, llm, settings)
