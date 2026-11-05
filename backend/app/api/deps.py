import uuid
from collections.abc import Callable, Iterator

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import tenant_session
from app.services.embeddings import Embedder, get_embedder
from app.services.generation.llm_client import AnthropicClient, LLMClient
from app.services.vectorstore import PgVectorStore, VectorStore


def current_user_id(x_user_id: str = Header(...)) -> uuid.UUID:
    """Phase 1 dev auth: the caller states its user id. Replaced by JWT in Phase 5."""
    try:
        return uuid.UUID(x_user_id)
    except ValueError as exc:
        raise HTTPException(401, "invalid X-User-Id") from exc


def db_session(user_id: uuid.UUID = Depends(current_user_id)) -> Iterator[Session]:
    with tenant_session(user_id) as session:
        yield session


def vector_store_dep(session: Session = Depends(db_session)) -> VectorStore:
    return PgVectorStore(session)


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
