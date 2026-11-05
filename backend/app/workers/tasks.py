import uuid

from app.config import get_settings
from app.db import tenant_session
from app.services.embeddings import get_embedder
from app.services.ingestion.pipeline import ingest_version
from app.workers.celery_app import celery_app


@celery_app.task(
    name="ingest_document",
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=3,
)
def ingest_document(version_id: str, user_id: str) -> None:
    settings = get_settings()
    with tenant_session(uuid.UUID(user_id)) as session:
        ingest_version(session, uuid.UUID(version_id), get_embedder(settings), settings.pipeline)
