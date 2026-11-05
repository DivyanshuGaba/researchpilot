import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.config import PipelineConfig
from app.models import Chunk, Document, DocumentVersion, IngestionJob
from app.services.embeddings import Embedder
from app.services.ingestion.chunkers import get_chunker
from app.services.ingestion.parsers import ParseError, parse
from app.services.vectorstore import PgVectorStore

log = logging.getLogger(__name__)

INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "you are now",
    "system prompt",
)


def flag_injection(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in INJECTION_PATTERNS)


def _stage(session: Session, version: DocumentVersion, doc: Document, stage: str) -> None:
    version.status = stage
    doc.status = "processing" if stage != "ready" else "ready"
    session.add(
        IngestionJob(
            version_id=version.id, user_id=version.user_id, stage=stage, status="done", attempts=1
        )
    )
    session.flush()


def ingest_version(
    session: Session, version_id: uuid.UUID, embedder: Embedder, config: PipelineConfig
) -> None:
    """parse -> chunk -> embed -> store. Idempotent: old chunks for the version are replaced."""
    version = session.get(DocumentVersion, version_id)
    if version is None:
        raise ValueError("unknown version")
    doc = session.get(Document, version.document_id)
    assert doc is not None
    try:
        _stage(session, version, doc, "parsing")
        parsed = parse(doc.source_type, doc.file_path)
        doc.page_count = parsed.page_count
        if doc.source_type == "url":
            doc.title = parsed.title
        # For files the title stays what the user uploaded: the stored name is random and
        # embedded PDF metadata titles are often junk.

        _stage(session, version, doc, "chunking")
        drafts = get_chunker(config).chunk(parsed.blocks)

        _stage(session, version, doc, "embedding")
        children = [d for d in drafts if d.level == "child"]
        vectors = embedder.embed_documents([c.text for c in children])
        vec_by_child = dict(zip((id(c) for c in children), vectors, strict=True))

        PgVectorStore(session).delete_version(version.id)
        parent_ids: dict[int, uuid.UUID] = {}
        for d in drafts:
            if d.level == "parent":
                parent_ids[d.index] = uuid.uuid4()
        for d in drafts:
            is_child = d.level == "child"
            session.add(
                Chunk(
                    id=parent_ids[d.index] if not is_child else uuid.uuid4(),
                    version_id=version.id,
                    document_id=doc.id,
                    user_id=version.user_id,
                    parent_id=parent_ids[d.parent_index] if is_child and d.parent_index is not None else None,
                    level=d.level,
                    chunk_index=d.index,
                    content=d.text,
                    page_start=d.page_start,
                    page_end=d.page_end,
                    bbox_list=d.bbox_list,
                    heading_path=d.heading_path,
                    token_count=d.token_count,
                    embedding=vec_by_child.get(id(d)),
                    embedding_model_id=embedder.model_id if is_child else None,
                    meta={"injection_flag": flag_injection(d.text)},
                )
            )
        session.flush()
        _stage(session, version, doc, "indexed")
        _stage(session, version, doc, "ready")
    except ParseError as exc:
        _fail(session, version, doc, str(exc))
    except Exception as exc:
        log.exception("ingestion failed")
        _fail(session, version, doc, f"internal error: {type(exc).__name__}")
        raise


def _fail(session: Session, version: DocumentVersion, doc: Document, reason: str) -> None:
    version.status = "failed"
    doc.status = "failed"
    session.add(
        IngestionJob(
            version_id=version.id,
            user_id=version.user_id,
            stage="failed",
            status="failed",
            error=reason,
            finished_at=datetime.now(UTC),
        )
    )
    session.flush()
