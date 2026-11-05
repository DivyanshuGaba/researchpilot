import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import (
    EnqueueFn,
    current_user_id,
    db_session,
    enqueue_dep,
    limiter_dep,
    settings_dep,
)
from app.config import Settings
from app.db import get_factory
from app.models import Document, DocumentVersion
from app.services.ingestion.parsers import sniff_type
from app.services.limits import Limiter
from app.services.outbox import add_ingest_event, dispatch_pending

router = APIRouter(prefix="/documents", tags=["documents"])
log = logging.getLogger(__name__)


class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    source_type: str
    status: str
    stage: str = ""
    page_count: int


class UrlIn(BaseModel):
    url: str


def _out(d: Document, stage: str = "") -> DocumentOut:
    return DocumentOut(
        id=d.id,
        title=d.title,
        source_type=d.source_type,
        status=d.status,
        stage=stage,
        page_count=d.page_count,
    )


def _stage(session: Session, document_id: uuid.UUID) -> str:
    ver = session.scalar(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version_no.desc())
    )
    return ver.status if ver else ""


async def _dispatch_now(enqueue: EnqueueFn) -> None:
    """Best effort right after commit; the background dispatcher repairs any failure."""
    try:
        await run_in_threadpool(dispatch_pending, get_factory(), enqueue)
    except Exception:
        log.exception("immediate outbox dispatch failed; the background loop will retry")


def _create(
    session: Session, user_id: uuid.UUID, title: str, source_type: str, location: str, digest: str
) -> tuple[Document, DocumentVersion]:
    doc = Document(user_id=user_id, title=title, source_type=source_type, file_path=location)
    session.add(doc)
    session.flush()
    version = DocumentVersion(document_id=doc.id, user_id=user_id, file_hash=digest)
    session.add(version)
    session.flush()
    add_ingest_event(session, version.id, user_id)  # same transaction: the job cannot be lost
    return doc, version


@router.post("", status_code=202)
async def upload(
    file: UploadFile,
    user_id: uuid.UUID = Depends(current_user_id),
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    enqueue: EnqueueFn = Depends(enqueue_dep),
    limiter: Limiter = Depends(limiter_dep),
) -> DocumentOut:
    ok, retry = await limiter.hit(f"upload:{user_id}", 30, 3600)
    if not ok:
        raise HTTPException(429, "too many uploads, try again later", headers={"Retry-After": str(retry)})
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
    kind = sniff_type(data[:8])
    if kind is None:
        raise HTTPException(415, "only PDF and DOCX files are supported")

    # Random name, outside any served directory; the original name is only kept as the title.
    directory = Path(settings.upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.{kind}"
    path.write_bytes(data)
    title = Path(file.filename or "untitled").stem
    doc, _ = _create(
        session, user_id, title, kind, str(path), hashlib.sha256(data).hexdigest()
    )
    session.commit()  # the document and its outbox row commit together
    await _dispatch_now(enqueue)
    return _out(doc)


@router.post("/url", status_code=202)
async def add_url(
    body: UrlIn,
    user_id: uuid.UUID = Depends(current_user_id),
    session: Session = Depends(db_session),
    enqueue: EnqueueFn = Depends(enqueue_dep),
) -> DocumentOut:
    digest = hashlib.sha256(body.url.encode()).hexdigest()
    doc, _ = _create(session, user_id, body.url, "url", body.url, digest)
    session.commit()
    await _dispatch_now(enqueue)
    return _out(doc)


@router.get("")
def list_documents(session: Session = Depends(db_session)) -> list[DocumentOut]:
    docs = session.scalars(select(Document).order_by(Document.created_at.desc()))
    return [_out(d, _stage(session, d.id)) for d in docs]


@router.get("/{document_id}")
def get_document(document_id: uuid.UUID, session: Session = Depends(db_session)) -> DocumentOut:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return _out(doc, _stage(session, doc.id))


@router.get("/{document_id}/file")
def get_file(document_id: uuid.UUID, session: Session = Depends(db_session)) -> FileResponse:
    """Authenticated file access (RLS hides other tenants' rows, so they get a 404)."""
    doc = session.get(Document, document_id)
    if doc is None or doc.source_type != "pdf" or not Path(doc.file_path).is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(doc.file_path, media_type="application/pdf")
