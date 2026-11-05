import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.api.deps import chat_deps, current_user_id, enforce_rate_limit
from app.models import Chunk, Document, DocumentVersion, FlashcardSet, Quiz, Summary
from app.schemas.study import (
    FlashcardRequest,
    FlashcardSetOut,
    QuizOut,
    QuizRequest,
    SummaryOut,
    SummaryRequest,
)
from app.services.chat_service import Deps, record_usage
from app.services.generation.llm_client import LLMUnavailable
from app.services.generation.study_tools import (
    Section,
    StudyGenerationError,
    generate_flashcards,
    generate_quiz,
    summarize_document,
)

router = APIRouter(prefix="/study", tags=["study"])
log = logging.getLogger(__name__)

SAFE_GENERATION_ERROR = "The model could not produce a valid result. Please try again."
SAFE_UPSTREAM_ERROR = "The model could not be reached. Please try again."


def _unavailable(exc: LLMUnavailable) -> HTTPException:
    status = {"rate_limited": 429, "timeout": 504}.get(exc.kind, 503)
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return HTTPException(status, exc.message, headers=headers)


def _load(
    deps: Deps, user_id: uuid.UUID, document_id: uuid.UUID, level: str = "parent"
) -> tuple[uuid.UUID, list[Section]]:
    """Latest ready version's passages. Summaries use parent chunks (sections); quizzes and
    flashcards use child chunks, which never span pages, so a cited page is exact and the
    source can be highlighted. RLS makes other tenants' ids a 404."""
    with deps.open_session(user_id) as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        if doc.status != "ready":
            raise HTTPException(409, f"document is not ready yet (status: {doc.status})")
        version = s.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_no.desc())
        )
        if version is None:
            raise HTTPException(409, "document has no indexed content")
        rows = s.scalars(
            select(Chunk)
            .where(Chunk.version_id == version.id, Chunk.level == level)
            .order_by(Chunk.chunk_index)
        ).all()
        sections = [
            Section(c.id, c.page_start, c.page_end, c.content, tuple(c.bbox_list or [])) for c in rows
        ]
    if not sections:
        raise HTTPException(409, "document has no indexed content")
    return version.id, sections


def _save(
    deps: Deps, user_id: uuid.UUID, model: type[Quiz] | type[FlashcardSet],
    document_id: uuid.UUID, payload: dict[str, Any],
) -> Quiz | FlashcardSet:
    with deps.open_session(user_id) as s:
        row = model(document_id=document_id, user_id=user_id, payload=payload)
        s.add(row)
        s.flush()
        s.refresh(row)
        return row


def _quiz_out(row: Quiz) -> QuizOut:
    p = row.payload
    return QuizOut(
        id=row.id, document_id=row.document_id, requested=p["requested"],
        questions=p["questions"], created_at=row.created_at,
    )


def _cards_out(row: FlashcardSet) -> FlashcardSetOut:
    p = row.payload
    return FlashcardSetOut(
        id=row.id, document_id=row.document_id, requested=p["requested"],
        cards=p["cards"], created_at=row.created_at,
    )


@router.post("/quizzes", status_code=201, dependencies=[Depends(enforce_rate_limit)])
async def create_quiz(
    body: QuizRequest, user_id: uuid.UUID = Depends(current_user_id), deps: Deps = Depends(chat_deps)
) -> QuizOut:
    version_id, sections = await run_in_threadpool(
        _load, deps, user_id, body.document_id, "child"
    )
    try:
        gen = await generate_quiz(deps.llm, deps.settings, sections, body.num_questions)
        await record_usage(deps, user_id)
    except LLMUnavailable as exc:
        await record_usage(deps, user_id)
        raise _unavailable(exc) from exc
    except StudyGenerationError as exc:
        await record_usage(deps, user_id)
        log.warning("quiz generation failed: %s", exc)
        raise HTTPException(502, SAFE_GENERATION_ERROR) from exc
    except Exception as exc:
        log.exception("quiz generation error")
        raise HTTPException(502, SAFE_UPSTREAM_ERROR) from exc
    payload = {
        "requested": gen.requested, "questions": gen.questions, "attempts": gen.attempts,
        "version_id": str(version_id), "model": deps.settings.llm_model,
    }
    row = await run_in_threadpool(_save, deps, user_id, Quiz, body.document_id, payload)
    assert isinstance(row, Quiz)
    return _quiz_out(row)


@router.get("/quizzes")
def list_quizzes(
    document_id: uuid.UUID | None = None,
    user_id: uuid.UUID = Depends(current_user_id),
    deps: Deps = Depends(chat_deps),
) -> list[QuizOut]:
    with deps.open_session(user_id) as s:
        stmt = select(Quiz).order_by(Quiz.created_at.desc()).limit(20)
        if document_id:
            stmt = stmt.where(Quiz.document_id == document_id)
        return [_quiz_out(r) for r in s.scalars(stmt)]


@router.get("/quizzes/{quiz_id}")
def get_quiz(
    quiz_id: uuid.UUID, user_id: uuid.UUID = Depends(current_user_id), deps: Deps = Depends(chat_deps)
) -> QuizOut:
    with deps.open_session(user_id) as s:
        row = s.get(Quiz, quiz_id)
        if row is None:
            raise HTTPException(404, "quiz not found")
        return _quiz_out(row)


@router.post("/flashcards", status_code=201, dependencies=[Depends(enforce_rate_limit)])
async def create_flashcards(
    body: FlashcardRequest,
    user_id: uuid.UUID = Depends(current_user_id),
    deps: Deps = Depends(chat_deps),
) -> FlashcardSetOut:
    version_id, sections = await run_in_threadpool(
        _load, deps, user_id, body.document_id, "child"
    )
    try:
        gen = await generate_flashcards(deps.llm, deps.settings, sections, body.num_cards)
        await record_usage(deps, user_id)
    except LLMUnavailable as exc:
        await record_usage(deps, user_id)
        raise _unavailable(exc) from exc
    except StudyGenerationError as exc:
        await record_usage(deps, user_id)
        log.warning("flashcard generation failed: %s", exc)
        raise HTTPException(502, SAFE_GENERATION_ERROR) from exc
    except Exception as exc:
        log.exception("flashcard generation error")
        raise HTTPException(502, SAFE_UPSTREAM_ERROR) from exc
    payload = {
        "requested": gen.requested, "cards": gen.cards, "attempts": gen.attempts,
        "version_id": str(version_id), "model": deps.settings.llm_model,
    }
    row = await run_in_threadpool(_save, deps, user_id, FlashcardSet, body.document_id, payload)
    assert isinstance(row, FlashcardSet)
    return _cards_out(row)


@router.get("/flashcards")
def list_flashcards(
    document_id: uuid.UUID | None = None,
    user_id: uuid.UUID = Depends(current_user_id),
    deps: Deps = Depends(chat_deps),
) -> list[FlashcardSetOut]:
    with deps.open_session(user_id) as s:
        stmt = select(FlashcardSet).order_by(FlashcardSet.created_at.desc()).limit(20)
        if document_id:
            stmt = stmt.where(FlashcardSet.document_id == document_id)
        return [_cards_out(r) for r in s.scalars(stmt)]


@router.get("/flashcards/{set_id}")
def get_flashcards(
    set_id: uuid.UUID, user_id: uuid.UUID = Depends(current_user_id), deps: Deps = Depends(chat_deps)
) -> FlashcardSetOut:
    with deps.open_session(user_id) as s:
        row = s.get(FlashcardSet, set_id)
        if row is None:
            raise HTTPException(404, "flashcard set not found")
        return _cards_out(row)


# --- summaries (cached per document version) --------------------------------------------------


def _load_cached_summaries(
    deps: Deps, user_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[str | None, dict[uuid.UUID, str]]:
    with deps.open_session(user_id) as s:
        rows = s.scalars(select(Summary).where(Summary.version_id == version_id)).all()
    doc = next((r.content for r in rows if r.kind == "document"), None)
    return doc, {r.chunk_id: r.content for r in rows if r.kind == "section" and r.chunk_id}


def _save_summaries(
    deps: Deps, user_id: uuid.UUID, document_id: uuid.UUID, version_id: uuid.UUID,
    new_sections: dict[uuid.UUID, str], summary: str,
) -> None:
    rows: list[dict[str, Any]] = [
        {"document_id": document_id, "version_id": version_id, "user_id": user_id,
         "kind": "section", "chunk_id": cid, "content": text}
        for cid, text in new_sections.items()
    ]
    rows.append({"document_id": document_id, "version_id": version_id, "user_id": user_id,
                 "kind": "document", "chunk_id": None, "content": summary})  # fmt: skip
    with deps.open_session(user_id) as s:
        # A concurrent request may have saved the same rows first; the partial unique indexes
        # make that a no-op instead of an error.
        s.execute(insert(Summary).values(rows).on_conflict_do_nothing())


@router.post("/summaries", dependencies=[Depends(enforce_rate_limit)])
async def create_summary(
    body: SummaryRequest, user_id: uuid.UUID = Depends(current_user_id), deps: Deps = Depends(chat_deps)
) -> SummaryOut:
    version_id, sections = await run_in_threadpool(_load, deps, user_id, body.document_id)
    cached_doc, cached_sections = await run_in_threadpool(
        _load_cached_summaries, deps, user_id, version_id
    )
    if cached_doc is not None:
        return SummaryOut(
            document_id=body.document_id, summary=cached_doc, cached=True, sections=len(sections)
        )
    try:
        summary, section_map, truncated = await summarize_document(
            deps.llm, deps.settings, sections, cached_sections
        )
        await record_usage(deps, user_id)
    except LLMUnavailable as exc:
        await record_usage(deps, user_id)
        raise _unavailable(exc) from exc
    except Exception as exc:
        log.exception("summary generation error")
        raise HTTPException(502, SAFE_UPSTREAM_ERROR) from exc
    new = {k: v for k, v in section_map.items() if k not in cached_sections}
    await run_in_threadpool(_save_summaries, deps, user_id, body.document_id, version_id, new, summary)
    return SummaryOut(
        document_id=body.document_id, summary=summary, cached=False, sections=len(sections),
        truncated=truncated, meta={"sections_summarized": len(new), "sections_reused": len(cached_sections)},
    )


@router.get("/summaries/{document_id}")
def get_summary(
    document_id: uuid.UUID, user_id: uuid.UUID = Depends(current_user_id), deps: Deps = Depends(chat_deps)
) -> SummaryOut:
    with deps.open_session(user_id) as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        version = s.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_no.desc())
        )
        row = (
            s.scalar(
                select(Summary).where(Summary.version_id == version.id, Summary.kind == "document")
            )
            if version
            else None
        )
        if row is None:
            raise HTTPException(404, "no summary yet")
        return SummaryOut(document_id=document_id, summary=row.content, cached=True, sections=0)
