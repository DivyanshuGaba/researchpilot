import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import chat_deps, current_user_id, enforce_rate_limit
from app.services.chat_service import (
    ChatLLMError,
    ChatRequestError,
    Deps,
    answer_turn,
    finish_turn,
    persist_exchange,
    record_usage,
    stream_chat,
    usage_of,
    validate_request,
)

router = APIRouter(tags=["chat"])


class ChatIn(BaseModel):
    question: str
    conversation_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = None
    mode: Literal["qa", "compare", "agent"] = "qa"

    def scope_ids(self) -> tuple[uuid.UUID, ...] | None:
        return tuple(self.document_ids) if self.document_ids else None


class CitationOut(BaseModel):
    marker: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int
    quote: str = ""
    bbox_list: list[Any] = []


class ChatOut(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[CitationOut]
    closest: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}


def _validate(body: ChatIn) -> None:
    try:
        validate_request(body.question, body.mode, body.scope_ids())
    except ChatRequestError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/chat", dependencies=[Depends(enforce_rate_limit)])
async def chat(
    body: ChatIn,
    user_id: uuid.UUID = Depends(current_user_id),
    deps: Deps = Depends(chat_deps),
) -> ChatOut:
    _validate(body)
    started = time.perf_counter()
    try:
        out = await answer_turn(
            deps, user_id, body.conversation_id, body.question, body.scope_ids(), body.mode
        )
    except ChatLLMError as exc:
        await record_usage(deps, user_id)
        await finish_turn(
            deps, user_id, body.mode, body.question, "", exc.turn,
            int((time.perf_counter() - started) * 1000), error=exc.kind,
        )
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(
            exc.status_code,
            detail={"message": exc.message, "code": exc.kind, "closest": exc.closest},
            headers=headers,
        ) from exc
    result = out.answer
    usage = usage_of(deps)
    latency_ms = int((time.perf_counter() - started) * 1000)
    conv_id, msg_id = await run_in_threadpool(
        persist_exchange, deps.open_session, user_id, body.conversation_id, body.question,
        result.answer, result.citations, usage.input_tokens or result.input_tokens,
        usage.output_tokens or result.output_tokens, False, out.tool_calls, latency_ms,
        usage.cost_usd,
    )
    await record_usage(deps, user_id)
    await finish_turn(
        deps, user_id, body.mode, body.question, result.answer, out.turn, latency_ms,
        conv_id, msg_id,
    )
    return ChatOut(
        conversation_id=conv_id,
        message_id=msg_id,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        closest=result.closest,
        meta=out.meta,
    )


@router.post("/chat/stream", dependencies=[Depends(enforce_rate_limit)])
async def chat_stream(
    body: ChatIn,
    user_id: uuid.UUID = Depends(current_user_id),
    deps: Deps = Depends(chat_deps),
) -> StreamingResponse:
    _validate(body)
    events = stream_chat(
        deps=deps,
        user_id=user_id,
        question=body.question,
        conversation_id=body.conversation_id,
        document_ids=body.scope_ids(),
        mode=body.mode,
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
