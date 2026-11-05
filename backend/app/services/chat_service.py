"""Streaming chat orchestration (architecture.md section 8).

Events: status, token, citation, done, error. A heartbeat comment is sent while idle so proxies
keep the connection open. If the client disconnects, the partial answer is saved as incomplete.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from typing import Any

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Conversation, Message
from app.services.embeddings import Embedder
from app.services.generation.llm_client import LLMClient
from app.services.generation.rag_chain import Citation, RagAnswer, stream_answer
from app.services.vectorstore import Hit, Scope, VectorStore

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0
SessionOpener = Callable[[uuid.UUID], AbstractContextManager[Session]]
StoreFactory = Callable[[Session], VectorStore]


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def persist_exchange(
    open_session: SessionOpener,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    question: str,
    answer: str,
    citations: list[Citation],
    input_tokens: int,
    output_tokens: int,
    incomplete: bool,
) -> tuple[uuid.UUID, uuid.UUID]:
    with open_session(user_id) as session:
        conv = session.get(Conversation, conversation_id) if conversation_id else None
        if conv is None:
            conv = Conversation(user_id=user_id, title=question[:80])
            session.add(conv)
            session.flush()
        session.add(Message(conversation_id=conv.id, user_id=user_id, role="user", content=question))
        reply = Message(
            conversation_id=conv.id,
            user_id=user_id,
            role="assistant",
            content=answer,
            citations=[c.to_json() for c in citations],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            incomplete=incomplete,
        )
        session.add(reply)
        session.flush()
        return conv.id, reply.id


async def stream_chat(
    *,
    user_id: uuid.UUID,
    question: str,
    conversation_id: uuid.UUID | None,
    document_ids: tuple[uuid.UUID, ...] | None,
    open_session: SessionOpener,
    make_store: StoreFactory,
    embedder: Embedder,
    llm: LLMClient,
    settings: Settings,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(
        _produce(
            queue, user_id, question, conversation_id, document_ids,
            open_session, make_store, embedder, llm, settings,
        )
    )
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is None:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()  # client went away; _produce saves the partial answer


async def _produce(
    queue: asyncio.Queue[str | None],
    user_id: uuid.UUID,
    question: str,
    conversation_id: uuid.UUID | None,
    document_ids: tuple[uuid.UUID, ...] | None,
    open_session: SessionOpener,
    make_store: StoreFactory,
    embedder: Embedder,
    llm: LLMClient,
    settings: Settings,
) -> None:
    config = settings.pipeline
    partial: list[str] = []
    final: RagAnswer | None = None
    started = asyncio.get_running_loop().time()
    try:
        await queue.put(sse("status", {"stage": "searching"}))

        def retrieve() -> list[Hit]:
            vector = embedder.embed_query(question)
            with open_session(user_id) as session:
                return make_store(session).search(
                    vector, config.top_k, Scope(user_id, document_ids)
                )

        hits = await run_in_threadpool(retrieve)
        async for item in stream_answer(question, hits, llm, settings, config):
            if isinstance(item, RagAnswer):
                final = item
            else:
                if item.event == "token":
                    partial.append(str(item.data["text"]))
                await queue.put(sse(item.event, item.data))
        assert final is not None
        conv_id, msg_id = await run_in_threadpool(
            persist_exchange, open_session, user_id, conversation_id, question,
            final.answer, final.citations, final.input_tokens, final.output_tokens, False,
        )
        latency_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        await queue.put(
            sse(
                "done",
                {
                    "conversation_id": str(conv_id),
                    "message_id": str(msg_id),
                    "answer": final.answer,  # authoritative: invalid citation markers removed
                    "input_tokens": final.input_tokens,
                    "output_tokens": final.output_tokens,
                    "latency_ms": latency_ms,
                },
            )
        )
    except asyncio.CancelledError:
        text = "".join(partial)
        await asyncio.shield(
            run_in_threadpool(
                persist_exchange, open_session, user_id, conversation_id, question,
                text, [], 0, 0, True,
            )
        )
        raise
    except Exception:
        log.exception("chat stream failed")
        await queue.put(sse("error", {"code": "internal", "message": "Something went wrong."}))
    finally:
        await queue.put(None)
