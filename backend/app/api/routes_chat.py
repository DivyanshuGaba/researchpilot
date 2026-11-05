import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import (
    current_user_id,
    db_session,
    embedder_dep,
    llm_dep,
    settings_dep,
    vector_store_dep,
)
from app.config import Settings
from app.models import Conversation, Message
from app.services.embeddings import Embedder
from app.services.generation.llm_client import LLMClient
from app.services.generation.rag_chain import answer_question
from app.services.vectorstore import Scope, VectorStore

router = APIRouter(tags=["chat"])


class ChatIn(BaseModel):
    question: str
    conversation_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = None


class CitationOut(BaseModel):
    marker: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int


class ChatOut(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[CitationOut]


@router.post("/chat")
async def chat(
    body: ChatIn,
    user_id: uuid.UUID = Depends(current_user_id),
    session: Session = Depends(db_session),
    embedder: Embedder = Depends(embedder_dep),
    store: VectorStore = Depends(vector_store_dep),
    llm: LLMClient = Depends(llm_dep),
    settings: Settings = Depends(settings_dep),
) -> ChatOut:
    if not body.question.strip():
        raise HTTPException(422, "question is empty")
    config = settings.pipeline
    scope = Scope(user_id, tuple(body.document_ids) if body.document_ids else None)

    def retrieve() -> list:  # type: ignore[type-arg]
        vector = embedder.embed_query(body.question)
        return store.search(vector, config.top_k, scope)

    hits = await run_in_threadpool(retrieve)
    result = await answer_question(body.question, hits, llm, settings, config)

    def persist() -> tuple[uuid.UUID, uuid.UUID]:
        conv = session.get(Conversation, body.conversation_id) if body.conversation_id else None
        if body.conversation_id and conv is None:
            raise HTTPException(404, "conversation not found")
        if conv is None:
            conv = Conversation(user_id=user_id, title=body.question[:80])
            session.add(conv)
            session.flush()
        session.add(Message(conversation_id=conv.id, user_id=user_id, role="user", content=body.question))
        reply = Message(
            conversation_id=conv.id,
            user_id=user_id,
            role="assistant",
            content=result.answer,
            citations=[
                {**c.__dict__, "chunk_id": str(c.chunk_id), "document_id": str(c.document_id)}
                for c in result.citations
            ],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        session.add(reply)
        session.flush()
        return conv.id, reply.id

    conv_id, msg_id = await run_in_threadpool(persist)
    return ChatOut(
        conversation_id=conv_id,
        message_id=msg_id,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
    )
