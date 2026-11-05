"""Chat orchestration shared by /chat and /chat/stream (architecture.md sections 5 and 8).

prepare_turn: memory -> rewrite -> retrieve -> rerank -> confidence gate.
stream_chat:  SSE events status, token, citation, closest, done, error, plus a heartbeat.
If the client disconnects, the partial answer is saved as incomplete.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Conversation, Document, Message
from app.services import memory
from app.services.embeddings import Embedder
from app.services.generation.agent import AgentLimits, AgentResult, run_agent
from app.services.generation.llm_client import LLMClient, LLMUnavailable, Usage
from app.services.generation.rag_chain import (
    Citation,
    RagAnswer,
    StreamEvent,
    answer_question,
    closest_json,
    pack_context,
    stream_answer,
)
from app.services.generation.tools import ToolBox
from app.services.limits import Limiter
from app.services.retrieval.query_rewrite import rewrite_query
from app.services.retrieval.reranker import Reranker
from app.services.retrieval.service import Retrieval, finalize, gather_candidates
from app.services.turn import Group, Turn
from app.services.vectorstore import Hit, Scope, VectorStore

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0
SessionOpener = Callable[[uuid.UUID], AbstractContextManager[Session]]
StoreFactory = Callable[[Session], VectorStore]
StatusFn = Callable[[str], Awaitable[None]]


class ChatLLMError(Exception):
    """The model could not be reached. Carries the passages that were retrieved, so the user
    still gets something useful (architecture.md section 13)."""

    def __init__(self, cause: LLMUnavailable, closest: list[dict[str, Any]]) -> None:
        super().__init__(cause.kind)
        self.kind, self.message, self.retry_after = cause.kind, cause.message, cause.retry_after
        self.closest = closest

    @property
    def status_code(self) -> int:
        return {"rate_limited": 429, "timeout": 504}.get(self.kind, 503)


def retrieved_passages(turn: Turn, limit: int = 3) -> list[dict[str, Any]]:
    hits = [h for g in turn.groups for h in g.hits][:limit]
    return closest_json(hits)


class ChatRequestError(ValueError):
    """The request itself is invalid (maps to HTTP 422)."""


@dataclass
class Deps:
    open_session: SessionOpener
    make_store: StoreFactory
    embedder: Embedder
    reranker: Reranker | None
    llm: LLMClient
    settings: Settings
    limiter: Limiter | None = None


def validate_request(question: str, mode: str, document_ids: tuple[uuid.UUID, ...] | None) -> None:
    if not question.strip():
        raise ChatRequestError("question is empty")
    if mode == "compare" and (document_ids is None or len(set(document_ids)) < 2):
        raise ChatRequestError("compare mode needs at least two selected documents")


def usage_of(deps: Deps) -> Usage:
    """Totals for this request when the LLM is metered (it is in the API; fakes are not)."""
    return getattr(deps.llm, "usage", None) or Usage()


async def record_usage(deps: Deps, user_id: uuid.UUID) -> None:
    if deps.limiter is not None:
        await deps.limiter.add_tokens(user_id, usage_of(deps).total_tokens)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _no_status(_stage: str) -> None:
    return None


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _merge(into: dict[str, int], more: dict[str, int]) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v


async def load_memory(
    deps: Deps, user_id: uuid.UUID, conversation_id: uuid.UUID | None, turn: Turn
) -> tuple[memory.HistoryState, str]:
    """Recent messages verbatim, older ones folded into the rolling summary."""
    cfg = deps.settings.pipeline

    def load() -> memory.HistoryState:
        with deps.open_session(user_id) as s:
            return memory.load_history(s, conversation_id, cfg)

    t = time.perf_counter()
    state = await run_in_threadpool(load)
    summary = state.summary
    if state.to_fold and state.conversation_id is not None:
        conv_id, folded = state.conversation_id, state.summarized_count + len(state.to_fold)
        new_summary = await memory.fold_summary(state, deps.llm, deps.settings)
        if new_summary is None:
            turn.degraded.append("summary")
        else:
            summary = new_summary

            def save() -> None:
                with deps.open_session(user_id) as s:
                    memory.save_summary(s, conv_id, new_summary, folded)

            await run_in_threadpool(save)
    turn.timings_ms["memory"] = _ms(t)
    return state, summary


async def prepare_turn(
    deps: Deps,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    question: str,
    document_ids: tuple[uuid.UUID, ...] | None,
    mode: str = "qa",
    on_status: StatusFn = _no_status,
) -> Turn:
    cfg = deps.settings.pipeline
    min_score = deps.settings.retrieval_min_score
    turn = Turn(question=question, mode=mode)

    state, summary = await load_memory(deps, user_id, conversation_id, turn)
    turn.history_text = memory.history_text(summary, state.recent)

    t = time.perf_counter()
    if state.recent or summary or mode == "compare":
        await on_status("rewriting")
    turn.standalone, used_llm = await rewrite_query(
        question, state, summary, deps.llm, deps.settings, mode
    )
    if used_llm or turn.standalone != question:
        turn.timings_ms["rewrite"] = _ms(t)

    await on_status("searching")
    if mode == "compare":
        assert document_ids is not None
        await _retrieve_compare(deps, user_id, turn, tuple(dict.fromkeys(document_ids)), min_score)
        return turn

    scope = Scope(user_id, document_ids)

    def gather():  # type: ignore[no-untyped-def]
        with deps.open_session(user_id) as s:
            return gather_candidates(deps.make_store(s), deps.embedder, turn.standalone, scope, cfg)

    cands = await run_in_threadpool(gather)
    if cfg.retrieval_mode == "hybrid_rerank":
        await on_status("reranking")
    result = await run_in_threadpool(
        finalize, cands, turn.standalone, deps.reranker, cfg, min_score
    )
    turn.groups = [Group(None, result.hits)]
    turn.gated, turn.closest, turn.top_score = result.gated, result.closest, result.top_score
    _merge(turn.timings_ms, result.timings_ms)
    turn.degraded.extend(result.degraded)
    return turn


async def _retrieve_compare(
    deps: Deps, user_id: uuid.UUID, turn: Turn, doc_ids: tuple[uuid.UUID, ...], min_score: float
) -> None:
    """Retrieve per document so one paper cannot crowd out the other (architecture 5.7)."""
    cfg = deps.settings.pipeline
    per_doc_cfg = cfg.model_copy(update={"top_k": cfg.compare_per_doc})

    def run() -> tuple[dict[uuid.UUID, str], list[tuple[uuid.UUID, Retrieval]]]:
        with deps.open_session(user_id) as s:
            store = deps.make_store(s)
            titles = {
                d.id: d.title for d in s.scalars(select(Document).where(Document.id.in_(doc_ids)))
            }
            out = []
            for doc_id in doc_ids:
                cands = gather_candidates(
                    store, deps.embedder, turn.standalone, Scope(user_id, (doc_id,)), per_doc_cfg
                )
                out.append(
                    (doc_id, finalize(cands, turn.standalone, deps.reranker, per_doc_cfg, min_score))
                )
            return titles, out

    titles, results = await run_in_threadpool(run)
    for doc_id, r in results:
        turn.groups.append(Group(titles.get(doc_id, "Document"), r.hits))
        turn.closest.extend(r.closest)
        _merge(turn.timings_ms, r.timings_ms)
        turn.degraded.extend(x for x in r.degraded if x not in turn.degraded)
    scores = [r.top_score for _, r in results if r.top_score is not None]
    turn.top_score = max(scores) if scores else None
    turn.gated = not any(g.hits for g in turn.groups)


EmitFn = Callable[[StreamEvent], Awaitable[None]]


async def _no_emit(_ev: StreamEvent) -> None:
    return None


async def run_agent_turn(
    deps: Deps,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    question: str,
    document_ids: tuple[uuid.UUID, ...] | None,
    emit: EmitFn = _no_emit,
    http: Any = None,
    limits: AgentLimits | None = None,
) -> tuple[AgentResult, Turn]:
    """Let the model decide which tools to call. Retrieval inside search_documents keeps the
    same rerank and confidence gate as plain Q&A, so a gated search returns nothing."""
    cfg = deps.settings.pipeline
    turn = Turn(question=question, mode="agent")
    state, summary = await load_memory(deps, user_id, conversation_id, turn)
    turn.history_text = memory.history_text(summary, state.recent)

    by_marker: dict[str, Hit] = {}
    seen: set[uuid.UUID] = set()

    async def search(query: str, ids: tuple[uuid.UUID, ...] | None) -> str:
        scope = Scope(user_id, ids)

        def gather():  # type: ignore[no-untyped-def]
            with deps.open_session(user_id) as s:
                return gather_candidates(deps.make_store(s), deps.embedder, query, scope, cfg)

        cands = await run_in_threadpool(gather)
        r = await run_in_threadpool(
            finalize, cands, query, deps.reranker, cfg, deps.settings.retrieval_min_score
        )
        if not r.hits:
            return "No relevant passages were found for this query."
        rendered, _ = pack_context(
            [Group(None, r.hits)], cfg.context_token_budget // 2, False, by_marker, seen
        )
        return "\n".join(rendered) or "Those passages were already returned by an earlier search."

    toolbox = ToolBox(search, document_ids, http)
    result = await run_agent(
        question=question,
        history_text=turn.history_text,
        toolbox=toolbox,
        by_marker=by_marker,
        llm=deps.llm,
        settings=deps.settings,
        emit=emit,
        **({"limits": limits} if limits else {}),
    )
    return result, turn


def agent_meta(result: AgentResult) -> dict[str, Any]:
    return {
        "mode": "agent",
        "tool_calls": result.tool_log,
        "stopped": result.stopped,
        "degraded": [],
    }


@dataclass
class TurnAnswer:
    answer: RagAnswer
    meta: dict[str, Any]
    tool_calls: list[dict[str, Any]]


async def answer_turn(
    deps: Deps,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    question: str,
    document_ids: tuple[uuid.UUID, ...] | None,
    mode: str,
) -> TurnAnswer:
    """Non-streaming answer for any mode (used by POST /chat)."""
    if mode == "agent":
        try:
            res, _ = await run_agent_turn(deps, user_id, conversation_id, question, document_ids)
        except LLMUnavailable as exc:
            raise ChatLLMError(exc, []) from exc
        answer = RagAnswer(res.answer, res.citations, [], res.input_tokens, res.output_tokens)
        return TurnAnswer(answer, agent_meta(res), res.tool_log)
    turn = await prepare_turn(deps, user_id, conversation_id, question, document_ids, mode)
    try:
        answer = await answer_question(turn, deps.llm, deps.settings, deps.settings.pipeline)
    except LLMUnavailable as exc:
        raise ChatLLMError(exc, retrieved_passages(turn)) from exc
    return TurnAnswer(answer, turn_meta(turn), [])


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
    tool_calls: list[dict[str, Any]] | None = None,
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
            tool_calls=tool_calls or [],
        )
        session.add(reply)
        session.flush()
        return conv.id, reply.id


def turn_meta(turn: Turn) -> dict[str, Any]:
    return {
        "mode": turn.mode,
        "standalone_query": turn.standalone,
        "top_score": None if turn.top_score is None else round(turn.top_score, 3),
        "timings_ms": turn.timings_ms,
        "degraded": turn.degraded,
    }


async def stream_chat(
    *,
    deps: Deps,
    user_id: uuid.UUID,
    question: str,
    conversation_id: uuid.UUID | None,
    document_ids: tuple[uuid.UUID, ...] | None,
    mode: str = "qa",
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    task = asyncio.create_task(
        _produce(queue, deps, user_id, question, conversation_id, document_ids, mode)
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
    deps: Deps,
    user_id: uuid.UUID,
    question: str,
    conversation_id: uuid.UUID | None,
    document_ids: tuple[uuid.UUID, ...] | None,
    mode: str,
) -> None:
    config = deps.settings.pipeline
    partial: list[str] = []
    final: RagAnswer | None = None
    turn: Turn | None = None
    started = time.perf_counter()

    async def status(stage: str) -> None:
        await queue.put(sse("status", {"stage": stage}))

    try:
        tool_calls: list[dict[str, Any]] = []
        if mode == "agent":

            async def emit(ev: StreamEvent) -> None:
                await queue.put(sse(ev.event, ev.data))

            res, _turn = await run_agent_turn(
                deps, user_id, conversation_id, question, document_ids, emit
            )
            # The agent loop is not token-streamed: tools run first, then the answer arrives whole.
            await queue.put(sse("token", {"text": res.answer}))
            for c in res.citations:
                await queue.put(sse("citation", c.to_json()))
            final = RagAnswer(res.answer, res.citations, [], res.input_tokens, res.output_tokens)
            meta = agent_meta(res)
            tool_calls = res.tool_log
        else:
            turn = await prepare_turn(
                deps, user_id, conversation_id, question, document_ids, mode, status
            )
            async for item in stream_answer(turn, deps.llm, deps.settings, config):
                if isinstance(item, RagAnswer):
                    final = item
                else:
                    if item.event == "token":
                        partial.append(str(item.data["text"]))
                    await queue.put(sse(item.event, item.data))
            meta = turn_meta(turn)
        assert final is not None
        conv_id, msg_id = await run_in_threadpool(
            persist_exchange, deps.open_session, user_id, conversation_id, question,
            final.answer, final.citations, final.input_tokens, final.output_tokens, False,
            tool_calls,
        )
        await record_usage(deps, user_id)
        await queue.put(
            sse(
                "done",
                {
                    "conversation_id": str(conv_id),
                    "message_id": str(msg_id),
                    "answer": final.answer,  # authoritative: invalid citation markers removed
                    "input_tokens": final.input_tokens,
                    "output_tokens": final.output_tokens,
                    "latency_ms": _ms(started),
                    "closest": final.closest,
                    **meta,
                },
            )
        )
    except LLMUnavailable as exc:
        log.warning("chat stream: model unavailable (%s)", exc.kind)
        await record_usage(deps, user_id)
        if turn is not None and (passages := retrieved_passages(turn)):
            await queue.put(sse("closest", {"passages": passages}))  # still useful without the model
        await queue.put(
            sse("error", {"code": exc.kind, "message": exc.message, "retry_after": exc.retry_after})
        )
    except asyncio.CancelledError:
        await asyncio.shield(record_usage(deps, user_id))  # the tokens were spent either way
        await asyncio.shield(
            run_in_threadpool(
                persist_exchange, deps.open_session, user_id, conversation_id, question,
                "".join(partial), [], 0, 0, True,
            )
        )
        raise
    except Exception:
        log.exception("chat stream failed")
        await queue.put(sse("error", {"code": "internal", "message": "Something went wrong."}))
    finally:
        await queue.put(None)


__all__ = ["Deps", "answer_turn", "closest_json", "prepare_turn", "stream_chat", "validate_request"]
