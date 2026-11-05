import asyncio
import json
import uuid

from app.services import chat_service
from app.services.chat_service import stream_chat
from app.services.generation.llm_client import LLMEvent

from .fakes import FakeStore, make_deps, make_hit

USER = uuid.uuid4()


def _hit():
    return make_hit()


class StreamLLM:
    def __init__(self, deltas, hang_after: int | None = None):
        self.deltas, self.hang_after = deltas, hang_after

    async def complete(self, req):  # pragma: no cover
        raise AssertionError

    async def stream(self, req):
        for i, d in enumerate(self.deltas):
            if self.hang_after is not None and i == self.hang_after:
                await asyncio.sleep(30)
            yield LLMEvent("delta", text=d)
        yield LLMEvent("usage", input_tokens=11, output_tokens=5)


def _run(llm, hits, session=None, **kw):
    deps, session = make_deps(FakeStore(hits), llm, session=session)
    gen = stream_chat(
        deps=deps, user_id=USER, question="q", conversation_id=None, document_ids=None, **kw
    )
    return gen, session


def _parse(chunks: list[str]) -> list[tuple[str, dict]]:
    out = []
    for c in chunks:
        if c.startswith("event:"):
            name, data = c.split("\n")[:2]
            out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


async def test_stream_emits_ordered_events_and_persists():
    gen, session = _run(StreamLLM(["ATP is made ", "in mitochondria [c_1] ", "and [c_9]."]), [_hit()])
    events = _parse([c async for c in gen])
    names = [n for n, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    assert names.count("token") == 3 and names.count("citation") == 1
    assert names.index("citation") > max(i for i, n in enumerate(names) if n == "token")
    cite = next(d for n, d in events if n == "citation")
    assert cite["page_start"] == 7 and cite["bbox_list"][0]["bbox"] == [1, 2, 3, 4]
    done = events[-1][1]
    assert "[c_9]" not in done["answer"] and "[c_1]" in done["answer"]
    assert done["input_tokens"] == 11
    roles = [m.role for m in session.added if hasattr(m, "role")]
    assert roles == ["user", "assistant"]
    assert not next(m for m in session.added if getattr(m, "role", "") == "assistant").incomplete


async def test_stream_refuses_without_hits():
    gen, _ = _run(StreamLLM([]), [])
    events = _parse([c async for c in gen])
    assert [n for n, _ in events] == ["status", "status", "token", "done"]


async def test_disconnect_saves_partial_answer_as_incomplete():
    gen, session = _run(StreamLLM(["partial ", "never sent"], hang_after=1), [_hit()])
    seen = []
    async for chunk in gen:
        seen.append(chunk)
        if "partial" in chunk:
            break
    await gen.aclose()  # what Starlette does when the client disconnects
    await asyncio.sleep(0.3)
    reply = next(m for m in session.added if getattr(m, "role", "") == "assistant")
    assert reply.incomplete is True and reply.content == "partial "


async def test_stream_error_event_is_safe():
    class Boom(StreamLLM):
        async def stream(self, req):
            raise RuntimeError("secret internal detail")
            yield  # pragma: no cover

    gen, _ = _run(Boom([]), [_hit()])
    events = _parse([c async for c in gen])
    err = next(d for n, d in events if n == "error")
    assert "secret" not in json.dumps(err)


async def test_heartbeat_comment_while_idle(monkeypatch):
    monkeypatch.setattr(chat_service, "HEARTBEAT_SECONDS", 0.05)
    gen, _ = _run(StreamLLM(["x"], hang_after=0), [_hit()])
    chunks = []
    async for c in gen:
        chunks.append(c)
        if c.startswith(": ping"):
            break
    await gen.aclose()
    assert any(c.startswith(": ping") for c in chunks)
