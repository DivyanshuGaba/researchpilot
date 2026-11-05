import asyncio
import uuid

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app.api import deps
from app.config import Settings
from app.db import get_factory, tenant_session
from app.main import app
from app.models import Document, OutboxEvent
from app.services.generation.llm_client import (
    AnthropicClient,
    LLMRequest,
    LLMUnavailable,
    classify_error,
)
from app.services.outbox import MAX_ATTEMPTS, dispatch_pending

from .fakes import FakeLLM, FakeStore, auth, make_deps, make_hit
from .test_integration_db import PAGES, _db_ready, _make_pdf, _make_user

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _resp(status):
    return httpx.Response(status, request=REQ)


# --- error classification ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (anthropic.RateLimitError("rate sk-ant-SECRET", response=_resp(429), body=None), "rate_limited"),
        (anthropic.APITimeoutError(request=REQ), "timeout"),
        (anthropic.APIConnectionError(request=REQ), "unavailable"),
        (anthropic.InternalServerError("boom", response=_resp(500), body=None), "unavailable"),
        (anthropic.APIStatusError("overloaded", response=_resp(529), body=None), "unavailable"),
        (anthropic.AuthenticationError("bad key sk-ant-SECRET", response=_resp(401), body=None), "misconfigured"),
        (anthropic.PermissionDeniedError("nope", response=_resp(403), body=None), "misconfigured"),
        (TimeoutError(), "timeout"),
        (anthropic.BadRequestError("Your credit balance is too low to access the Anthropic API.", response=_resp(400), body=None), "billing"),
    ],
)
def test_provider_errors_map_to_safe_categories(exc, kind):
    mapped = classify_error(exc)
    assert mapped is not None and mapped.kind == kind
    assert "sk-ant" not in mapped.message and "SECRET" not in mapped.message and "429" not in mapped.message


def test_programming_errors_and_bad_requests_are_not_masked_as_outages():
    assert classify_error(ValueError("bug")) is None
    assert classify_error(anthropic.BadRequestError("invalid", response=_resp(400), body=None)) is None  # a real bug stays visible
    assert classify_error(anthropic.NotFoundError("no model", response=_resp(404), body=None)) is None


async def test_anthropic_client_raises_llm_unavailable_for_outages():
    client = AnthropicClient("test-key", max_retries=0)

    async def boom(**kw):
        raise anthropic.RateLimitError("limit", response=_resp(429), body=None)

    client._client.messages.create = boom  # type: ignore[method-assign]
    with pytest.raises(LLMUnavailable) as err:
        await client.complete(LLMRequest("s", "u", "m"))
    assert err.value.kind == "rate_limited" and err.value.retry_after == 30

    async def bug(**kw):
        raise KeyError("a real bug must stay visible")

    client._client.messages.create = bug  # type: ignore[method-assign]
    with pytest.raises(KeyError):
        await client.complete(LLMRequest("s", "u", "m"))


async def test_concurrency_semaphore_caps_inflight_model_calls():
    client = AnthropicClient("test-key", max_concurrency=2)
    inflight = peak = 0

    async def slow(**kw):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        raise anthropic.APIConnectionError(request=REQ)

    client._client.messages.create = slow  # type: ignore[method-assign]
    results = await asyncio.gather(
        *(client.complete(LLMRequest("s", "u", "m")) for _ in range(8)), return_exceptions=True
    )
    assert peak == 2 and all(isinstance(r, LLMUnavailable) for r in results)


# --- chat and study degrade gracefully -------------------------------------------------------

H = auth(uuid.uuid4())


def _client(llm):
    d, _ = make_deps(FakeStore([make_hit("Mitochondria make ATP.", page=4)]), llm)
    app.dependency_overrides[deps.chat_deps] = lambda: d
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear():
    yield
    app.dependency_overrides.clear()


def test_chat_returns_503_with_the_retrieved_passages_when_the_model_is_down():
    llm = FakeLLM()
    llm.error = LLMUnavailable("unavailable")
    r = _client(llm).post("/chat", json={"question": "ATP?"}, headers=H)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "unavailable" and "temporarily unavailable" in detail["message"]
    assert detail["closest"][0]["page_start"] == 4 and "ATP" in detail["closest"][0]["quote"]


def test_chat_returns_429_with_retry_after_when_the_provider_rate_limits():
    llm = FakeLLM()
    llm.error = LLMUnavailable("rate_limited", retry_after=30)
    r = _client(llm).post("/chat", json={"question": "ATP?"}, headers=H)
    assert r.status_code == 429 and r.headers["retry-after"] == "30"
    llm.error = LLMUnavailable("timeout")
    assert _client(llm).post("/chat", json={"question": "ATP?"}, headers=H).status_code == 504


def test_stream_sends_passages_then_a_clear_error_and_no_done_event():
    llm = FakeLLM()
    llm.error = LLMUnavailable("unavailable")
    with _client(llm).stream("POST", "/chat/stream", json={"question": "ATP?"}, headers=H) as r:
        body = "".join(r.iter_text())
    assert body.index("event: closest") < body.index("event: error")
    assert "event: done" not in body and "temporarily unavailable" in body
    assert "Traceback" not in body and "LLMUnavailable" not in body


def test_agent_mode_reports_outages_too():
    llm = FakeLLM()
    llm.error = LLMUnavailable("timeout")
    r = _client(llm).post("/chat", json={"question": "q", "mode": "agent"}, headers=H)
    assert r.status_code == 504 and "took too long" in r.json()["detail"]["message"]


# --- outbox ------------------------------------------------------------------------------------

db = pytest.mark.skipif(not _db_ready(), reason="Postgres not available")


@db
def test_a_failed_enqueue_never_loses_the_job(tmp_path):
    user = _make_user()
    pdf = tmp_path / "bio.pdf"
    _make_pdf(pdf, PAGES)
    calls: list[tuple[str, str]] = []
    mode = {"fail": True}

    def enqueue(version_id: str, user_id: str) -> None:
        if mode["fail"]:
            raise ConnectionError("broker is down")
        calls.append((version_id, user_id))

    settings = Settings(upload_dir=str(tmp_path / "up"))
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.enqueue_dep] = lambda: enqueue
    client = TestClient(app)
    h = auth(user)
    up = client.post("/documents", headers=h, files={"file": ("bio.pdf", pdf.read_bytes())})
    assert up.status_code == 202  # the upload succeeds even though the broker is down

    with get_factory()() as s:
        ev = s.scalars(select(OutboxEvent).where(OutboxEvent.payload["user_id"].astext == str(user))).one()
        assert ev.dispatched_at is None and ev.attempts == 1  # tried once right after commit
        version_id = ev.payload["version_id"]
    with tenant_session(user) as s:
        doc = s.scalar(select(Document))
        assert doc is not None and doc.status == "queued"  # saved, waiting for its job

    mode["fail"] = False  # broker recovers; the background dispatcher picks the row up
    assert dispatch_pending(get_factory(), enqueue) >= 1
    assert (version_id, str(user)) in calls
    before = len(calls)
    dispatch_pending(get_factory(), enqueue)
    assert len(calls) == before  # dispatched exactly once

    with get_factory()() as s:
        ev = s.scalars(select(OutboxEvent).where(OutboxEvent.payload["user_id"].astext == str(user))).one()
        assert ev.dispatched_at is not None and ev.attempts == 2


@db
def test_dispatch_gives_up_after_max_attempts():
    with get_factory()() as s, s.begin():
        ev = OutboxEvent(kind="ingest", payload={"version_id": str(uuid.uuid4()), "user_id": str(uuid.uuid4())})
        s.add(ev)
        s.flush()
        event_id = ev.id

    def always_fails(v, u):
        raise ConnectionError("down")

    for _ in range(MAX_ATTEMPTS + 3):
        dispatch_pending(get_factory(), always_fails)
    with get_factory()() as s:
        row = s.get(OutboxEvent, event_id)
        assert row is not None and row.attempts == MAX_ATTEMPTS and row.dispatched_at is None
        s.execute(text("DELETE FROM outbox WHERE id = :i"), {"i": event_id})
        s.commit()


@db
def test_study_endpoints_report_a_model_outage_safely(tmp_path):
    from .test_integration_db import _study_setup

    llm = FakeLLM()
    client, h, doc_id = _study_setup(tmp_path, llm)
    body = {"document_id": doc_id}
    llm.error = LLMUnavailable("rate_limited", retry_after=30)
    r = client.post("/study/quizzes", headers=h, json=body)
    assert r.status_code == 429 and r.headers["retry-after"] == "30" and "busy" in r.json()["detail"]
    llm.error = LLMUnavailable("unavailable")
    for path in ("/study/quizzes", "/study/flashcards", "/study/summaries"):
        r = client.post(path, headers=h, json=body)
        assert r.status_code == 503 and "temporarily unavailable" in r.json()["detail"], path
        assert "sk-ant" not in r.text
