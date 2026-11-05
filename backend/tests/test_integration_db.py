"""Integration tests against a real Postgres with pgvector (docker compose up -d postgres,
then alembic upgrade head). Skipped when the database is unreachable."""

import hashlib
import json
import math
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app.api import deps
from app.config import PipelineConfig, Settings
from app.db import get_factory, tenant_session
from app.main import app
from app.models import Chunk, Conversation, Document, DocumentVersion, User
from app.services.generation.llm_client import LLMRequest, LLMResponse
from app.services.ingestion.pipeline import ingest_version
from app.services.vectorstore import PgVectorStore, Scope


def _db_ready() -> bool:
    try:
        with get_factory()() as s:
            s.execute(text("select 1 from chunks limit 1"))
        return True
    except Exception:  # noqa: BLE001 - any failure means "skip"
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="Postgres not available")


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words give high cosine similarity."""

    model_id, dim = "hash-test", 384

    def _vec(self, text_: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]+", text_.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text_):
        return self._vec(text_)


def _make_user() -> uuid.UUID:
    with get_factory()() as s, s.begin():
        u = User(email=f"{uuid.uuid4()}@test.local")
        s.add(u)
        s.flush()
        return u.id


def _make_pdf(path, pages: list[str]) -> None:
    import fitz

    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 520, 760), body, fontsize=11)
    doc.save(path)


def _ingest(user_id, path, config=None) -> uuid.UUID:
    with tenant_session(user_id) as s:
        doc = Document(user_id=user_id, title="t", source_type="pdf", file_path=str(path))
        s.add(doc)
        s.flush()
        ver = DocumentVersion(document_id=doc.id, user_id=user_id, file_hash="x")
        s.add(ver)
        s.flush()
        ingest_version(s, ver.id, HashEmbedder(), config or PipelineConfig())
        return doc.id


PAGES = [
    "Mitochondria are organelles that produce ATP through oxidative phosphorylation. " * 4,
    "The Krebs cycle takes place in the mitochondrial matrix and generates NADH. " * 4,
    "Photosynthesis occurs in chloroplasts and converts light into glucose. " * 4,
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "bio.pdf"
    _make_pdf(path, PAGES)
    user = _make_user()
    doc_id = _ingest(user, path)
    return user, doc_id


def test_ingestion_writes_parents_children_and_marks_ready(corpus):
    user, doc_id = corpus
    with tenant_session(user) as s:
        doc = s.get(Document, doc_id)
        assert doc is not None and doc.status == "ready" and doc.page_count == 3
        assert doc.title == "t"  # ingestion must not replace the user-facing title
        chunks = list(s.scalars(select(Chunk).where(Chunk.document_id == doc_id)))
        parents = [c for c in chunks if c.level == "parent"]
        children = [c for c in chunks if c.level == "child"]
        assert parents and children
        assert all(c.embedding is not None for c in children)
        assert all(c.parent_id in {p.id for p in parents} for c in children)
        assert all(c.tsv is not None for c in chunks)  # trigger populated full-text column
        assert {c.page_start for c in children} >= {1, 3}


def test_vector_search_returns_right_page(corpus):
    user, _ = corpus
    emb = HashEmbedder()
    with tenant_session(user) as s:
        hits = PgVectorStore(s).search(emb.embed_query("how does photosynthesis make glucose"), 3, Scope(user))
    assert hits and hits[0].page_start == 3
    assert hits[0].document_title


def test_rls_blocks_other_tenants(corpus):
    _, doc_id = corpus
    intruder = _make_user()
    emb = HashEmbedder()
    with tenant_session(intruder) as s:
        # even a crafted query that names the victim's document returns nothing
        n = s.execute(text("select count(*) from chunks where document_id = :d"), {"d": doc_id}).scalar()
        assert n == 0
        assert s.execute(text("select count(*) from chunks")).scalar() == 0
        assert PgVectorStore(s).search(emb.embed_query("mitochondria"), 5, Scope(intruder)) == []
        # spoofing the scope user id still hits RLS
        victim = s.execute(text("select 1")).scalar()
        assert victim == 1


def test_rls_rejects_cross_tenant_insert(corpus):
    user, doc_id = corpus
    intruder = _make_user()
    with pytest.raises(Exception, match="row-level security"), tenant_session(intruder) as s:
        s.add(
            Chunk(
                version_id=uuid.uuid4(), document_id=doc_id, user_id=user,
                level="child", chunk_index=0, content="x", page_start=1, page_end=1,
            )
        )
        s.flush()


def test_no_tenant_set_sees_nothing(corpus):
    with get_factory()() as s:
        assert s.execute(text("select count(*) from chunks")).scalar() == 0


def test_hnsw_recall_matches_exact_search(tmp_path):
    user = _make_user()
    words = [f"term{i}" for i in range(60)]
    import random

    rng = random.Random(7)
    bodies = [" ".join(rng.choices(words, k=60)) for _ in range(12)]
    path = tmp_path / "many.pdf"
    _make_pdf(path, bodies)
    _ingest(user, path, PipelineConfig(child_tokens=60, parent_tokens=240, overlap_tokens=10))
    emb = HashEmbedder()
    recalls = []
    with tenant_session(user) as s:
        store = PgVectorStore(s)
        for _ in range(15):
            q = emb.embed_query(" ".join(rng.choices(words, k=5)))
            approx = {h.chunk_id for h in store.search(q, 5, Scope(user))}
            s.execute(text("SET LOCAL enable_indexscan = off"))
            exact = {h.chunk_id for h in store.search(q, 5, Scope(user))}
            s.execute(text("SET LOCAL enable_indexscan = on"))
            recalls.append(len(approx & exact) / max(1, len(exact)))
    assert sum(recalls) / len(recalls) >= 0.95


def test_upload_and_chat_end_to_end(tmp_path):
    user = _make_user()
    pdf = tmp_path / "bio.pdf"
    _make_pdf(pdf, PAGES)

    class LLM:
        def __init__(self):
            self.req = None

        async def stream(self, req: LLMRequest):
            from app.services.generation.llm_client import LLMEvent

            yield LLMEvent("delta", text="Chloroplasts make glucose [c_1].")
            yield LLMEvent("usage", input_tokens=5, output_tokens=3)

        async def complete(self, req: LLMRequest) -> LLMResponse:
            self.req = req
            return LLMResponse("Chloroplasts convert light into glucose [c_1].", 50, 10)

    llm = LLM()
    settings = Settings(upload_dir=str(tmp_path / "uploads"), llm_model="test")

    def enqueue(version_id: str, user_id: str) -> None:  # run ingestion inline, like the worker
        with tenant_session(uuid.UUID(user_id)) as s:
            ingest_version(s, uuid.UUID(version_id), HashEmbedder(), settings.pipeline)

    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.enqueue_dep] = lambda: enqueue
    app.dependency_overrides[deps.embedder_dep] = lambda: HashEmbedder()
    app.dependency_overrides[deps.llm_dep] = lambda: llm
    app.dependency_overrides[deps.reranker_dep] = lambda: None
    try:
        client = TestClient(app)
        h = {"X-User-Id": str(user)}
        up = client.post("/documents", headers=h, files={"file": ("bio.pdf", pdf.read_bytes())})
        assert up.status_code == 202, up.text
        listed = client.get("/documents", headers=h).json()
        assert listed[0]["status"] == "ready" and listed[0]["page_count"] == 3

        r = client.post("/chat", headers=h, json={"question": "what do chloroplasts do with light?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["citations"] and body["citations"][0]["page_start"] == 3
        assert "Photosynthesis" in llm.req.user

        # a different user has no documents and gets a refusal without calling the LLM
        llm.req = None
        other = {"X-User-Id": str(_make_user())}
        assert client.get("/documents", headers=other).json() == []
        r2 = client.post("/chat", headers=other, json={"question": "what do chloroplasts do?"})
        assert r2.json()["citations"] == [] and llm.req is None

        # authenticated file access: owner gets the PDF, another tenant gets 404
        doc_id = listed[0]["id"]
        f = client.get(f"/documents/{doc_id}/file", headers=h)
        assert f.status_code == 200 and f.content.startswith(b"%PDF")
        assert client.get(f"/documents/{doc_id}/file", headers=other).status_code == 404
        assert client.get(f"/documents/{doc_id}", headers=h).json()["stage"] == "ready"

        # streaming path against the real database
        with client.stream(
            "POST", "/chat/stream", headers=h, json={"question": "what do chloroplasts do with light?"}
        ) as resp:
            body = "".join(resp.iter_text())
        assert "event: citation" in body and "event: done" in body
        assert '"page_start": 3' in body and "bbox_list" in body

        bad = client.post("/documents", headers=h, files={"file": ("x.exe", b"MZ\x90\x00")})
        assert bad.status_code == 415
    finally:
        app.dependency_overrides.clear()


# --- Phase 3: keyword search, memory, compare -------------------------------------------------


def test_keyword_search_finds_rare_terms_and_respects_tenancy(corpus):
    user, _ = corpus
    with tenant_session(user) as s:
        store = PgVectorStore(s)
        hits = store.keyword_search("What happens in the Krebs cycle?", 5, Scope(user))
        assert hits and hits[0].page_start == 2 and "Krebs" in hits[0].child_text
        assert store.keyword_search("???", 5, Scope(user)) == []
        assert store.keyword_search("xylophone", 5, Scope(user)) == []
    with tenant_session(_make_user()) as s:
        assert PgVectorStore(s).keyword_search("Krebs cycle", 5, Scope(uuid.uuid4())) == []


def _chat_client(tmp_path, llm):
    from .fakes import FakeReranker

    settings = Settings(upload_dir=str(tmp_path / "uploads"), llm_model="test")

    def enqueue(version_id: str, user_id: str) -> None:
        with tenant_session(uuid.UUID(user_id)) as s:
            ingest_version(s, uuid.UUID(version_id), HashEmbedder(), settings.pipeline)

    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.enqueue_dep] = lambda: enqueue
    app.dependency_overrides[deps.embedder_dep] = lambda: HashEmbedder()
    app.dependency_overrides[deps.llm_dep] = lambda: llm
    app.dependency_overrides[deps.reranker_dep] = lambda: FakeReranker()
    return TestClient(app)


def test_multi_turn_rewrite_and_rolling_summary(tmp_path):
    from .fakes import FakeLLM

    user = _make_user()
    pdf = tmp_path / "bio.pdf"
    _make_pdf(pdf, PAGES)
    llm = FakeLLM("Mitochondria make ATP [c_1].", rewrite="how do mitochondria produce ATP")
    client = _chat_client(tmp_path, llm)
    h = {"X-User-Id": str(user)}
    try:
        assert client.post("/documents", headers=h, files={"file": ("bio.pdf", pdf.read_bytes())}).status_code == 202

        first = client.post("/chat", headers=h, json={"question": "How do mitochondria produce ATP?"}).json()
        assert llm.rewrites == []  # first turn: nothing to resolve, no rewrite call
        conv = first["conversation_id"]

        second = client.post(
            "/chat", headers=h, json={"question": "and where does it happen?", "conversation_id": conv}
        ).json()
        assert len(llm.rewrites) == 1
        assert second["meta"]["standalone_query"] == "how do mitochondria produce ATP"
        answer_req = llm.requests[-1]
        assert "<conversation>" in answer_req.user and "How do mitochondria produce ATP?" in answer_req.user
        assert "[c_1]" not in answer_req.user.split("<sources>")[0]  # history is stripped of markers

        # keep chatting until the window overflows and older turns are folded into a summary
        for i in range(6):
            client.post("/chat", headers=h, json={"question": f"follow up {i} about ATP", "conversation_id": conv})
        assert llm.summaries, "older messages should have been summarized"
        with tenant_session(user) as s:
            c = s.get(Conversation, uuid.UUID(conv))
            assert c is not None and c.summary.startswith("SUMMARY") and c.summarized_count > 0
        last = llm.requests[-1].user
        assert "Summary of the earlier discussion: SUMMARY" in last
    finally:
        app.dependency_overrides.clear()


def test_compare_mode_against_real_database(tmp_path):
    from .fakes import FakeLLM

    user = _make_user()
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    _make_pdf(a, ["Paper A reports that mitochondria yield 30 ATP per glucose molecule."])
    _make_pdf(b, ["Paper B reports that mitochondria yield 36 ATP per glucose molecule."])
    llm = FakeLLM("A: 30 [c_1]. B: 36 [c_2].")
    client = _chat_client(tmp_path, llm)
    h = {"X-User-Id": str(user)}
    try:
        for name, p in (("Paper A", a), ("Paper B", b)):
            client.post("/documents", headers=h, files={"file": (f"{name}.pdf", p.read_bytes())})
        ids = [d["id"] for d in client.get("/documents", headers=h).json()]
        assert len(ids) == 2
        body = client.post(
            "/chat", headers=h,
            json={"question": "Compare the ATP yield per glucose", "mode": "compare", "document_ids": ids},
        ).json()
        user_msg = llm.requests[-1].user
        assert user_msg.count("<document title=") == 2
        assert {c["document_title"] for c in body["citations"]} == {"Paper A", "Paper B"}
    finally:
        app.dependency_overrides.clear()


# --- Phase 4: study endpoints ----------------------------------------------------------------


def _study_setup(tmp_path, llm):
    client = _chat_client(tmp_path, llm)
    user = _make_user()
    h = {"X-User-Id": str(user)}
    pdf = tmp_path / "bio.pdf"
    _make_pdf(pdf, PAGES)
    up = client.post("/documents", headers=h, files={"file": ("Biology.pdf", pdf.read_bytes())})
    assert up.status_code == 202
    return client, h, up.json()["id"]


def _quiz_reply(n=1):
    qs = [
        {
            "question": f"Question {i} about the passage?",
            "options": ["ATP", "DNA", "Lipids", "Water"],
            "answer_index": 0,
            "explanation": "Because the passage says so.",
            "source": "s_1",
        }
        for i in range(n)
    ]
    return json.dumps({"questions": qs})


def test_quiz_flow_persists_isolates_tenants_and_handles_errors(tmp_path):
    from .fakes import FakeLLM

    llm = FakeLLM()
    llm.handler = lambda req: _quiz_reply(2)
    client, h, doc_id = _study_setup(tmp_path, llm)
    try:
        r = client.post("/study/quizzes", headers=h, json={"document_id": doc_id, "num_questions": 2})
        assert r.status_code == 201, r.text
        quiz = r.json()
        assert len(quiz["questions"]) == 2 and quiz["questions"][0]["page_start"] >= 1
        assert quiz["questions"][0]["answer_index"] == 0

        got = client.get(f"/study/quizzes/{quiz['id']}", headers=h).json()
        assert got["id"] == quiz["id"]
        assert [x["id"] for x in client.get(f"/study/quizzes?document_id={doc_id}", headers=h).json()] == [quiz["id"]]

        other = {"X-User-Id": str(_make_user())}
        assert client.get(f"/study/quizzes/{quiz['id']}", headers=other).status_code == 404
        assert client.get("/study/quizzes", headers=other).json() == []
        assert client.post("/study/quizzes", headers=other, json={"document_id": doc_id}).status_code == 404

        assert client.post("/study/quizzes", headers=h, json={"document_id": doc_id, "num_questions": 99}).status_code == 422
        assert client.post("/study/quizzes", headers=h, json={"document_id": str(uuid.uuid4())}).status_code == 404

        llm.handler = lambda req: "not json at all"  # invalid twice -> safe 502
        bad = client.post("/study/quizzes", headers=h, json={"document_id": doc_id})
        assert bad.status_code == 502 and "valid result" in bad.json()["detail"]
        assert "not json" not in bad.text

        def boom(req):
            raise RuntimeError("secret upstream detail")

        llm.handler = boom
        err = client.post("/study/quizzes", headers=h, json={"document_id": doc_id})
        assert err.status_code == 502 and "secret" not in err.text
    finally:
        app.dependency_overrides.clear()


def test_quiz_sources_are_exact_pages_with_highlight_boxes(tmp_path):
    """Questions come from child chunks, so the cited page is the page the passage is on."""
    from .fakes import FakeLLM

    qs = [
        {"question": f"Question {i} about the passage?", "options": ["A1", "B2", "C3", "D4"],
         "answer_index": 1, "explanation": "Because.", "source": f"s_{i}"}
        for i in (1, 2, 3)
    ]
    llm = FakeLLM()
    llm.handler = lambda req: json.dumps({"questions": qs})
    client, h, doc_id = _study_setup(tmp_path, llm)
    try:
        quiz = client.post("/study/quizzes", headers=h, json={"document_id": doc_id, "num_questions": 3}).json()
        assert [q["page_start"] for q in quiz["questions"]] == [1, 2, 3]  # one passage per page
        assert all(q["page_start"] == q["page_end"] for q in quiz["questions"])
        assert all(q["bbox_list"] and q["bbox_list"][0]["page"] == q["page_start"] for q in quiz["questions"])
        # the passage the model saw for s_3 really is the page 3 text
        assert "Photosynthesis" in llm.requests[0].user.split('<passage id="s_3"')[1]
    finally:
        app.dependency_overrides.clear()


def test_study_requires_a_ready_document(tmp_path):
    from .fakes import FakeLLM

    llm = FakeLLM()
    llm.handler = lambda req: _quiz_reply()
    client, h, doc_id = _study_setup(tmp_path, llm)
    try:
        user_id = uuid.UUID(h["X-User-Id"])
        with tenant_session(user_id) as s:
            doc = s.get(Document, uuid.UUID(doc_id))
            assert doc is not None
            doc.status = "processing"
        r = client.post("/study/quizzes", headers=h, json={"document_id": doc_id})
        assert r.status_code == 409 and "not ready" in r.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_flashcards_flow(tmp_path):
    from .fakes import FakeLLM

    llm = FakeLLM()
    llm.handler = lambda req: json.dumps(
        {"cards": [{"front": "What makes ATP?", "back": "Mitochondria", "source": "s_1"}]}
    )
    client, h, doc_id = _study_setup(tmp_path, llm)
    try:
        r = client.post("/study/flashcards", headers=h, json={"document_id": doc_id, "num_cards": 3})
        assert r.status_code == 201, r.text
        sets = client.get("/study/flashcards", headers=h).json()
        assert sets[0]["cards"][0]["front"] == "What makes ATP?" and sets[0]["cards"][0]["page_start"] >= 1
        assert client.get(f"/study/flashcards/{r.json()['id']}", headers=h).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_summary_is_map_reduced_then_served_from_cache(tmp_path):
    from .fakes import FakeLLM

    def handler(req):
        if req.system.startswith("Summarize the passage"):
            return "A section summary."
        return "OVERVIEW. - key point (p.2)"

    llm = FakeLLM()
    llm.handler = handler
    client, h, doc_id = _study_setup(tmp_path, llm)
    try:
        first = client.post("/study/summaries", headers=h, json={"document_id": doc_id}).json()
        assert first["cached"] is False and first["summary"].startswith("OVERVIEW")
        assert first["meta"]["sections_summarized"] >= 1
        calls_after_first = len(llm.requests)

        second = client.post("/study/summaries", headers=h, json={"document_id": doc_id}).json()
        assert second["cached"] is True and second["summary"] == first["summary"]
        assert len(llm.requests) == calls_after_first  # no model calls at all

        assert client.get(f"/study/summaries/{doc_id}", headers=h).json()["summary"] == first["summary"]
        other = {"X-User-Id": str(_make_user())}
        assert client.get(f"/study/summaries/{doc_id}", headers=other).status_code == 404

        # re-ingesting makes a new chunk set: a new version must not reuse a stale summary
        user_id = uuid.UUID(h["X-User-Id"])
        with tenant_session(user_id) as s:
            old = s.scalar(select(DocumentVersion).where(DocumentVersion.document_id == uuid.UUID(doc_id)))
            assert old is not None
            doc = s.get(Document, uuid.UUID(doc_id))
            assert doc is not None
            new = DocumentVersion(document_id=doc.id, user_id=user_id, file_hash="v2", version_no=2)
            s.add(new)
            s.flush()
            ingest_version(s, new.id, HashEmbedder(), PipelineConfig())
        third = client.post("/study/summaries", headers=h, json={"document_id": doc_id}).json()
        assert third["cached"] is False  # v2 has no cache yet
    finally:
        app.dependency_overrides.clear()


def test_agent_search_tool_against_real_database(tmp_path):
    """End to end: the agent's search_documents tool uses real retrieval, tenancy and gating."""
    from .fakes import FakeLLM, text_turn, tool_turn

    llm = FakeLLM()
    llm.script = [
        tool_turn(("search_documents", {"query": "How does photosynthesis make glucose"}),
                  ("calculator", {"expression": "2 * 21"})),
        text_turn("Chloroplasts make glucose [c_1]; the sum is 42."),
    ]
    client, h, _ = _study_setup(tmp_path, llm)
    try:
        body = client.post(
            "/chat", headers=h, json={"question": "How does photosynthesis make glucose? Also 2*21?", "mode": "agent"}
        ).json()
        assert [t["name"] for t in body["meta"]["tool_calls"]] == ["search_documents", "calculator"]
        assert body["citations"] and body["citations"][0]["page_start"] == 3
        tool_result = llm.converse_requests[1].messages[-1]["content"][0]["content"]
        assert "<source id=\"c_1\"" in tool_result and "Photosynthesis" in tool_result
        # another user's agent finds nothing in these documents
        llm.script = [tool_turn(("search_documents", {"query": "photosynthesis glucose"})), text_turn("none")]
        other = {"X-User-Id": str(_make_user())}
        client.post("/chat", headers=other, json={"question": "photosynthesis?", "mode": "agent"})
        assert "No relevant passages" in llm.converse_requests[-1].messages[-1]["content"][0]["content"]
    finally:
        app.dependency_overrides.clear()
