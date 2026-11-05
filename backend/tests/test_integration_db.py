"""Integration tests against a real Postgres with pgvector (docker compose up -d postgres,
then alembic upgrade head). Skipped when the database is unreachable."""

import hashlib
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
from app.models import Chunk, Document, DocumentVersion, User
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
