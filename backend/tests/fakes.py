"""Shared fakes: no network, no database, no models."""

import dataclasses
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

from app.config import Settings
from app.services.chat_service import Deps
from app.services.generation.llm_client import LLMEvent, LLMRequest, LLMResponse
from app.services.vectorstore import Hit

DOC_A, DOC_B = uuid.uuid4(), uuid.uuid4()
CHUNK_ID = uuid.uuid4()


def make_hit(text="Mitochondria produce ATP.", doc=DOC_A, title="Biology", page=7, parent=None):
    pid = parent or uuid.uuid4()
    return Hit(
        pid, doc, title, page, page, 0.9, text, text,
        bbox_list=[{"page": page, "bbox": [1, 2, 3, 4]}], child_id=uuid.uuid4(),
    )


class FakeStore:
    def __init__(self, hits=None, sparse=None):
        self.hits, self.sparse, self.scopes = list(hits or []), list(sparse or []), []
        self.queries = []

    def _filter(self, hits, scope):
        self.scopes.append(scope)
        if scope.document_ids:
            return [h for h in hits if h.document_id in scope.document_ids]
        return hits

    def search(self, vector, k, scope):
        return self._filter(self.hits, scope)

    def keyword_search(self, query, k, scope):
        self.queries.append(query)
        return self._filter(self.sparse, scope)

    def delete_version(self, version_id):  # pragma: no cover
        pass


class FakeEmbedder:
    model_id, dim = "fake", 3

    def embed_documents(self, texts):
        return [[0.0, 0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 1.0]


class FakeReranker:
    """Score 0.9 when the passage contains 'ATP' or 'glucose', otherwise 0.02."""

    def rerank(self, query, hits, top_n):
        scored = [
            dataclasses.replace(h, score=0.9 if ("ATP" in h.child_text or "glucose" in h.child_text) else 0.02)
            for h in hits
        ]
        return sorted(scored, key=lambda h: -h.score)[:top_n]


class FakeLLM:
    def __init__(self, text="ATP comes from mitochondria [c_1].", rewrite="standalone question"):
        self.text, self.rewrite, self.requests, self.rewrites = text, rewrite, [], []
        self.summaries = []

    async def complete(self, req: LLMRequest) -> LLMResponse:
        if req.system.startswith("You maintain"):
            self.summaries.append(req)
            return LLMResponse("SUMMARY: user is studying cell energy.", 5, 5)
        if req.system.startswith("Rewrite"):
            self.rewrites.append(req)
            return LLMResponse(self.rewrite, 5, 5)
        self.requests.append(req)
        return LLMResponse(self.text, 100, 20)

    async def stream(self, req: LLMRequest):
        self.requests.append(req)
        for word in self.text.split(" "):
            yield LLMEvent("delta", text=word + " ")
        yield LLMEvent("usage", input_tokens=100, output_tokens=20)


class FakeSession:
    def __init__(self, docs=()):
        self.added, self._docs = [], list(docs)

    def get(self, model, pk):
        return None

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    def scalars(self, stmt):
        return [SimpleNamespace(id=i, title=t) for i, t in self._docs]


def make_deps(store=None, llm=None, reranker="default", session=None, settings=None):
    session = session or FakeSession([(DOC_A, "Paper A"), (DOC_B, "Paper B")])

    @contextmanager
    def opener(_uid):
        yield session

    deps = Deps(
        open_session=opener,
        make_store=lambda _s: store or FakeStore(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker() if reranker == "default" else reranker,
        llm=llm or FakeLLM(),
        settings=settings or Settings(llm_model="test-model"),
    )
    return deps, session
