import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings
from app.main import app
from app.services.generation.llm_client import LLMRequest, LLMResponse
from app.services.generation.rag_chain import pack_context, validate_citations
from app.services.vectorstore import Hit

DOC_ID = uuid.uuid4()
CHUNK_ID = uuid.uuid4()


def _hit(text: str = "Mitochondria produce ATP.") -> Hit:
    return Hit(CHUNK_ID, DOC_ID, "Biology", 7, 7, 0.9, text, text)


class FakeStore:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits, self.scopes = hits, []

    def search(self, vector, k, scope):
        self.scopes.append(scope)
        return self.hits

    def delete_version(self, version_id):  # pragma: no cover
        pass


class FakeEmbedder:
    model_id, dim = "fake", 3

    def embed_documents(self, texts):
        return [[0.0, 0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 1.0]


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text, self.requests = text, []

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return LLMResponse(self.text, 100, 20)


class FakeSession:
    def __init__(self) -> None:
        self.added = []

    def get(self, model, pk):
        return None

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()


@pytest.fixture
def setup():
    store, llm, session = FakeStore([_hit()]), FakeLLM("ATP comes from mitochondria [c_1]."), FakeSession()
    app.dependency_overrides[deps.vector_store_dep] = lambda: store
    app.dependency_overrides[deps.embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[deps.llm_dep] = lambda: llm
    app.dependency_overrides[deps.db_session] = lambda: session
    app.dependency_overrides[deps.settings_dep] = lambda: Settings(llm_model="test-model")
    yield SimpleNamespace(store=store, llm=llm, session=session, client=TestClient(app))
    app.dependency_overrides.clear()


USER = str(uuid.uuid4())


def test_chat_returns_grounded_answer_with_page_citation(setup):
    r = setup.client.post("/chat", json={"question": "Where does ATP come from?"},
                          headers={"X-User-Id": USER})
    assert r.status_code == 200
    body = r.json()
    assert "[c_1]" in body["answer"]
    assert body["citations"][0]["page_start"] == 7
    assert body["citations"][0]["document_id"] == str(DOC_ID)
    # tenant is always part of the search scope
    assert str(setup.store.scopes[0].user_id) == USER
    # model sees context wrapped as untrusted data
    assert "<source id=\"c_1\"" in setup.llm.requests[0].user
    assert "untrusted data" in setup.llm.requests[0].system
    assert [m.role for m in setup.session.added if hasattr(m, "role")] == ["user", "assistant"]


def test_chat_refuses_without_hits_and_skips_llm(setup):
    setup.store.hits = []
    r = setup.client.post("/chat", json={"question": "anything"}, headers={"X-User-Id": USER})
    assert r.status_code == 200
    assert r.json()["citations"] == []
    assert setup.llm.requests == []


def test_chat_requires_user_header(setup):
    assert setup.client.post("/chat", json={"question": "hi"}).status_code == 422


def test_invalid_citation_markers_are_dropped():
    _, by_marker = pack_context([_hit()])
    text, cites = validate_citations("Real [c_1] and fake [c_9].", by_marker)
    assert "[c_9]" not in text and "[c_1]" in text
    assert len(cites) == 1


def test_pack_context_dedupes_shared_parents():
    rendered, by_marker = pack_context([_hit("a"), _hit("b")])
    assert len(rendered) == 1 and list(by_marker) == ["c_1"]
