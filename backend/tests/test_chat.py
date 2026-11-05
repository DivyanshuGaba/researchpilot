import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import deps as api_deps
from app.main import app
from app.services.generation.rag_chain import Group, pack_context, validate_citations

from .fakes import DOC_A, DOC_B, FakeLLM, FakeReranker, FakeStore, make_deps, make_hit

USER = str(uuid.uuid4())
H = {"X-User-Id": USER}


@pytest.fixture
def setup():
    state = {}

    def build(store, llm=None, reranker="default"):
        deps, session = make_deps(store, llm, reranker)
        app.dependency_overrides[api_deps.chat_deps] = lambda: deps
        state.update(deps=deps, session=session, llm=deps.llm, store=store)
        return TestClient(app)

    yield build, state
    app.dependency_overrides.clear()


def test_chat_returns_grounded_answer_with_page_citation(setup):
    build, st = setup
    client = build(FakeStore([make_hit()]))
    r = client.post("/chat", json={"question": "Where does ATP come from?"}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert "[c_1]" in body["answer"]
    assert body["citations"][0]["page_start"] == 7 and body["citations"][0]["bbox_list"]
    assert str(st["store"].scopes[0].user_id) == USER  # tenant always in scope
    req = st["llm"].requests[0]
    assert '<source id="c_1"' in req.user and "untrusted data" in req.system
    assert [m.role for m in st["session"].added if hasattr(m, "role")] == ["user", "assistant"]


def test_no_hits_refuses_without_calling_llm(setup):
    build, st = setup
    client = build(FakeStore([]))
    r = client.post("/chat", json={"question": "anything"}, headers=H)
    assert r.json()["citations"] == [] and st["llm"].requests == []


def test_confidence_gate_refuses_and_shows_closest_passages(setup):
    build, st = setup
    client = build(FakeStore([make_hit("Totally unrelated gardening text about roses.", page=3)]))
    r = client.post("/chat", json={"question": "What is the capital of Mongolia?"}, headers=H)
    body = r.json()
    assert "don't seem to cover" in body["answer"]
    assert body["citations"] == []  # no fake citation on a refusal
    assert body["closest"][0]["page_start"] == 3 and "roses" in body["closest"][0]["quote"]
    assert st["llm"].requests == []  # gated before generation
    assert body["meta"]["top_score"] == 0.02


def test_relevant_hit_passes_the_gate(setup):
    build, st = setup
    client = build(FakeStore([make_hit("noise"), make_hit("Mitochondria make ATP.")]))
    body = client.post("/chat", json={"question": "ATP?"}, headers=H).json()
    assert body["closest"] == [] and st["llm"].requests
    assert body["meta"]["top_score"] == 0.9


def test_missing_reranker_degrades_but_still_answers(setup):
    build, _ = setup
    client = build(FakeStore([make_hit()]), reranker=None)
    body = client.post("/chat", json={"question": "ATP?"}, headers=H).json()
    assert body["citations"] and body["meta"]["degraded"] == ["rerank"]


def test_chat_requires_user_header(setup):
    build, _ = setup
    assert build(FakeStore()).post("/chat", json={"question": "hi"}).status_code == 422


def test_empty_question_and_bad_compare_are_422(setup):
    build, _ = setup
    client = build(FakeStore())
    assert client.post("/chat", json={"question": "  "}, headers=H).status_code == 422
    one = {"question": "compare", "mode": "compare", "document_ids": [str(DOC_A)]}
    assert client.post("/chat", json=one, headers=H).status_code == 422
    assert client.post("/chat", json={**one, "document_ids": None}, headers=H).status_code == 422


def test_compare_mode_retrieves_per_document_and_labels_sources(setup):
    build, _ = setup
    store = FakeStore([
        make_hit("Paper A: glucose yields 30 ATP.", DOC_A, "Paper A", 2),
        make_hit("Paper B: glucose yields 36 ATP.", DOC_B, "Paper B", 5),
    ])
    llm = FakeLLM("A says 30 [c_1]; B says 36 [c_2].")
    client = build(store, llm)
    body = client.post(
        "/chat",
        json={"question": "How do they differ on ATP yield?", "mode": "compare",
              "document_ids": [str(DOC_A), str(DOC_B)]},
        headers=H,
    ).json()
    scoped = [s.document_ids for s in store.scopes]
    assert (DOC_A,) in scoped and (DOC_B,) in scoped  # one search per document
    req = llm.requests[0]
    assert '<document title="Paper A">' in req.user and '<document title="Paper B">' in req.user
    assert "comparison" in req.system
    assert {c["document_title"] for c in body["citations"]} == {"Paper A", "Paper B"}
    assert body["meta"]["mode"] == "compare"


def test_compare_flags_a_document_with_no_relevant_passages(setup):
    build, st = setup
    store = FakeStore([
        make_hit("Paper A: glucose yields 30 ATP.", DOC_A, "Paper A", 2),
        make_hit("Paper B discusses medieval pottery.", DOC_B, "Paper B", 5),
    ])
    client = build(store)
    client.post(
        "/chat",
        json={"question": "ATP?", "mode": "compare", "document_ids": [str(DOC_A), str(DOC_B)]},
        headers=H,
    )
    user = st["llm"].requests[0].user
    assert "(no relevant passages were found in this document)" in user


def test_invalid_citation_markers_are_dropped():
    _, by_marker = pack_context([Group(None, [make_hit()])], 6000)
    text, cites = validate_citations("Real [c_1] and fake [c_9].", by_marker)
    assert "[c_9]" not in text and "[c_1]" in text and len(cites) == 1


def test_pack_context_dedupes_shared_parents_and_respects_budget():
    parent = uuid.uuid4()
    rendered, by_marker = pack_context(
        [Group(None, [make_hit("a", parent=parent), make_hit("b", parent=parent)])], 6000
    )
    assert len(rendered) == 1 and list(by_marker) == ["c_1"]
    big = [make_hit("x" * 4000), make_hit("y" * 4000), make_hit("z" * 4000)]  # ~1000 tokens each
    rendered, _ = pack_context([Group(None, big)], 1500)
    assert len(rendered) == 1  # stops at the budget instead of a fixed k
    assert len(pack_context([Group(None, big)], 100)[0]) == 1  # always keeps at least one


def test_reranker_fake_is_used_for_scores():
    assert FakeReranker().rerank("q", [make_hit("ATP")], 1)[0].score == 0.9


def test_compare_mode_searches_for_the_topic_not_the_comparison_wording(setup):
    build, _ = setup
    store = FakeStore([make_hit("Paper A: glucose yields 30 ATP.", DOC_A, "Paper A", 2)])
    llm = FakeLLM(rewrite="ATP yield per glucose")
    client = build(store, llm)
    client.post(
        "/chat",
        json={"question": "How do the two papers differ on ATP yield?", "mode": "compare",
              "document_ids": [str(DOC_A), str(DOC_B)]},
        headers=H,
    )
    assert len(llm.rewrites) == 1  # rewritten even on the first turn
    assert llm.rewrites[0].system.startswith("Rewrite the user's request")
    assert set(store.queries) == {"ATP yield per glucose"}  # the neutral topic, for every document
