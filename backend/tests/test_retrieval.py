import uuid

from app.config import PipelineConfig
from app.services.retrieval.hybrid import diversify, rrf
from app.services.retrieval.service import Candidates, finalize, gather_candidates
from app.services.vectorstore import Hit, Scope, or_tsquery

DOC = uuid.uuid4()


def hit(name: str, parent: str | None = None, score: float = 0.0) -> Hit:
    ids = {n: uuid.uuid5(uuid.NAMESPACE_DNS, n) for n in (name, parent or name)}
    return Hit(ids[parent or name], DOC, "Doc", 1, 1, score, name, name, child_id=ids[name])


def test_rrf_rewards_agreement_between_lists():
    a, b, c, d, e = (hit(x) for x in "abcde")
    fused = rrf([[a, b, c], [d, b, e]], k=60)
    assert fused[0].child_text == "b"  # the only item both lists agree on
    assert {h.child_text for h in fused} == set("abcde")  # union, no duplicates
    assert len(fused) == 5


def test_rrf_uses_ranks_not_scores():
    hi = hit("hi", score=999.0)
    lo = hit("lo", score=0.001)
    fused = rrf([[lo], [lo], [hi]], k=60)
    assert fused[0].child_text == "lo"


def test_diversify_caps_hits_per_parent():
    hits = [hit(f"c{i}", parent="P") for i in range(5)] + [hit("other")]
    out = diversify(hits, 2)
    assert [h.child_text for h in out] == ["c0", "c1", "other"]


def test_or_tsquery_is_or_and_sanitized():
    assert or_tsquery("What is the ATP yield? (x) 'drop;table'") == (
        "what | is | the | atp | yield | drop | table"
    )
    assert or_tsquery("!!! ???") == ""


class FakeStore:
    def __init__(self, dense, sparse, boom=False):
        self.dense, self.sparse, self.boom = dense, sparse, boom

    def search(self, vector, k, scope):
        return self.dense

    def keyword_search(self, query, k, scope):
        if self.boom:
            raise RuntimeError("fts down")
        return self.sparse


class FakeEmb:
    model_id, dim = "f", 2

    def embed_query(self, t):
        return [1.0, 0.0]


class FakeReranker:
    """Scores by whether the passage contains the word 'relevant'."""

    def __init__(self, boom=False):
        self.boom = boom

    def rerank(self, query, hits, top_n):
        if self.boom:
            raise RuntimeError("model crashed")
        import dataclasses

        scored = [dataclasses.replace(h, score=0.9 if "relevant" in h.child_text else 0.02) for h in hits]
        return sorted(scored, key=lambda h: -h.score)[:top_n]


SCOPE = Scope(uuid.uuid4())
CFG = PipelineConfig()


def test_hybrid_unions_dense_and_sparse():
    store = FakeStore([hit("d1"), hit("shared")], [hit("shared"), hit("s1")])
    cands = gather_candidates(store, FakeEmb(), "q", SCOPE, CFG)
    assert {h.child_text for h in cands.hits} == {"d1", "shared", "s1"}
    assert cands.hits[0].child_text == "shared"


def test_vector_mode_skips_keyword_search():
    store = FakeStore([hit("d1")], [], boom=True)
    cands = gather_candidates(store, FakeEmb(), "q", SCOPE, PipelineConfig(retrieval_mode="vector"))
    assert [h.child_text for h in cands.hits] == ["d1"] and not cands.degraded


def test_keyword_failure_degrades_to_dense():
    store = FakeStore([hit("d1")], [], boom=True)
    cands = gather_candidates(store, FakeEmb(), "q", SCOPE, CFG)
    assert [h.child_text for h in cands.hits] == ["d1"] and cands.degraded == ["sparse"]


def test_rerank_reorders_and_passes_gate():
    cands = Candidates([hit("noise"), hit("relevant passage")])
    r = finalize(cands, "q", FakeReranker(), CFG, min_score=0.35)
    assert not r.gated and r.hits[0].child_text == "relevant passage"


def test_gate_refuses_when_best_rerank_score_is_low():
    cands = Candidates([hit("noise one"), hit("noise two")])
    r = finalize(cands, "q", FakeReranker(), CFG, min_score=0.35)
    assert r.gated and r.hits == [] and len(r.closest) == 2 and r.top_score == 0.02


def test_rerank_failure_falls_back_to_fused_without_gating():
    cands = Candidates([hit("a"), hit("b")])
    r = finalize(cands, "q", FakeReranker(boom=True), CFG, min_score=0.35)
    assert not r.gated and len(r.hits) == 2 and "rerank" in r.degraded


def test_missing_reranker_is_degraded_not_gated():
    r = finalize(Candidates([hit("a")]), "q", None, CFG, min_score=0.35)
    assert not r.gated and r.degraded == ["rerank"]
