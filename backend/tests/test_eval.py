import json
import uuid
from pathlib import Path

import pytest

from app.config import PipelineConfig, Settings
from app.services.vectorstore import Hit
from eval import metrics as M
from eval import run_eval
from eval.recorded import RecordingLLM, RecordingMiss, ReplayLLM
from eval.run_eval import aggregate, check_baseline, evaluate, load_dataset
from eval.runtime import Retrieved, build_config

from .fakes import FakeLLM

EVAL = Path(run_eval.__file__).parent

# --- metrics ----------------------------------------------------------------------------------


def R(doc, a, b=None):
    return M.Ranked(doc, a, b or a)


def test_retrieval_metrics_cover_hit_recall_and_rank():
    gold = {("bert", 3), ("bert", 13)}
    ranked = [R("rag", 3), R("bert", 5), R("bert", 3), R("bert", 12, 13)]
    m = M.retrieval_metrics(ranked, gold, ks=(2, 5))
    assert m["hit@2"] == 0 and m["recall@2"] == 0  # nothing relevant in the top 2
    assert m["hit@5"] == 1 and m["recall@5"] == 1.0  # page 3 and the 12-13 range cover both
    assert m["rr"] == pytest.approx(1 / 3)  # first relevant hit is at rank 3
    one = M.retrieval_metrics([R("bert", 3)], gold, ks=(5,))
    assert one["recall@5"] == 0.5 and one["rr"] == 1.0
    assert M.retrieval_metrics([], gold, ks=(5,)) == {"hit@5": 0.0, "recall@5": 0.0, "rr": 0.0}


def test_a_hit_needs_the_right_document_not_just_the_page_number():
    assert M.retrieval_metrics([R("rag", 3)], {("bert", 3)}, ks=(5,))["hit@5"] == 0.0


def test_comparison_questions_need_both_documents_for_full_recall():
    gold = M.gold_pages([{"doc": "bert", "pages": [3]}, {"doc": "attention", "pages": [9]}])
    only_one = M.retrieval_metrics([R("bert", 3)], gold, ks=(5,))
    both = M.retrieval_metrics([R("bert", 3), R("attention", 9)], gold, ks=(5,))
    assert only_one["recall@5"] == 0.5 and both["recall@5"] == 1.0


def test_percentiles_and_averages():
    assert M.percentile([1, 2, 3, 4], 50) == 2.5 and M.percentile([5], 95) == 5 and M.percentile([], 50) is None
    assert M.percentile(list(range(1, 101)), 95) == pytest.approx(95.05, abs=1e-3)
    assert M.avg([1, 2, None]) == 1.5 and M.avg([]) is None  # type: ignore[list-item]


@pytest.mark.parametrize(
    "text",
    ["The sources don't contain that.", "The documents do not mention a carbon footprint.",
     "There is no information about this.", "I cannot find this in your documents.",
     "Your documents don't seem to cover this question, so I won't guess."],
)
def test_refusal_detector_recognises_common_phrasings(text):
    assert M.looks_like_refusal(text, gated=False)


def test_refusal_detector_does_not_flag_real_answers():
    assert not M.looks_like_refusal("BERT-BASE has 12 layers [c_1].", gated=False)
    assert M.looks_like_refusal("anything", gated=True)


# --- evaluate() with a fake runtime -----------------------------------------------------------

DOC_ID = {"bert": uuid.uuid4(), "attention": uuid.uuid4()}


def hit(doc, page, score=0.9, text="BERT-BASE has 12 layers."):
    pid = uuid.uuid4()
    return Hit(pid, DOC_ID[doc], doc, page, page, score, text, text, child_id=uuid.uuid4())


class FakeRuntime:
    """Scripted retrieval: question id -> Retrieved."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def index(self, cfg):
        return dict(DOC_ID)

    def retrieve(self, cfg, query, doc_ids):
        self.calls.append(query)
        return self.script[query]


def judge_json(**kw):
    base = {"refused": False, "faithfulness": 1.0, "correctness": "correct", "relevance": 5,
            "citations": [{"marker": "c_1", "supports_claim": True}], "notes": "ok"}
    return json.dumps({**base, **kw})


ROWS = [
    {"id": "a1", "type": "factual", "question": "q-a1", "reference_answer": "12", "sources": [{"doc": "bert", "pages": [3], "evidence": "x"}]},
    {"id": "a2", "type": "factual", "question": "q-a2", "reference_answer": "8", "sources": [{"doc": "attention", "pages": [5], "evidence": "x"}]},
    {"id": "a3", "type": "table", "question": "q-a3", "reference_answer": "86.3", "sources": [{"doc": "bert", "pages": [7], "evidence": "x"}]},
    {"id": "u1", "type": "unanswerable", "question": "q-u1", "reference_answer": "n/a", "sources": []},
    {"id": "u2", "type": "unanswerable", "question": "q-u2", "reference_answer": "n/a", "sources": []},
]
SCRIPT = {
    "q-a1": Retrieved([hit("bert", 3)], False, 0.99),                      # perfect retrieval
    "q-a2": Retrieved([hit("bert", 1), hit("attention", 5)], False, 0.8),  # hit at rank 2
    "q-a3": Retrieved([hit("bert", 1, 0.01)], True, 0.01),                 # wrongly gated
    "q-u1": Retrieved([hit("bert", 1, 0.001)], True, 0.001),               # correctly gated
    "q-u2": Retrieved([hit("bert", 2, 0.8)], False, 0.8),                  # not gated, model must refuse
}


def make_llms():
    llm = FakeLLM()

    def handler(req):
        if req.system.startswith("You are a strict evaluator"):
            q = req.user.split("QUESTION:\n")[1].split("\n")[0]
            if q == "q-a2":
                return judge_json(correctness="partial", faithfulness=0.5,
                                  citations=[{"marker": "c_1", "supports_claim": False}])
            if q == "q-u2":
                return judge_json(refused=True, faithfulness=1.0, correctness="correct", citations=[])
            return judge_json()
        q = req.user.split("Question: ")[1]
        return "The sources do not contain that." if q == "q-u2" else f"Answer to {q} [c_1]."

    llm.handler = handler
    return llm


async def run():
    rt = FakeRuntime(SCRIPT)
    llm = make_llms()
    settings = Settings(llm_model="claude-sonnet-5-5")
    res = await evaluate(rt, PipelineConfig(), ROWS, settings=settings, answer_llm=llm, judge_llm=llm, judge_model="claude-opus-5-5")
    return res, llm, rt


async def test_evaluate_computes_retrieval_generation_and_refusal_metrics():
    res, llm, _ = await run()
    a = res["aggregate"]
    assert a["n"] == 5 and a["n_answerable"] == 3
    assert a["hit@5"] == pytest.approx(2 / 3, abs=1e-3)  # a3's scripted retrieval misses its gold page
    assert a["mrr"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
    by_id = {q["id"]: q for q in res["questions"]}
    assert by_id["a3"]["gated"] and by_id["a3"]["refused"]  # the gate refused an answerable question
    assert by_id["u1"]["gated"] and by_id["u1"]["refused"]
    assert by_id["u2"]["refused"] and by_id["u2"]["judge"]["refused"]  # the judge says the model refused
    assert a["refusal_accuracy"] == 1.0
    assert a["false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-3)
    # correctness: a1 correct (1), a2 partial (.5), a3 refused answerable counts as 0
    assert a["correctness"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
    assert a["faithfulness"] == pytest.approx((1.0 + 0.5) / 2, abs=1e-3)  # refused answers excluded
    assert a["citation_precision"] == pytest.approx(0.5, abs=1e-3)  # one supporting and one non-supporting check
    assert a["relevance"] == 1.0
    # gated questions never reach the answering model or the judge
    answered = [r for r in llm.requests if not r.system.startswith("You are a strict")]
    assert len(answered) == 3  # a1, a2, u2


async def test_cost_and_latency_are_reported_per_stage():
    res, _, _ = await run()
    a = res["aggregate"]
    assert a["answer_cost_usd_per_query"] > 0 and a["judge_cost_usd_per_query"] > 0
    assert {"retrieve_total", "generation"} <= set(a["latency_ms"])
    assert a["latency_ms"]["retrieve_total"]["p95"] >= a["latency_ms"]["retrieve_total"]["p50"]
    q = next(x for x in res["questions"] if x["id"] == "a1")
    assert q["answer_tokens"] == [10, 10] and q["citations"][0]["doc"] in DOC_ID


async def test_retrieval_only_mode_makes_no_model_calls():
    rt = FakeRuntime(SCRIPT)
    res = await evaluate(rt, PipelineConfig(), ROWS, settings=Settings(), answer_llm=None, judge_llm=None)
    a = res["aggregate"]
    assert "faithfulness" not in a and a["hit@5"] == pytest.approx(2 / 3, abs=1e-3) and len(rt.calls) == 5


def test_smoke_check_fails_on_a_regression_and_passes_above_the_floor(tmp_path, monkeypatch, capsys):
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps({"floors": {"hit@5": 0.9, "refusal_accuracy": 1.0}}))
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    assert check_baseline({"hit@5": 0.95, "refusal_accuracy": 1.0}) == 0
    assert check_baseline({"hit@5": 0.7, "refusal_accuracy": 1.0}) == 1
    assert "REGRESSION: hit@5" in capsys.readouterr().out
    assert check_baseline({"hit@5": 0.95}) == 1  # a missing metric is a failure, not a pass


# --- recording and replay ---------------------------------------------------------------------


async def test_recorded_replies_replay_identically_and_a_changed_prompt_is_a_miss(tmp_path):
    from app.services.generation.llm_client import LLMRequest

    path = tmp_path / "rec.json"
    inner = FakeLLM("recorded answer")
    rec = RecordingLLM(inner, path)
    req = LLMRequest("system", "user text", "m")
    first = await rec.complete(req)
    rec.save()
    assert first.text == "recorded answer"

    replay = ReplayLLM(path)
    again = await replay.complete(req)
    assert (again.text, again.input_tokens, again.output_tokens) == (first.text, first.input_tokens, first.output_tokens)
    with pytest.raises(RecordingMiss, match="re-record"):
        await replay.complete(LLMRequest("system", "a different prompt", "m"))


# --- dataset ----------------------------------------------------------------------------------


def test_dataset_meets_the_requirements_and_is_well_formed():
    rows = load_dataset()
    assert len(rows) >= 30
    assert sum(r["type"] == "unanswerable" for r in rows) >= 5
    assert len({r["id"] for r in rows}) == len(rows)
    docs = {d["key"] for d in json.loads((EVAL / "docs.json").read_text())}
    assert len(docs) >= 3
    used = set()
    for r in rows:
        assert r["question"].strip() and r["reference_answer"].strip()
        if r["type"] == "unanswerable":
            assert r["sources"] == []
        else:
            assert r["sources"], r["id"]
            for s in r["sources"]:
                assert s["doc"] in docs and s["pages"] and all(isinstance(p, int) and p >= 1 for p in s["pages"])
                assert len(s["evidence"]) >= 10
                used.add(s["doc"])
    assert used == docs  # every document is actually asked about
    assert {r["type"] for r in rows} >= {"factual", "multi_hop", "table", "comparison", "unanswerable"}


def test_smoke_subset_exists_and_covers_the_question_types():
    ids = json.loads((EVAL / "smoke.json").read_text())["ids"]
    rows = {r["id"]: r for r in load_dataset()}
    assert len(ids) == 10 and all(i in rows for i in ids)
    assert {rows[i]["type"] for i in ids} >= {"factual", "table", "multi_hop", "comparison", "unanswerable"}


def test_sweep_sizes_map_to_parent_and_child_tokens():
    cfg = build_config("semantic", 1200, "vector")
    assert (cfg.parent_tokens, cfg.child_tokens, cfg.chunker, cfg.retrieval_mode) == (1200, 300, "semantic", "vector")
    assert build_config("recursive", 400, "hybrid").child_tokens == 100


def test_aggregate_of_nothing_is_safe():
    assert aggregate([])["n"] == 0
