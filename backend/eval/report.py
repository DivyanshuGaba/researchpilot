"""Build docs/eval_results.md from eval/results/*.json, so the published tables are generated
from the stored runs and cannot drift from them.

    python -m eval.report
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = ROOT.parent.parent / "docs" / "eval_results.md"
SIZES, MODES = (400, 800, 1200), ("vector", "hybrid", "hybrid_rerank")


def load(name: str) -> dict[str, Any] | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def se(p: float, n: int) -> float:
    """Standard error of a proportion: how much to trust a difference between two rows."""
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def retrieval_table() -> list[str]:
    lines = ["| chunking | retrieval | hit@5 | recall@5 | hit@10 | recall@10 | MRR | rerank p50 (ms) | retrieve total p50 (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = [("recursive", s, m) for s in SIZES for m in MODES] + [("semantic", 800, m) for m in MODES]
    for chunker, size, mode in rows:
        r = load(f"retr-{chunker}-{size}-{mode}")
        if r is None:
            continue
        a = r["aggregate"]
        lat = a["latency_ms"]
        rr = lat.get("rerank", {}).get("p50")
        lines.append(f"| {chunker} {size}/{size // 4} | {mode} | {f(a['hit@5'])} | {f(a['recall@5'])} | {f(a['hit@10'])} | "
                     f"{f(a['recall@10'])} | {f(a['mrr'])} | {f(rr, 0) if rr is not None else '-'} | {f(lat['retrieve_total']['p50'], 0)} |")
    return lines


def generation_table() -> tuple[list[str], list[str]]:
    cols = ["faithfulness", "correctness", "relevance", "citation_precision", "refusal_accuracy", "false_refusal_rate"]
    lines = ["| config | " + " | ".join(c.replace("_", " ") for c in cols) + " | answer $/query | judge $/query |",
             "|---|" + "---|" * (len(cols) + 2)]
    missing = []
    for target in ("recursive-800-hybrid_rerank", "recursive-800-vector", "recursive-800-hybrid",
                   "recursive-1200-hybrid_rerank", "semantic-800-hybrid_rerank"):
        r = load(f"gen-{target}")
        if r is None:
            missing.append(target)
            continue
        a = r["aggregate"]
        lines.append(f"| {target} | " + " | ".join(f(a.get(c)) for c in cols)
                     + f" | ${f(a.get('answer_cost_usd_per_query'), 4)} | ${f(a.get('judge_cost_usd_per_query'), 4)} |")
    return lines, missing


def by_type_table(name: str) -> list[str]:
    r = load(name)
    if r is None:
        return []
    lines = ["| question type | n | hit@5 | recall@5 | MRR |", "|---|---|---|---|---|"]
    for t, v in r["aggregate"]["by_type"].items():
        lines.append(f"| {t} | {v['n']} | {f(v['hit@5'])} | {f(v['recall@5'])} | {f(v['mrr'])} |")
    return lines


def cache_section() -> list[str]:
    c = load("cache")
    if c is None:
        return ["_not run_"]
    lines = [f"Embedding model `{c['embedding_model']}`; {c['n_different']} near-duplicate pairs whose correct answers differ "
             f"(a hit is a wrong answer) and {c['n_same']} true paraphrases (a hit saves a model call).", "",
             "| similarity threshold | wrong-answer hits | paraphrase hits |", "|---|---|---|"]
    for t in c["thresholds"]:
        lines.append(f"| {t['threshold']:.2f} | {t['false_hits']}/{c['n_different']} ({t['false_hit_rate']:.0%}) | "
                     f"{t['true_hits']}/{c['n_same']} ({t['true_hit_rate']:.0%}) |")
    worst = sorted((p for p in c["pairs"] if p["kind"] == "different_answer"), key=lambda p: -p["similarity"])[:3]
    lines += ["", "Closest different-answer pairs:", ""]
    lines += [f"- {p['similarity']:.3f}: “{p['a']}” vs “{p['b']}”" for p in worst]
    return lines


def injection_section() -> list[str]:
    r = load("injection")
    if r is None:
        return ["_Not yet run with the corrected scoring (obeying an instruction versus merely quoting it); "
                "run `python -m eval.adversarial injection`. The unit tests in `tests/test_guardrails.py` cover the defenses "
                "with a scripted model that obeys._"]
    lines = [f"Model `{r['model']}`, {r['trials']} trials per attack, question: “{r['question']}”.", "",
             "| attack | flagged at ingestion | answered correctly | model refused | obeyed (raw) | leaked after defenses |", "|---|---|---|---|---|---|"]
    for name, a in r["attacks"].items():
        n = r["trials"]
        lines.append(f"| {name} | {a['retrieved_chunks_flagged']} | {a['answered_correctly']}/{n} | {a.get('model_refused', '?')}/{n} | "
                     f"{a['model_complied_raw']}/{n} | {a['leaked_after_defenses']}/{n} |")
    return lines


def build() -> str:
    base = load("retr-recursive-800-hybrid_rerank")
    meta = ""
    if base:
        n, na = base["aggregate"]["n"], base["aggregate"]["n_answerable"]
        meta = (f"{n} questions ({na} answerable, {n - na} unanswerable) over 4 open-access arXiv papers, "
                f"dataset `{base['dataset']['sha256']}`, embedding `{base['models']['embedding']}`, "
                f"reranker `{base['models']['reranker']}`, answers by `{base['models']['answer']}`, "
                f"judge `{base['models']['judge']}`, confidence threshold {base['models']['min_score']}.")
    gen, missing = generation_table()
    out = [
        "# Evaluation results", "",
        "Generated by `python -m eval.report` from `backend/eval/results/*.json`. Do not edit by hand.", "",
        "## Setup", "", meta, "",
        "- **Corpus:** *Attention Is All You Need*, *BERT*, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, "
        "*Lost in the Middle* (arXiv, pinned by SHA-256 in `eval/docs.json`; fetch with `python -m eval.fetch_docs`).",
        "- **Questions:** hand written against the text. Every answer carries a verbatim evidence phrase that "
        "`eval/verify_dataset.py` checks appears on the stated page, and the unanswerable questions' key terms are checked to be absent. "
        "Types: factual, multi-hop, table, cross-document comparison, unanswerable (5 off-topic, 3 near-miss).",
        "- **Chunking size** is the section (parent) size sent to the model; the searched children are a quarter of it "
        "(`400/100`, `800/200`, `1200/300` tokens).",
        "- **Retrieval metrics** are page-level: a hit is a retrieved chunk whose page range overlaps a gold page of the right document. "
        "`recall@k` is the share of gold pages covered by the top k, `MRR` the reciprocal rank of the first hit. "
        "They use the reranked, diversified top 10 before the confidence gate.",
        "- **Generation metrics** come from an LLM judge that sees the sources the model saw: *faithfulness* is the share of claims "
        "supported by those sources (refusals excluded), *correctness* scores correct/partial/incorrect as 1/0.5/0 against the reference "
        "(a refusal of an answerable question counts 0), *relevance* is 1-5 scaled to 0-1, *citation precision* is the share of `[c_N]` "
        "markers whose source supports the claim. *Refusal accuracy* is the share of unanswerable questions refused; "
        "*false refusal rate* the share of answerable ones refused.", "",
        "## How much to trust a difference", "",
        "With 41 answerable questions one question is worth 0.024, and the standard error of a proportion near 0.85 is about 0.056. "
        "**Differences smaller than roughly 0.05 to 0.1 between two rows are within noise.** Treat the tables as ranking evidence, "
        "not proof, and re-run on a larger set before changing a default for a small gain.", "",
        "## Retrieval: chunking and strategy", "", *retrieval_table(), "",
        "Latency is the time to embed, search, fuse and rerank one query on a laptop CPU (`bge-small`, `ms-marco-MiniLM`).", "",
        "### By question type (recursive 800, hybrid + rerank)", "", *by_type_table("retr-recursive-800-hybrid_rerank"), "",
        "## Generation quality and cost", "", *gen, "",
    ]
    if missing:
        out += [f"_Not yet run: {', '.join(missing)}. The generation runs stopped when the API account ran out of credit; "
                "re-run with `python -m eval.sweep --skip-retrieval --generate <config,...>`._", ""]
    out += [
        "Cost is per question at list prices (Sonnet 5.5 $2/$10 and Opus 5.5 $4/$20 per million tokens). "
        "The answer cost is what a production query costs; the judge cost is evaluation overhead. Generation latency is not "
        "reported: these runs timed generation from before the concurrency slot was acquired, so the numbers included queueing. "
        "The timer is fixed for future runs; production latency per stage is in `query_logs` and `/admin/metrics`.", "",
        "## Semantic cache: wrong-answer risk vs savings", "", *cache_section(), "",
        "## Prompt injection", "", *injection_section(), "",
    ]
    return "\n".join(out)


def main() -> int:
    OUT.write_text(build() + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
