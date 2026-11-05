"""Run the evaluation set against one pipeline configuration.

    python -m eval.run_eval --label baseline --size 800 --mode hybrid_rerank
    python -m eval.run_eval --label vec --size 800 --mode vector --no-generate   # retrieval only
    python -m eval.run_eval --smoke            # 10 questions, recorded model replies (CI)
    python -m eval.run_eval --smoke --record   # re-record those replies with the real model

Calls the same service layer as the API (retrieval, gate, prompt, citation validation), with the
response cache off, and writes JSON plus a Markdown table to eval/results/.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import PipelineConfig, Settings, get_settings
from app.services.generation.llm_client import AnthropicClient, LLMClient, MeteredLLM
from app.services.generation.rag_chain import answer_question, pack_context
from app.services.observability import config_hash
from app.services.turn import Group, Turn
from eval import metrics as M
from eval.judge import JudgeResult, judge_answer
from eval.recorded import RecordingLLM, ReplayLLM
from eval.runtime import EvalRuntime, build_config, sha256_file

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DEFAULT_JUDGE = "claude-opus-5-5"


def load_dataset(ids: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (ROOT / "dataset.jsonl").read_text().splitlines() if line.strip()]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows[:limit] if limit else rows


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True,
            cwd=ROOT, timeout=5
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


async def _one(
    rt: Any, cfg: PipelineConfig, row: dict[str, Any], key_by_id: dict[Any, str],
    doc_ids: tuple[Any, ...], answer_llm: LLMClient | None, judge_llm: LLMClient | None,
    judge_model: str, settings: Settings, retrieval_lock: asyncio.Semaphore, llm_sem: asyncio.Semaphore,
) -> dict[str, Any]:
    q = row["question"]
    async with retrieval_lock:
        t0 = time.perf_counter()
        got = await asyncio.to_thread(rt.retrieve, cfg, q, doc_ids)
        retrieve_ms = int((time.perf_counter() - t0) * 1000)
    ranked = [M.Ranked(key_by_id[h.document_id], h.page_start, h.page_end) for h in got.ranked]
    out: dict[str, Any] = {
        "id": row["id"], "type": row["type"], "question": q,
        "retrieved": [{"doc": r.doc, "p0": r.page_start, "p1": r.page_end, "score": round(h.score, 4)}
                      for r, h in zip(ranked, got.ranked, strict=True)],
        "gated": got.gated, "top_score": None if got.top_score is None else round(got.top_score, 4),
        "timings_ms": {**got.timings_ms, "retrieve_total": retrieve_ms}, "degraded": got.degraded,
    }
    if row["type"] != "unanswerable":
        out["retrieval"] = M.retrieval_metrics(ranked, M.gold_pages(row["sources"]))
    if answer_llm is None:
        out["refused"] = got.gated
        return out

    hits = [] if got.gated else got.ranked[: cfg.top_k]
    turn = Turn(question=q, groups=[Group(None, hits)], gated=got.gated,
                closest=got.ranked[:3] if got.gated else [])
    metered = MeteredLLM(answer_llm)
    async with llm_sem:
        t1 = time.perf_counter()  # start after acquiring the slot: queueing is not generation time
        ans = await answer_question(turn, metered, settings, cfg)
        out["timings_ms"]["generation"] = int((time.perf_counter() - t1) * 1000)
    out["answer"] = ans.answer
    out["citations"] = [{"marker": c.marker, "doc": key_by_id.get(c.document_id, "?"), "page": c.page_start}
                        for c in ans.citations]
    out["answer_cost_usd"] = round(metered.usage.cost_usd, 6)
    out["answer_tokens"] = [metered.usage.input_tokens, metered.usage.output_tokens]
    sources, _ = pack_context(turn.groups, cfg.context_token_budget)

    judged: JudgeResult | None = None
    if judge_llm is not None and not got.gated:
        jm = MeteredLLM(judge_llm)
        async with llm_sem:
            judged = await judge_answer(jm, settings, judge_model, q, row["reference_answer"],
                                        "\n".join(sources), ans.answer)
        out["judge_cost_usd"] = round(jm.usage.cost_usd, 6)
    if judged is not None:
        out["judge"] = judged.model_dump()
        out["refused"] = judged.refused
    else:
        out["refused"] = M.looks_like_refusal(ans.answer, got.gated)
        if got.gated:  # refused before generation: nothing to judge, the label is exact
            out["judge"] = None
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in results if r["type"] != "unanswerable" and "retrieval" in r]
    agg: dict[str, Any] = {"n": len(results), "n_answerable": len(answerable)}
    for k in ("hit@5", "recall@5", "hit@10", "recall@10", "rr"):
        agg["mrr" if k == "rr" else k] = M.avg([r["retrieval"][k] for r in answerable])
    by_type: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in answerable:
        groups[r["type"]].append(r)
    for t, rs in groups.items():
        by_type[t] = {"n": len(rs), "hit@5": M.avg([x["retrieval"]["hit@5"] for x in rs]),
                      "recall@5": M.avg([x["retrieval"]["recall@5"] for x in rs]),
                      "mrr": M.avg([x["retrieval"]["rr"] for x in rs])}
    agg["by_type"] = by_type

    if any("answer" in r for r in results):
        rows = [r for r in results if "answer" in r]
        agg.update(M.refusal_metrics(rows))
        judged = [r for r in rows if r.get("judge")]
        an_judged = [r for r in judged if r["type"] != "unanswerable"]
        non_refused = [r for r in an_judged if not r["judge"]["refused"]]
        agg["faithfulness"] = M.avg([r["judge"]["faithfulness"] for r in non_refused])
        agg["relevance"] = M.avg([r["judge"]["relevance"] / 5 for r in an_judged])
        # a refused answerable question is incorrect, so it counts as 0
        agg["correctness"] = M.avg(
            [0.0 if r["refused"] else (M.correctness_score(r["judge"]["correctness"]) or 0.0)
             for r in rows if r["type"] != "unanswerable" and (r.get("judge") or r["refused"])]
        )
        checks = [c for r in non_refused for c in r["judge"]["citations"]]
        agg["citation_precision"] = M.avg([1.0 if c["supports_claim"] else 0.0 for c in checks])
        agg["answer_cost_usd_per_query"] = M.avg([r.get("answer_cost_usd", 0.0) for r in rows])
        agg["judge_cost_usd_per_query"] = M.avg([r.get("judge_cost_usd", 0.0) for r in rows])
    stages: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in r["timings_ms"].items():
            stages[k].append(float(v))
    agg["latency_ms"] = {k: {"p50": M.percentile(v, 50), "p95": M.percentile(v, 95)} for k, v in stages.items()}
    return agg


async def evaluate(
    rt: Any, cfg: PipelineConfig, rows: list[dict[str, Any]], *, settings: Settings,
    answer_llm: LLMClient | None, judge_llm: LLMClient | None, judge_model: str = DEFAULT_JUDGE,
    concurrency: int = 4,
) -> dict[str, Any]:
    doc_map = await asyncio.to_thread(rt.index, cfg)
    key_by_id = {v: k for k, v in doc_map.items()}
    doc_ids = tuple(doc_map.values())
    lock, sem = asyncio.Semaphore(1), asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_one(rt, cfg, r, key_by_id, doc_ids, answer_llm, judge_llm, judge_model, settings, lock, sem)
          for r in rows)
    )
    return {"questions": list(results), "aggregate": aggregate(list(results))}


def result_envelope(label: str, cfg: PipelineConfig, settings: Settings, judge_model: str, body: dict[str, Any], n: int) -> dict[str, Any]:
    ds = ROOT / "dataset.jsonl"
    return {
        "label": label, "created": datetime.now(UTC).isoformat(timespec="seconds"), "git_sha": git_sha(),
        "config": cfg.model_dump(), "config_hash": config_hash(cfg, min_score=settings.retrieval_min_score,
                                                                embed=settings.embedding_model, reranker=settings.reranker),
        "dataset": {"file": ds.name, "sha256": sha256_file(ds)[:16], "questions": n},
        "models": {"answer": settings.llm_model, "judge": judge_model, "embedding": settings.embedding_model,
                   "reranker": settings.reranker, "min_score": settings.retrieval_min_score},
        **body,
    }


def markdown(env: dict[str, Any]) -> str:
    a, c = env["aggregate"], env["config"]
    f = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    header = (
        f"`{c['chunker']}` parent {c['parent_tokens']} / child {c['child_tokens']} tokens, "
        f"`{c['retrieval_mode']}`; config `{env['config_hash']}`, git `{env['git_sha']}`, "
        f"{a['n']} questions ({a['n_answerable']} answerable)."
    )
    lines = [f"# {env['label']}", "", header, "", "| metric | value |", "|---|---|"]
    for k in ("hit@5", "recall@5", "hit@10", "recall@10", "mrr", "faithfulness", "correctness", "relevance",
              "citation_precision", "refusal_accuracy", "false_refusal_rate",
              "answer_cost_usd_per_query", "judge_cost_usd_per_query"):
        if k in a:
            lines.append(f"| {k} | {f(a[k])} |")
    lat = a["latency_ms"]
    lines += ["", "| stage | p50 ms | p95 ms |", "|---|---|---|"]
    lines += [f"| {k} | {f(v['p50'])} | {f(v['p95'])} |" for k, v in lat.items()]
    return "\n".join(lines) + "\n"


def make_llm(settings: Settings, record: Path | None, replay: Path | None) -> tuple[LLMClient, Any]:
    if replay is not None:
        return ReplayLLM(replay), None
    key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set (needed for generation and judging)")
    real = AnthropicClient(key, max_concurrency=4)
    if record is not None:
        rec = RecordingLLM(real, record)
        return rec, rec
    return real, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="run")
    ap.add_argument("--chunker", default="recursive", choices=["recursive", "semantic"])
    ap.add_argument("--size", type=int, default=800, help="section (parent) tokens; children are size/4")
    ap.add_argument("--mode", default="hybrid_rerank", choices=["vector", "hybrid", "hybrid_rerank"])
    ap.add_argument("--no-generate", action="store_true", help="retrieval metrics only (no model calls)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--ids", default="", help="comma separated question ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="the smoke subset with recorded model replies")
    ap.add_argument("--record", action="store_true", help="with --smoke: call the real model and save replies")
    args = ap.parse_args(argv)

    settings = get_settings()
    cfg = build_config(args.chunker, args.size, args.mode)
    ids = [i for i in args.ids.split(",") if i] or None
    generate = not args.no_generate
    answer_llm: LLMClient | None = None
    judge_llm: LLMClient | None = None
    recorder: Any = None
    if args.smoke:
        ids = json.loads((ROOT / "smoke.json").read_text())["ids"]
        rec_path = ROOT / "recordings" / "smoke.json"
        answer_llm, recorder = make_llm(settings, rec_path if args.record else None, None if args.record else rec_path)
        judge_llm = answer_llm  # one recording holds both the answer and the judge replies
    elif generate:
        answer_llm, _ = make_llm(settings, None, None)
        judge_llm = None if args.no_judge else answer_llm
    rows = load_dataset(ids, args.limit or None)
    rt = EvalRuntime(settings)
    print(f"evaluating {len(rows)} questions, {cfg.chunker} {args.size}, {cfg.retrieval_mode}, generate={generate}")
    t0 = time.time()
    body = asyncio.run(evaluate(rt, cfg, rows, settings=settings, answer_llm=answer_llm, judge_llm=judge_llm,
                                judge_model=args.judge_model, concurrency=args.concurrency))
    if recorder is not None:
        recorder.save()
    env = result_envelope(args.label, cfg, settings, args.judge_model, body, len(rows))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.label}.json").write_text(json.dumps(env, indent=1, ensure_ascii=False))
    (RESULTS / f"{args.label}.md").write_text(markdown(env))
    print(markdown(env))
    print(f"done in {time.time() - t0:.0f}s -> eval/results/{args.label}.json")
    if args.smoke:
        return check_baseline(env["aggregate"])
    return 0


def check_baseline(agg: dict[str, Any]) -> int:
    """Fail when quality drops more than the allowed margin below the stored baseline."""
    path = ROOT / "baseline.json"
    if not path.exists():
        print("no baseline.json: skipping the regression check")
        return 0
    base = json.loads(path.read_text())
    failures = []
    for k, floor in base["floors"].items():
        v = agg.get(k)
        if v is None or v < floor:
            failures.append(f"{k}: {v} < floor {floor}")
    for msg in failures:
        print("REGRESSION:", msg)
    print("smoke regression check:", "FAILED" if failures else "passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
