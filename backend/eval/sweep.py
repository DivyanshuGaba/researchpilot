"""Run the retrieval grid (chunker x size x retrieval mode), optionally with generation.

    python -m eval.sweep                       # retrieval metrics only: no model calls, free
    python -m eval.sweep --generate recursive-800-hybrid_rerank,recursive-800-vector
"""

import argparse
import asyncio
import json
import sys

from app.config import get_settings
from eval.run_eval import (
    DEFAULT_JUDGE,
    RESULTS,
    evaluate,
    load_dataset,
    make_llm,
    markdown,
    result_envelope,
)
from eval.runtime import EvalRuntime, build_config

SIZES = (400, 800, 1200)
MODES = ("vector", "hybrid", "hybrid_rerank")
GRID = [("recursive", s, m) for s in SIZES for m in MODES] + [("semantic", 800, m) for m in MODES]


def label(chunker: str, size: int, mode: str, prefix: str = "retr") -> str:
    return f"{prefix}-{chunker}-{size}-{mode}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", default="", help="comma separated chunker-size-mode configs to also run with generation and judging")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--skip-retrieval", action="store_true", help="skip the retrieval-only grid")
    args = ap.parse_args()

    settings = get_settings()
    rt = EvalRuntime(settings)
    rows = load_dataset()
    RESULTS.mkdir(exist_ok=True)
    gen_targets = {g for g in args.generate.split(",") if g}
    llm = make_llm(settings, None, None)[0] if gen_targets else None

    for chunker, size, mode in [] if args.skip_retrieval else GRID:
        cfg = build_config(chunker, size, mode)
        name = label(chunker, size, mode)
        body = asyncio.run(evaluate(rt, cfg, rows, settings=settings, answer_llm=None, judge_llm=None))
        env = result_envelope(name, cfg, settings, args.judge_model, body, len(rows))
        (RESULTS / f"{name}.json").write_text(json.dumps(env, indent=1, ensure_ascii=False))
        (RESULTS / f"{name}.md").write_text(markdown(env))
        a = body["aggregate"]
        print(f"{name:42} hit@5={a['hit@5']:.3f} recall@5={a['recall@5']:.3f} mrr={a['mrr']:.3f}", flush=True)

    for target in sorted(gen_targets):
        chunker, size_s, mode = target.split("-", 2)
        cfg = build_config(chunker, int(size_s), mode)
        name = label(chunker, int(size_s), mode, prefix="gen")
        body = asyncio.run(evaluate(rt, cfg, rows, settings=settings, answer_llm=llm, judge_llm=llm,
                                    judge_model=args.judge_model, concurrency=args.concurrency))
        env = result_envelope(name, cfg, settings, args.judge_model, body, len(rows))
        (RESULTS / f"{name}.json").write_text(json.dumps(env, indent=1, ensure_ascii=False))
        (RESULTS / f"{name}.md").write_text(markdown(env))
        a = body["aggregate"]
        print(f"{name:42} faith={a.get('faithfulness')} correct={a.get('correctness')} "
              f"refusal_acc={a.get('refusal_accuracy')} false_refusal={a.get('false_refusal_rate')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
