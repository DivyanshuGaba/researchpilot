"""Pure metric functions for the evaluation harness (unit tested, no I/O)."""

import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

# (document key, page) pairs are the unit of relevance: a hit is a retrieved chunk whose page
# range overlaps a gold page of the right document.
Page = tuple[str, int]


@dataclass(frozen=True)
class Ranked:
    doc: str
    page_start: int
    page_end: int

    def pages(self) -> set[Page]:
        return {(self.doc, p) for p in range(self.page_start, self.page_end + 1)}


def gold_pages(sources: list[dict[str, Any]]) -> set[Page]:
    return {(s["doc"], p) for s in sources for p in s["pages"]}


def retrieval_metrics(ranked: list[Ranked], gold: set[Page], ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """hit@k (any gold page in the top k), recall@k (share of gold pages covered by the top k)
    and the reciprocal rank of the first hit. Only defined for answerable questions."""
    out: dict[str, float] = {}
    for k in ks:
        covered: set[Page] = set()
        for r in ranked[:k]:
            covered |= r.pages() & gold
        out[f"hit@{k}"] = 1.0 if covered else 0.0
        out[f"recall@{k}"] = len(covered) / len(gold) if gold else 0.0
    out["rr"] = 0.0
    for rank, r in enumerate(ranked, start=1):
        if r.pages() & gold:
            out["rr"] = 1.0 / rank
            break
    return out


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def avg(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


_REFUSAL = re.compile(
    r"don'?t seem to cover|do(?:es)?(?: not|n't) (?:contain|cover|include|mention|say|state|provide|specify|address)"
    r"|(?:no|not enough|insufficient) (?:relevant )?information|cannot (?:find|answer)|can'?t (?:find|answer)"
    r"|(?:isn'?t|is not|aren'?t|are not) (?:mentioned|covered|stated|addressed|in the (?:sources|documents))"
    r"|doesn'?t (?:say|mention|state|contain|provide)|couldn'?t find|no mention of",
    re.IGNORECASE,
)


def looks_like_refusal(answer: str, gated: bool) -> bool:
    """Fallback refusal detector for runs without a judge. The judge's own flag is preferred."""
    return gated or bool(_REFUSAL.search(answer[:400]))


def refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Refusal accuracy on unanswerable questions, and false refusals on answerable ones."""
    un = [r for r in rows if r["type"] == "unanswerable"]
    an = [r for r in rows if r["type"] != "unanswerable"]
    return {
        "refusal_accuracy": avg([1.0 if r["refused"] else 0.0 for r in un]),
        "false_refusal_rate": avg([1.0 if r["refused"] else 0.0 for r in an]),
        "unanswerable_n": len(un),
        "answerable_n": len(an),
    }


def correctness_score(label: str | None) -> float | None:
    return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(label or "")
