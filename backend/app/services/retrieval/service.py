"""Retrieval orchestration: dense (+ sparse) -> RRF -> rerank -> diversify -> confidence gate.

Split in two so the streaming path can announce the "reranking" stage between the steps.
Every optional stage fails soft (architecture.md section 13).
"""

import logging
import time
from dataclasses import dataclass, field

from app.config import PipelineConfig
from app.services.embeddings import Embedder
from app.services.retrieval.hybrid import diversify, rrf
from app.services.retrieval.reranker import Reranker
from app.services.vectorstore import Hit, Scope, VectorStore

log = logging.getLogger(__name__)


@dataclass
class Candidates:
    hits: list[Hit]
    timings_ms: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    hits: list[Hit]
    gated: bool = False
    top_score: float | None = None
    closest: list[Hit] = field(default_factory=list)  # shown to the user when gated
    timings_ms: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def gather_candidates(
    store: VectorStore,
    embedder: Embedder,
    query: str,
    scope: Scope,
    cfg: PipelineConfig,
) -> Candidates:
    out = Candidates(hits=[])
    t = time.perf_counter()
    vector = embedder.embed_query(query)
    out.timings_ms["embed"] = _ms(t)

    t = time.perf_counter()
    dense = store.search(vector, cfg.dense_k, scope)
    out.timings_ms["dense"] = _ms(t)
    if cfg.retrieval_mode == "vector":
        out.hits = dense
        return out

    t = time.perf_counter()
    try:
        sparse = store.keyword_search(query, cfg.sparse_k, scope)
    except Exception:
        log.exception("keyword search failed")
        sparse = []
        out.degraded.append("sparse")
    out.timings_ms["sparse"] = _ms(t)
    out.hits = rrf([dense, sparse], cfg.rrf_k)
    return out


def finalize(
    cands: Candidates,
    query: str,
    reranker: Reranker | None,
    cfg: PipelineConfig,
    min_score: float,
) -> Retrieval:
    result = Retrieval(
        hits=[], timings_ms=dict(cands.timings_ms), degraded=list(cands.degraded)
    )
    ranked = cands.hits
    reranked = False
    if cfg.retrieval_mode == "hybrid_rerank" and reranker is not None and ranked:
        t = time.perf_counter()
        try:
            ranked = reranker.rerank(query, ranked[: cfg.rerank_candidates], cfg.rerank_candidates)
            reranked = True
        except Exception:
            log.exception("rerank failed")
            result.degraded.append("rerank")
        result.timings_ms["rerank"] = _ms(t)
    elif cfg.retrieval_mode == "hybrid_rerank" and reranker is None:
        result.degraded.append("rerank")

    ranked = diversify(ranked, cfg.max_per_parent)[: cfg.top_k]
    result.top_score = ranked[0].score if ranked else None
    # The gate only makes sense on reranker scores; fused/cosine scores are not comparable.
    if reranked and (not ranked or ranked[0].score < min_score):
        result.gated = True
        result.closest = ranked[:3]
        return result
    result.hits = ranked
    return result
