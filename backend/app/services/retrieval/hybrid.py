"""Fusion and diversification, kept as pure functions so they are trivial to unit test."""

import dataclasses
from collections import defaultdict

from app.services.vectorstore import Hit


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: needs no score calibration between the dense and sparse lists."""
    scores: dict[object, float] = defaultdict(float)
    first: dict[object, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.key] += 1.0 / (k + rank)
            first.setdefault(hit.key, hit)
    fused = [dataclasses.replace(h, score=scores[key]) for key, h in first.items()]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused


def diversify(hits: list[Hit], max_per_parent: int) -> list[Hit]:
    """Cap hits per parent so one section cannot crowd out everything else."""
    counts: dict[object, int] = defaultdict(int)
    out: list[Hit] = []
    for h in hits:
        if counts[h.chunk_id] < max_per_parent:
            counts[h.chunk_id] += 1
            out.append(h)
    return out
