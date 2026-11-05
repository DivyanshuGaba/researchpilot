import dataclasses
import math
from typing import Any, Protocol

from app.config import Settings
from app.services.vectorstore import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


class CrossEncoderReranker:
    """ms-marco cross-encoder. predict() returns raw logits in this sentence-transformers
    version (measured: relevant about -1 to 8, irrelevant about -11), so we apply the sigmoid
    ourselves to get stable 0..1 scores. They separate relevant from irrelevant far better than
    cosine similarity, which is why the confidence gate uses them (architecture.md 5.4)."""

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_id = model_id
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._load().predict([(query, h.child_text) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda p: float(p[1]), reverse=True)
        return [dataclasses.replace(h, score=sigmoid(float(s))) for h, s in ranked[:top_n]]


_cached: Reranker | None = None


def get_reranker(settings: Settings) -> Reranker | None:
    global _cached
    if settings.reranker == "none":
        return None
    if _cached is None:
        if settings.reranker != "cross_encoder":
            raise ValueError(f"unknown reranker {settings.reranker!r}")
        _cached = CrossEncoderReranker()
    return _cached
