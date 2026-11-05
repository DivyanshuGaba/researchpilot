from typing import Any, Protocol

from app.config import Settings


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """sentence-transformers model, vectors normalized so cosine equals inner product."""

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_id: str, dim: int) -> None:
        self.model_id = model_id
        self.dim = dim
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._load().encode(texts, batch_size=32, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._load().encode([self._QUERY_PREFIX + text], normalize_embeddings=True)[0]
        return vec.tolist()  # type: ignore[no-any-return]


_cached: Embedder | None = None


def get_embedder(settings: Settings) -> Embedder:
    global _cached
    if _cached is None:
        if settings.embedding_provider != "local":
            raise ValueError(f"unknown embedding provider {settings.embedding_provider!r}")
        _cached = LocalEmbedder(settings.embedding_model, settings.embedding_dim)
    return _cached
