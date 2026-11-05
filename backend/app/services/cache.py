"""Caching layers (architecture.md section 7).

1. Prompt cache: the static system prompt is marked cacheable in the LLM client.
2. Embedding cache: hash(model id + text) -> vector.
3. Exact response cache: hash(user, document-set version, standalone query, config, mode).
4. Semantic cache: reuse an answer when the query embedding is at least 0.95 similar, scoped to
   the same user, document-set version and config.

Correctness rules: every key contains the document-set version hash, so adding, deleting or
re-uploading a document invalidates stale answers; entries are never shared across users;
agent answers are never cached; degraded answers are never cached. If Redis is down every
operation is a no-op and the API stays up.
"""

import base64
import hashlib
import json
import logging
import re
import time
import uuid
from array import array
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentVersion
from app.services.embeddings import Embedder

log = logging.getLogger(__name__)
SEMANTIC_MAX_ENTRIES = 50
EMBEDDING_TTL = 7 * 86400
OUTAGE_BACKOFF_S = 30.0


def _h(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()


def encode_vec(vec: list[float]) -> str:
    return base64.b64encode(array("f", vec).tobytes()).decode()


def decode_vec(b64: str) -> list[float]:
    a = array("f")
    a.frombytes(base64.b64decode(b64))
    return list(a)


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.! ")


def docset_hash(session: Session, user_id: uuid.UUID, document_ids: tuple[uuid.UUID, ...] | None) -> str:
    """Identity of the indexed content a question will be answered from. Changes whenever a
    document in scope becomes ready, is re-uploaded (new version) or is removed."""
    stmt = (
        select(DocumentVersion.id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.user_id == user_id, Document.status == "ready", DocumentVersion.status == "ready")
        .order_by(DocumentVersion.id)
    )
    if document_ids:
        stmt = stmt.where(Document.id.in_(document_ids))
    ids = [str(i) for i in session.scalars(stmt)]
    return _h(*ids)[:16] if ids else "empty"


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 3600, enabled: bool = True) -> None:
        self._url, self.ttl, self.enabled = url, ttl_seconds, enabled
        self._r: Any = None
        self._down_until = 0.0

    def _client(self) -> Any:
        if self._r is None:
            import redis

            self._r = redis.Redis.from_url(self._url, socket_timeout=0.3, socket_connect_timeout=0.3)
        return self._r

    def _call(self, fn: Any, default: Any = None) -> Any:
        if not self.enabled or time.monotonic() < self._down_until:
            return default
        try:
            return fn(self._client())
        except Exception:  # noqa: BLE001 - Redis down: bypass the cache, keep serving
            self._down_until = time.monotonic() + OUTAGE_BACKOFF_S
            log.warning("cache: Redis unavailable, bypassing the cache for %ds", OUTAGE_BACKOFF_S)
            return default

    # embeddings
    def get_embeddings(self, keys: list[str]) -> list[list[float] | None]:
        raw = self._call(lambda r: r.mget(keys), [None] * len(keys))
        return [decode_vec(v.decode()) if v else None for v in raw]

    def set_embeddings(self, items: dict[str, list[float]]) -> None:
        def run(r: Any) -> None:
            pipe = r.pipeline()
            for k, v in items.items():
                pipe.set(k, encode_vec(v), ex=EMBEDDING_TTL)
            pipe.execute()

        self._call(run)

    # exact
    def exact_key(self, user_id: uuid.UUID, docset: str, cfg: str, mode: str, query: str) -> str:
        return "rp:exact:" + _h(user_id, docset, cfg, mode, normalize_query(query))

    def get_exact(self, key: str) -> dict[str, Any] | None:
        raw = self._call(lambda r: r.get(key))
        return json.loads(raw) if raw else None

    def set_exact(self, key: str, payload: dict[str, Any]) -> None:
        self._call(lambda r: r.set(key, json.dumps(payload), ex=self.ttl))

    # semantic
    def semantic_key(self, user_id: uuid.UUID, docset: str, cfg: str, mode: str) -> str:
        return "rp:sem:" + _h(user_id, docset, cfg, mode)

    def semantic_entries(self, key: str) -> list[tuple[list[float], dict[str, Any]]]:
        raw = self._call(lambda r: r.lrange(key, 0, SEMANTIC_MAX_ENTRIES - 1), [])
        out = []
        for item in raw:
            d = json.loads(item)
            out.append((decode_vec(d["v"]), d["p"]))
        return out

    def semantic_add(self, key: str, vec: list[float], payload: dict[str, Any]) -> None:
        def run(r: Any) -> None:
            r.lpush(key, json.dumps({"v": encode_vec(vec), "p": payload}))
            r.ltrim(key, 0, SEMANTIC_MAX_ENTRIES - 1)
            r.expire(key, self.ttl)

        self._call(run)

    def lookup(
        self, key_exact: str, key_sem: str, query_vec: list[float], threshold: float
    ) -> tuple[str, dict[str, Any], float] | None:
        """Returns (layer, payload, similarity) for the best cache hit, exact first."""
        hit = self.get_exact(key_exact)
        if hit is not None:
            return "exact", hit, 1.0
        best: tuple[float, dict[str, Any]] | None = None
        for vec, payload in self.semantic_entries(key_sem):
            sim = sum(a * b for a, b in zip(query_vec, vec, strict=False))  # vectors are normalized
            if sim >= threshold and (best is None or sim > best[0]):
                best = (sim, payload)
        return ("semantic", best[1], best[0]) if best else None

    def store(
        self, key_exact: str, key_sem: str, query_vec: list[float], payload: dict[str, Any]
    ) -> None:
        self.set_exact(key_exact, payload)
        self.semantic_add(key_sem, query_vec, payload)


class CachedEmbedder:
    """Embedder wrapper with a Redis cache, keyed by model id and text (architecture 7.2)."""

    def __init__(self, inner: Embedder, cache: RedisCache) -> None:
        self._inner, self._cache = inner, cache
        self.model_id, self.dim = inner.model_id, inner.dim

    def _key(self, text: str) -> str:
        return "rp:emb:" + _h(self.model_id, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        cached = self._cache.get_embeddings(keys) if texts else []
        missing = [i for i, v in enumerate(cached) if v is None]
        if missing:
            fresh = self._inner.embed_documents([texts[i] for i in missing])
            for i, vec in zip(missing, fresh, strict=True):
                cached[i] = vec
            self._cache.set_embeddings({keys[i]: v for i, v in zip(missing, fresh, strict=True)})
        return [v for v in cached if v is not None]

    def embed_query(self, text: str) -> list[float]:
        key = "rp:embq:" + _h(self.model_id, text)  # queries use a different prefix than passages
        cached = self._cache.get_embeddings([key])[0]
        if cached is not None:
            return cached
        vec = self._inner.embed_query(text)
        self._cache.set_embeddings({key: vec})
        return vec
