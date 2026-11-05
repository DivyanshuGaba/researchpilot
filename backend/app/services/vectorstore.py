import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models import Chunk, Document


@dataclass(frozen=True)
class Scope:
    """Always carries a tenant, so a store cannot be called without one."""

    user_id: uuid.UUID
    document_ids: tuple[uuid.UUID, ...] | None = None


@dataclass
class Hit:
    chunk_id: uuid.UUID  # the parent chunk shown to the model (or the chunk itself)
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int
    score: float
    child_text: str
    parent_text: str
    bbox_list: list[Any] = field(default_factory=list)
    child_id: uuid.UUID | None = None
    flagged: bool = False  # the passage the model will read looks like an injection attempt

    @property
    def key(self) -> uuid.UUID:
        """Identity of the matched child, used for fusion and dedupe."""
        return self.child_id or self.chunk_id


class VectorStore(Protocol):
    def search(self, vector: list[float], k: int, scope: Scope) -> list[Hit]: ...
    def keyword_search(self, query: str, k: int, scope: Scope) -> list[Hit]: ...
    def delete_version(self, version_id: uuid.UUID) -> None: ...


_WORD = re.compile(r"[a-z0-9]+")


def or_tsquery(query: str, max_terms: int = 32) -> str:
    """Natural-language question -> OR query. websearch/plainto would AND every word, which
    returns nothing for a full sentence; the english config drops stopwords for us."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(query.lower()):
        if len(w) > 1:
            seen.setdefault(w)
    return " | ".join(list(seen)[:max_terms])


class PgVectorStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _to_hits(self, rows: list[Any]) -> list[Hit]:
        parent_ids = {c.parent_id for c, _, _ in rows if c.parent_id}
        parent_rows = (
            {p.id: p for p in self.session.scalars(select(Chunk).where(Chunk.id.in_(parent_ids)))}
            if parent_ids
            else {}
        )
        parents = {pid: p.content for pid, p in parent_rows.items()}
        parent_flag = {pid: bool((p.meta or {}).get("injection_flag")) for pid, p in parent_rows.items()}
        return [
            Hit(
                chunk_id=c.parent_id or c.id,
                child_id=c.id,
                document_id=c.document_id,
                document_title=title,
                page_start=c.page_start,
                page_end=c.page_end,
                score=float(score),
                child_text=c.content,
                parent_text=parents.get(c.parent_id, c.content) if c.parent_id else c.content,
                bbox_list=list(c.bbox_list or []),
                flagged=parent_flag.get(c.parent_id, bool((c.meta or {}).get("injection_flag")))
                if c.parent_id
                else bool((c.meta or {}).get("injection_flag")),
            )
            for c, title, score in rows
        ]

    def search(self, vector: list[float], k: int, scope: Scope) -> list[Hit]:
        # Iterative scans keep walking the HNSW graph until enough rows pass the filter.
        self.session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        self.session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        dist = Chunk.embedding.cosine_distance(vector).label("dist")
        stmt = (
            select(Chunk, Document.title, (1 - dist).label("score"))
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.level == "child", Chunk.user_id == scope.user_id)
            .order_by(dist)
            .limit(k)
        )
        if scope.document_ids:
            stmt = stmt.where(Chunk.document_id.in_(scope.document_ids))
        return self._to_hits(list(self.session.execute(stmt).all()))

    def keyword_search(self, query: str, k: int, scope: Scope) -> list[Hit]:
        terms = or_tsquery(query)
        if not terms:
            return []
        tsq = func.to_tsquery("english", terms)
        rank = func.ts_rank_cd(Chunk.tsv, tsq).label("rank")
        stmt = (
            select(Chunk, Document.title, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.level == "child",
                Chunk.user_id == scope.user_id,
                Chunk.tsv.op("@@")(tsq),
            )
            .order_by(rank.desc())
            .limit(k)
        )
        if scope.document_ids:
            stmt = stmt.where(Chunk.document_id.in_(scope.document_ids))
        return self._to_hits(list(self.session.execute(stmt).all()))

    def delete_version(self, version_id: uuid.UUID) -> None:
        self.session.execute(text("DELETE FROM chunks WHERE version_id = :v"), {"v": version_id})
