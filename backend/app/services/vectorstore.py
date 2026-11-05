import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import Chunk, Document


@dataclass(frozen=True)
class Scope:
    """Always carries a tenant, so a store cannot be called without one."""

    user_id: uuid.UUID
    document_ids: tuple[uuid.UUID, ...] | None = None


@dataclass
class Hit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int
    score: float
    child_text: str
    parent_text: str
    bbox_list: list[Any] = field(default_factory=list)


class VectorStore(Protocol):
    def search(self, vector: list[float], k: int, scope: Scope) -> list[Hit]: ...
    def delete_version(self, version_id: uuid.UUID) -> None: ...


class PgVectorStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search(self, vector: list[float], k: int, scope: Scope) -> list[Hit]:
        # Iterative scans keep walking the HNSW graph until enough rows pass the filter.
        self.session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        self.session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        dist = Chunk.embedding.cosine_distance(vector).label("dist")
        stmt = (
            select(Chunk, Document.title, dist)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.level == "child", Chunk.user_id == scope.user_id)
            .order_by(dist)
            .limit(k)
        )
        if scope.document_ids:
            stmt = stmt.where(Chunk.document_id.in_(scope.document_ids))
        rows = self.session.execute(stmt).all()

        parent_ids = {c.parent_id for c, _, _ in rows if c.parent_id}
        parents = {
            p.id: p.content
            for p in self.session.scalars(select(Chunk).where(Chunk.id.in_(parent_ids)))
        }
        return [
            Hit(
                chunk_id=c.parent_id or c.id,
                document_id=c.document_id,
                document_title=title,
                page_start=c.page_start,
                page_end=c.page_end,
                score=1.0 - float(d),
                child_text=c.content,
                parent_text=parents.get(c.parent_id, c.content) if c.parent_id else c.content,
                bbox_list=list(c.bbox_list or []),
            )
            for c, title, d in rows
        ]

    def delete_version(self, version_id: uuid.UUID) -> None:
        self.session.execute(text("DELETE FROM chunks WHERE version_id = :v"), {"v": version_id})
