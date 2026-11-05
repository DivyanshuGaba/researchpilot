"""Evaluation runtime: indexes the corpus under any chunking config and retrieves through the
same service layer the API uses (not over HTTP), so results describe the real pipeline."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from app.config import PipelineConfig, Settings
from app.db import get_factory, tenant_session
from app.models import Document, DocumentVersion, User
from app.services.embeddings import Embedder, get_embedder
from app.services.ingestion.pipeline import ingest_version
from app.services.observability import config_hash
from app.services.retrieval.reranker import Reranker, get_reranker
from app.services.retrieval.service import finalize, gather_candidates
from app.services.vectorstore import Hit, PgVectorStore, Scope

ROOT = Path(__file__).parent
EVAL_EMAIL = "eval@researchpilot.local"
CHUNK_FIELDS = ("chunker", "child_tokens", "parent_tokens", "overlap_tokens", "semantic_percentile")


def build_config(chunker: str, size: int, mode: str, **extra: object) -> PipelineConfig:
    """`size` is the section (parent) size sent to the model; children, which are embedded and
    searched, are a quarter of it. Overlap is a sixth of a child."""
    child = max(50, size // 4)
    return PipelineConfig(
        chunker=chunker, parent_tokens=size, child_tokens=child, overlap_tokens=child // 6,
        retrieval_mode=mode, **extra,  # type: ignore[arg-type]
    )


def chunking_hash(cfg: PipelineConfig, settings: Settings) -> str:
    only_chunking = PipelineConfig(**{f: getattr(cfg, f) for f in CHUNK_FIELDS})
    return config_hash(only_chunking, embed=settings.embedding_model)


@dataclass
class Retrieved:
    ranked: list[Hit]  # up to 10 children, best first, no confidence gate applied
    gated: bool
    top_score: float | None
    timings_ms: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


class EvalRuntime:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, reranker: Reranker | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.reranker = reranker if reranker is not None else get_reranker(settings)
        self.manifest = json.loads((ROOT / "docs.json").read_text())
        self._registry_path = ROOT / ".index.json"
        self._user: uuid.UUID | None = None

    @property
    def user_id(self) -> uuid.UUID:
        if self._user is None:
            with get_factory()() as s, s.begin():
                u = s.scalar(select(User).where(User.email == EVAL_EMAIL))
                if u is None:
                    u = User(email=EVAL_EMAIL)
                    s.add(u)
                    s.flush()
                self._user = u.id
        return self._user

    def _registry(self) -> dict[str, dict[str, str]]:
        return json.loads(self._registry_path.read_text()) if self._registry_path.exists() else {}

    def index(self, cfg: PipelineConfig) -> dict[str, uuid.UUID]:
        """doc key -> document id, ingesting the corpus under this chunking config if needed."""
        key = chunking_hash(cfg, self.settings)
        reg = self._registry()
        known = reg.get(key, {})
        if known and self._all_ready([uuid.UUID(v) for v in known.values()]):
            return {k: uuid.UUID(v) for k, v in known.items()}
        out: dict[str, uuid.UUID] = {}
        for d in self.manifest:
            path = ROOT / "docs" / f"{d['key']}.pdf"
            with tenant_session(self.user_id) as s:
                doc = Document(
                    user_id=self.user_id, title=d["title"], source_type="pdf", file_path=str(path)
                )
                s.add(doc)
                s.flush()
                ver = DocumentVersion(document_id=doc.id, user_id=self.user_id, file_hash=d["sha256"])
                s.add(ver)
                s.flush()
                ingest_version(s, ver.id, self.embedder, cfg)
                if doc.status != "ready":
                    raise RuntimeError(f"ingestion of {d['key']} failed under {key}")
                out[d["key"]] = doc.id
        reg[key] = {k: str(v) for k, v in out.items()}
        self._registry_path.write_text(json.dumps(reg, indent=1))
        return out

    def _all_ready(self, ids: list[uuid.UUID]) -> bool:
        with tenant_session(self.user_id) as s:
            rows = s.scalars(select(Document).where(Document.id.in_(ids))).all()
            return len(rows) == len(ids) and all(r.status == "ready" for r in rows)

    def retrieve(self, cfg: PipelineConfig, query: str, doc_ids: tuple[uuid.UUID, ...]) -> Retrieved:
        scope = Scope(self.user_id, doc_ids)
        with tenant_session(self.user_id) as s:
            cands = gather_candidates(PgVectorStore(s), self.embedder, query, scope, cfg)
        wide = cfg.model_copy(update={"top_k": 10})
        res = finalize(cands, query, self.reranker, wide, min_score=0.0)
        reranked = cfg.retrieval_mode == "hybrid_rerank" and "rerank" not in res.degraded
        top = res.hits[0].score if res.hits else None
        gated = reranked and (top is None or top < self.settings.retrieval_min_score)
        return Retrieved(res.hits, gated, top, res.timings_ms, res.degraded)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
