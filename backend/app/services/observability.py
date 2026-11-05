"""Per-request observability: query logs in Postgres, cost accounting, optional Langfuse traces.

Everything here is best effort: a logging failure must never fail or slow an answer.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import PipelineConfig, Settings
from app.models import QueryLog
from app.services.guardrails import mask_pii

log = logging.getLogger(__name__)


def config_hash(cfg: PipelineConfig, **extra: object) -> str:
    """Stable id of a pipeline configuration. Eval runs and query logs both carry it, so any two
    results can be compared like for like."""
    payload = json.dumps({**cfg.model_dump(), **extra}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class QueryRecord:
    user_id: uuid.UUID
    mode: str
    config_hash: str
    message_id: uuid.UUID | None = None
    conversation_id: uuid.UUID | None = None
    rewritten_query: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    top_score: float | None = None
    gated: bool = False
    cache_status: str = "miss"
    latency_ms: int = 0
    latency_breakdown: dict[str, int] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    degraded: list[str] = field(default_factory=list)
    error: str | None = None


SessionOpener = Callable[[uuid.UUID], AbstractContextManager[Session]]


def write_query_log(open_session: SessionOpener, rec: QueryRecord, mask: bool = True) -> None:
    """Persist one query record. The rewritten query is user text, so it is PII-masked."""
    query = mask_pii(rec.rewritten_query) if mask else rec.rewritten_query
    try:
        with open_session(rec.user_id) as s:
            s.add(
                QueryLog(
                    user_id=rec.user_id, message_id=rec.message_id,
                    conversation_id=rec.conversation_id, mode=rec.mode,
                    config_hash=rec.config_hash, rewritten_query=query[:1000],
                    retrieved_chunk_ids=rec.retrieved_chunk_ids, top_score=rec.top_score,
                    gated=rec.gated, cache_status=rec.cache_status, latency_ms=rec.latency_ms,
                    latency_breakdown=rec.latency_breakdown, tokens_in=rec.tokens_in,
                    tokens_out=rec.tokens_out, cache_read_tokens=rec.cache_read_tokens,
                    cost_usd=round(rec.cost_usd, 6), degraded=rec.degraded, error=rec.error,
                )
            )
    except Exception:
        log.exception("could not write query log")


# --- Langfuse (optional) -----------------------------------------------------------------------


class Tracer:
    """Sends one trace per request to Langfuse's public ingestion API when keys are configured.

    Inputs and outputs are PII-masked and truncated first: no document text, no keys. Failures
    are swallowed. Not verified against a live Langfuse project (no account in development)."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(
            settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host
        )
        self._settings = settings
        self._tasks: set[asyncio.Task[None]] = set()

    def trace(self, rec: QueryRecord, question: str, answer: str, model: str) -> None:
        if not self.enabled:
            return
        task = asyncio.create_task(self._send(rec, question, answer, model))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def build_batch(self, rec: QueryRecord, question: str, answer: str, model: str) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        trace_id = str(uuid.uuid4())
        clip = lambda t: mask_pii(t)[:2000]
        events: list[dict[str, Any]] = [
            {
                "id": str(uuid.uuid4()), "type": "trace-create", "timestamp": now,
                "body": {
                    "id": trace_id, "name": f"chat-{rec.mode}", "userId": mask_pii(str(rec.user_id)),
                    "input": clip(question), "output": clip(answer),
                    "metadata": {"config_hash": rec.config_hash, "cache": rec.cache_status,
                                 "gated": rec.gated, "degraded": rec.degraded},
                },
            }
        ]
        for stage, ms in rec.latency_breakdown.items():
            events.append(
                {
                    "id": str(uuid.uuid4()), "type": "span-create", "timestamp": now,
                    "body": {"id": str(uuid.uuid4()), "traceId": trace_id, "name": stage,
                             "metadata": {"duration_ms": ms}},
                }
            )
        events.append(
            {
                "id": str(uuid.uuid4()), "type": "generation-create", "timestamp": now,
                "body": {
                    "id": str(uuid.uuid4()), "traceId": trace_id, "name": "answer", "model": model,
                    "usage": {"input": rec.tokens_in, "output": rec.tokens_out, "unit": "TOKENS",
                              "totalCost": rec.cost_usd},
                },
            }
        )
        return {"batch": events}

    async def _send(self, rec: QueryRecord, question: str, answer: str, model: str) -> None:
        s = self._settings
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{s.langfuse_host.rstrip('/')}/api/public/ingestion",
                    json=self.build_batch(rec, question, answer, model),
                    auth=(s.langfuse_public_key, s.langfuse_secret_key),
                )
        except Exception:  # noqa: BLE001
            log.warning("langfuse export failed (ignored)")
