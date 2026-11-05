"""Outbox pattern for ingestion jobs (architecture.md section 9).

The upload handler writes an outbox row in the same transaction as the document, so a file is
never saved without a job to process it. A dispatcher publishes pending rows to Celery: right
after commit for low latency, and from a background loop so a crash or a broker outage between
the two is repaired. Delivery is at least once; ingestion is idempotent per version.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import OutboxEvent

log = logging.getLogger(__name__)
INGEST = "ingest"
MAX_ATTEMPTS = 10


def add_ingest_event(session: Session, version_id: uuid.UUID, user_id: uuid.UUID) -> None:
    session.add(
        OutboxEvent(kind=INGEST, payload={"version_id": str(version_id), "user_id": str(user_id)})
    )


def dispatch_pending(
    factory: sessionmaker[Session], enqueue: Callable[[str, str], None], batch: int = 20
) -> int:
    """Publish pending events. SKIP LOCKED lets several dispatchers run without double sends."""
    sent = 0
    with factory() as session, session.begin():
        rows = session.scalars(
            select(OutboxEvent)
            .where(OutboxEvent.dispatched_at.is_(None), OutboxEvent.attempts < MAX_ATTEMPTS)
            .order_by(OutboxEvent.created_at)
            .limit(batch)
            .with_for_update(skip_locked=True)
        ).all()
        for row in rows:
            row.attempts += 1
            try:
                if row.kind == INGEST:
                    enqueue(row.payload["version_id"], row.payload["user_id"])
                row.dispatched_at = datetime.now(UTC)
                sent += 1
            except Exception:
                log.exception("outbox dispatch failed (attempt %d)", row.attempts)
    return sent


async def dispatcher_loop(
    factory: sessionmaker[Session], enqueue: Callable[[str, str], None], interval: float = 5.0
) -> None:
    while True:
        try:
            await asyncio.to_thread(dispatch_pending, factory, enqueue)
        except Exception:
            log.exception("outbox loop error")
        await asyncio.sleep(interval)
