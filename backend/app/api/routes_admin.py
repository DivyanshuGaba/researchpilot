import uuid
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import chat_deps, current_user_id, settings_dep
from app.config import Settings
from app.db import get_factory
from app.models import User
from app.services.chat_service import Deps
from app.services.metrics import collect_metrics

router = APIRouter(tags=["admin"])


@lru_cache
def _metrics_factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(create_engine(url, pool_pre_ping=True))


def metrics_session(settings: Settings = Depends(settings_dep)) -> Iterator[Session]:
    """Read-only role that can see every tenant's aggregates. Never used for anything else."""
    with _metrics_factory(settings.metrics_database_url)() as session:
        yield session


def require_admin(
    user_id: uuid.UUID = Depends(current_user_id), settings: Settings = Depends(settings_dep)
) -> uuid.UUID:
    with get_factory()() as s:
        user = s.get(User, user_id)
    allowed = {e.lower() for e in settings.admin_emails}
    if user is None or user.email.lower() not in allowed:
        raise HTTPException(403, "admin only")
    return user_id


@router.get("/admin/metrics")
def admin_metrics(
    days: int = Query(default=7, ge=1, le=90),
    _admin: uuid.UUID = Depends(require_admin),
    session: Session = Depends(metrics_session),
) -> dict[str, Any]:
    return collect_metrics(session, days)


@router.get("/usage")
def my_usage(
    days: int = Query(default=30, ge=1, le=365),
    user_id: uuid.UUID = Depends(current_user_id),
    deps: Deps = Depends(chat_deps),
) -> dict[str, Any]:
    """The signed-in user's own cost and token use (row level security scopes every query)."""
    with deps.open_session(user_id) as s:
        window = "created_at > now() - make_interval(days => :d)"
        total = s.execute(
            text(f"""select count(*) q, coalesce(sum(cost_usd),0) c, coalesce(sum(tokens_in),0) tin,
                            coalesce(sum(tokens_out),0) tout from query_logs where {window}"""),
            {"d": days},
        ).one()
        convs = s.execute(
            text(f"""select q.conversation_id id, c.title, count(*) n, coalesce(sum(q.cost_usd),0) cost
                     from query_logs q join conversations c on c.id = q.conversation_id
                     where q.{window} group by 1, 2
                     order by cost desc limit 10"""),
            {"d": days},
        ).all()
    return {
        "window_days": days,
        "queries": int(total.q),
        "cost_usd": round(float(total.c), 6),
        "tokens": {"input": int(total.tin), "output": int(total.tout)},
        "by_conversation": [
            {"conversation_id": str(r.id), "title": r.title, "queries": int(r.n), "cost_usd": round(float(r.cost), 6)}
            for r in convs
        ],
    }
