from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine = None
_factory: sessionmaker[Session] | None = None


def get_factory() -> sessionmaker[Session]:
    global _engine, _factory
    if _factory is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True)
        _factory = sessionmaker(_engine, expire_on_commit=False)
    return _factory


@contextmanager
def tenant_session(user_id: UUID) -> Iterator[Session]:
    """Session whose transaction sets app.user_id so Postgres RLS enforces tenancy."""
    with get_factory()() as session, session.begin():
        session.execute(text("SELECT set_config('app.user_id', :u, true)"), {"u": str(user_id)})
        yield session
