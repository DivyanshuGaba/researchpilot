from alembic import context
from sqlalchemy import create_engine

from app.config import get_settings
from app.models import Base

target_metadata = Base.metadata


def run_migrations_online() -> None:
    # Migrations run as the table owner (superuser); the app connects as rp_app so RLS applies.
    engine = create_engine(get_settings().migration_database_url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
