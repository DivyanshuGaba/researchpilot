"""initial schema, indexes, row level security

Revision ID: 0001
Revises:
"""
from alembic import op

from app.models import Base

revision = "0001"
down_revision = None

TENANT_TABLES = [
    "documents",
    "document_versions",
    "ingestion_jobs",
    "chunks",
    "conversations",
    "messages",
]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(op.get_bind())

    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    # Keep tsv in sync with content without app code having to remember.
    op.execute(
        """
        CREATE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
          NEW.tsv := to_tsvector('english', coalesce(NEW.content, ''));
          RETURN NEW;
        END $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER chunks_tsv BEFORE INSERT OR UPDATE OF content ON chunks "
        "FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update()"
    )

    # Non-owner, non-superuser app role so RLS is actually enforced.
    op.execute(
        """
        DO $$ BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rp_app') THEN
            CREATE ROLE rp_app LOGIN PASSWORD 'rp_app' NOSUPERUSER NOBYPASSRLS;
          END IF;
        END $$
        """
    )
    op.execute("GRANT USAGE ON SCHEMA public TO rp_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rp_app")

    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid) "
            f"WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)"
        )


def downgrade() -> None:
    Base.metadata.drop_all(op.get_bind())
    op.execute("DROP FUNCTION IF EXISTS chunks_tsv_update()")
