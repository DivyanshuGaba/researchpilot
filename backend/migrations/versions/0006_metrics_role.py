"""read-only role for cross-tenant aggregate metrics

Revision ID: 0006
Revises: 0005
"""
from alembic import op

revision = "0006"
down_revision = "0005"

TABLES = ["query_logs", "outbox", "documents", "document_versions", "ingestion_jobs", "users"]


def upgrade() -> None:
    # The API's own role (rp_app) is bound by row level security. The admin metrics endpoint needs
    # aggregates across tenants, so it uses a separate role that can only SELECT, never write.
    op.execute(
        """
        DO $$ BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rp_metrics') THEN
            CREATE ROLE rp_metrics LOGIN PASSWORD 'rp_metrics' NOSUPERUSER BYPASSRLS;
          END IF;
        END $$
        """
    )
    op.execute("GRANT USAGE ON SCHEMA public TO rp_metrics")
    for table in TABLES:
        op.execute(f"GRANT SELECT ON {table} TO rp_metrics")


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"REVOKE SELECT ON {table} FROM rp_metrics")
    op.execute("REVOKE USAGE ON SCHEMA public FROM rp_metrics")
    op.execute("DROP ROLE IF EXISTS rp_metrics")
