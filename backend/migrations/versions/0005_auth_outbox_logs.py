"""refresh tokens, outbox, query logs

Revision ID: 0005
Revises: 0004
"""
from alembic import op

from app.models import Base

revision = "0005"
down_revision = "0004"

NEW_TABLES = ["refresh_tokens", "outbox", "query_logs"]


def upgrade() -> None:
    Base.metadata.create_all(
        op.get_bind(), tables=[Base.metadata.tables[t] for t in NEW_TABLES], checkfirst=True
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_outbox_pending ON outbox (created_at) WHERE dispatched_at IS NULL")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rp_app")
    # refresh_tokens and outbox are system tables (looked up before a tenant is known), like
    # users; query_logs is per tenant.
    op.execute("ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON query_logs "
        "USING (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid) "
        "WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)"
    )


def downgrade() -> None:
    for table in reversed(NEW_TABLES):
        op.drop_table(table)
