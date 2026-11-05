"""quizzes, flashcards, summaries; messages.tool_calls

Revision ID: 0004
Revises: 0003
"""
from alembic import op

from app.models import Base

revision = "0004"
down_revision = "0003"

NEW_TABLES = ["quizzes", "flashcards", "summaries"]


def upgrade() -> None:
    op.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_calls jsonb NOT NULL DEFAULT '[]'")
    Base.metadata.create_all(
        op.get_bind(), tables=[Base.metadata.tables[t] for t in NEW_TABLES], checkfirst=True
    )
    # One cached summary per section per version, and one per version for the whole document.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_summaries_section ON summaries (version_id, chunk_id) "
        "WHERE kind = 'section'"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_summaries_document ON summaries (version_id) "
        "WHERE kind = 'document'"
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rp_app")
    for table in NEW_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid) "
            f"WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)"
        )


def downgrade() -> None:
    op.drop_column("messages", "tool_calls")
    for table in reversed(NEW_TABLES):
        op.drop_table(table)
