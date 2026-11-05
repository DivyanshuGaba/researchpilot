"""rolling conversation summary

Revision ID: 0003
Revises: 0002
"""
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary text NOT NULL DEFAULT ''")
    op.execute(
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summarized_count integer NOT NULL DEFAULT 0"
    )


def downgrade() -> None:
    op.drop_column("conversations", "summarized_count")
    op.drop_column("conversations", "summary")
