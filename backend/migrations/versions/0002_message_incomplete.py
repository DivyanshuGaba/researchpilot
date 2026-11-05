"""messages.incomplete for partially streamed answers

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    # IF NOT EXISTS: 0001 builds tables from the live models, so a fresh install may already
    # have this column.
    op.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS incomplete boolean NOT NULL DEFAULT false")


def downgrade() -> None:
    op.drop_column("messages", "incomplete")
