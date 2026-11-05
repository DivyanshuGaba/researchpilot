"""rolling conversation summary

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.add_column("conversations", sa.Column("summary", sa.Text(), server_default="", nullable=False))
    op.add_column(
        "conversations", sa.Column("summarized_count", sa.Integer(), server_default="0", nullable=False)
    )


def downgrade() -> None:
    op.drop_column("conversations", "summarized_count")
    op.drop_column("conversations", "summary")
