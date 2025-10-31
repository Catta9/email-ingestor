"""Add status, notes and tags for contacts

Revision ID: a5c61d2db0df
Revises: 
Create Date: 2024-11-23 00:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a5c61d2db0df"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("contacts") as batch_op:
        batch_op.add_column(sa.Column("status", sa.String(), nullable=False, server_default="new"))
        batch_op.add_column(sa.Column("notes", sa.Text(), nullable=True))

    op.execute("UPDATE contacts SET status='new' WHERE status IS NULL")

    with op.batch_alter_table("contacts") as batch_op:
        batch_op.alter_column("status", server_default=None)

    op.create_table(
        "contact_tags",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("contact_id", sa.String(), sa.ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tag", sa.String(), nullable=False),
    )
    op.create_index("ix_contact_tags_contact_id", "contact_tags", ["contact_id"])


def downgrade() -> None:
    op.drop_index("ix_contact_tags_contact_id", table_name="contact_tags")
    op.drop_table("contact_tags")

    with op.batch_alter_table("contacts") as batch_op:
        batch_op.drop_column("notes")
        batch_op.drop_column("status")
