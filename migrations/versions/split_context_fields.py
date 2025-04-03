"""Split context field into code_context and doc_context

Revision ID: split_context_fields
Revises: previous_revision
Create Date: 2024-04-03 07:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'split_context_fields'
down_revision = 'previous_revision'  # Set this to your previous migration
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns
    op.add_column('nodes', sa.Column('code_context', sa.Text(), nullable=True))
    op.add_column('nodes', sa.Column('doc_context', sa.Text(), nullable=True))
    
    # Copy data from context to code_context
    op.execute("""
        UPDATE nodes 
        SET code_context = context,
            doc_context = ''
        WHERE context IS NOT NULL
    """)
    
    # Drop old column
    op.drop_column('nodes', 'context')

def downgrade():
    # Add back the context column
    op.add_column('nodes', sa.Column('context', sa.Text(), nullable=True))
    
    # Merge code_context back into context
    op.execute("""
        UPDATE nodes 
        SET context = code_context 
        WHERE code_context IS NOT NULL
    """)
    
    # Drop new columns
    op.drop_column('nodes', 'code_context')
    op.drop_column('nodes', 'doc_context') 