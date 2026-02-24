# =============================================================================
# Initial Migration
# =============================================================================

"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升級數據庫"""
    
    # 創建用戶表
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('username', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=True),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('avatar_url', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('is_verified', sa.Boolean, default=False, nullable=False),
        sa.Column('is_superuser', sa.Boolean, default=False, nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    
    # 創建API密鑰表
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('hashed_key', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('permissions', sa.Text, nullable=True),
        sa.Column('rate_limit', sa.Integer, default=100, nullable=False),
        sa.Column('monthly_quota', sa.Integer, default=10000, nullable=False),
        sa.Column('monthly_used', sa.Integer, default=0, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    
    # 創建用戶提供者配置表
    op.create_table(
        'user_provider_configs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('provider_id', sa.String(50), nullable=False, index=True),
        sa.Column('encrypted_config', sa.Text, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('provider_name', sa.String(100), nullable=True),
        sa.Column('description', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    
    # 創建請求日誌表
    op.create_table(
        'request_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('request_id', sa.String(36), unique=True, nullable=False, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True, index=True),
        sa.Column('method', sa.String(10), nullable=False),
        sa.Column('path', sa.String(500), nullable=False, index=True),
        sa.Column('query_params', sa.Text, nullable=True),
        sa.Column('request_body', sa.Text, nullable=True),
        sa.Column('client_ip', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('status_code', sa.Integer, nullable=False, index=True),
        sa.Column('response_body', sa.Text, nullable=True),
        sa.Column('duration_ms', sa.Float, nullable=False, index=True),
        sa.Column('provider_id', sa.String(50), nullable=True, index=True),
        sa.Column('provider_duration_ms', sa.Float, nullable=True),
        sa.Column('provider_error', sa.Text, nullable=True),
        sa.Column('error_code', sa.String(50), nullable=True, index=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, index=True),
    )
    
    # 創建提供者調用日誌表
    op.create_table(
        'provider_call_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('request_id', sa.String(36), nullable=False, index=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True, index=True),
        sa.Column('provider_id', sa.String(50), nullable=False, index=True),
        sa.Column('capability', sa.String(50), nullable=False, index=True),
        sa.Column('operation', sa.String(100), nullable=False),
        sa.Column('request_payload', sa.Text, nullable=True),
        sa.Column('response_payload', sa.Text, nullable=True),
        sa.Column('success', sa.Integer, default=0, nullable=False),
        sa.Column('duration_ms', sa.Float, nullable=False),
        sa.Column('error_code', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, index=True),
    )
    
    # 創建複合索引
    op.create_index('ix_request_logs_user_created', 'request_logs', ['user_id', 'created_at'])
    op.create_index('ix_request_logs_path_created', 'request_logs', ['path', 'created_at'])
    op.create_index('ix_request_logs_provider_created', 'request_logs', ['provider_id', 'created_at'])
    op.create_index('ix_provider_logs_user_provider', 'provider_call_logs', ['user_id', 'provider_id'])
    op.create_index('ix_provider_logs_capability_created', 'provider_call_logs', ['capability', 'created_at'])


def downgrade() -> None:
    """降級數據庫"""
    op.drop_table('provider_call_logs')
    op.drop_table('request_logs')
    op.drop_table('user_provider_configs')
    op.drop_table('api_keys')
    op.drop_table('users')
