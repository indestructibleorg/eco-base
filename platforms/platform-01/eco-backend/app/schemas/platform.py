# =============================================================================
# Platform Integration Schemas
# =============================================================================

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

from app.schemas.base import BaseSchema, ResponseSchema


# =============================================================================
# Provider Schemas
# =============================================================================

class ProviderInfoSchema(BaseSchema):
    """提供者信息Schema"""
    provider_id: str
    domain: str
    capabilities: List[str]
    is_available: bool


class ProviderListResponse(BaseSchema):
    """提供者列表響應Schema"""
    items: List[ProviderInfoSchema]
    total: int


class ProviderHealthSchema(BaseSchema):
    """提供者健康狀態Schema"""
    provider_id: str
    healthy: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    checked_at: str


# =============================================================================
# Cognitive Compute Schemas
# =============================================================================

class GenerateTextRequest(BaseSchema):
    """文本生成請求Schema"""
    prompt: str = Field(..., min_length=1, max_length=10000)
    context: Optional[List[Dict[str, str]]] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    provider: Optional[str] = "gamma-cognitive"
    stream: bool = False


class GenerateTextResponse(BaseSchema):
    """文本生成響應Schema"""
    content: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    provider: str


class EmbedTextsRequest(BaseSchema):
    """文本嵌入請求Schema"""
    texts: List[str] = Field(..., min_length=1, max_length=100)
    provider: Optional[str] = "gamma-cognitive"


class EmbedTextsResponse(BaseSchema):
    """文本嵌入響應Schema"""
    embeddings: List[List[float]]
    model: Optional[str] = None
    provider: str


class FunctionCallRequest(BaseSchema):
    """函數調用請求Schema"""
    prompt: str
    functions: List[Dict[str, Any]]
    provider: Optional[str] = "gamma-cognitive"


class FunctionCallResponse(BaseSchema):
    """函數調用響應Schema"""
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    provider: str


# =============================================================================
# Data Persistence Schemas
# =============================================================================

class QueryDataRequest(BaseSchema):
    """數據查詢請求Schema"""
    table: str = Field(..., min_length=1, max_length=100)
    filters: Optional[Dict[str, Any]] = None
    ordering: Optional[List[str]] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    provider: Optional[str] = "alpha-persistence"


class QueryDataResponse(BaseSchema):
    """數據查詢響應Schema"""
    data: List[Dict[str, Any]]
    total: Optional[int] = None
    provider: str


class MutateDataRequest(BaseSchema):
    """數據變更請求Schema"""
    operation: str = Field(..., pattern="^(insert|update|delete|upsert)$")
    table: str = Field(..., min_length=1, max_length=100)
    data: Dict[str, Any]
    conditions: Optional[Dict[str, Any]] = None
    provider: Optional[str] = "alpha-persistence"


class MutateDataResponse(BaseSchema):
    """數據變更響應Schema"""
    affected_rows: int
    data: Optional[Any] = None
    provider: str


class VectorSearchRequest(BaseSchema):
    """向量搜索請求Schema"""
    table: str = Field(..., min_length=1, max_length=100)
    vector: List[float]
    top_k: int = Field(default=10, ge=1, le=100)
    provider: Optional[str] = "alpha-persistence"


class VectorSearchResponse(BaseSchema):
    """向量搜索響應Schema"""
    results: List[Dict[str, Any]]
    provider: str


# =============================================================================
# Code Engineering Schemas
# =============================================================================

class CompleteCodeRequest(BaseSchema):
    """代碼補全請求Schema"""
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(..., min_length=1, max_length=50)
    cursor_position: Optional[int] = None
    provider: Optional[str] = "zeta-code"


class CompleteCodeResponse(BaseSchema):
    """代碼補全響應Schema"""
    completions: List[str]
    suggested_code: Optional[str] = None
    provider: str


class ExplainCodeRequest(BaseSchema):
    """代碼解釋請求Schema"""
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(..., min_length=1, max_length=50)
    provider: Optional[str] = "zeta-code"


class ExplainCodeResponse(BaseSchema):
    """代碼解釋響應Schema"""
    explanation: str
    key_points: List[str]
    provider: str


class ReviewCodeRequest(BaseSchema):
    """代碼審查請求Schema"""
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(..., min_length=1, max_length=50)
    review_type: str = Field(default="general", pattern="^(general|security|performance|style)$")
    provider: Optional[str] = "zeta-code"


class ReviewCodeResponse(BaseSchema):
    """代碼審查響應Schema"""
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    score: Optional[float] = None
    provider: str


# =============================================================================
# Collaboration Schemas
# =============================================================================

class SendMessageRequest(BaseSchema):
    """發送消息請求Schema"""
    channel: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=4000)
    attachments: Optional[List[Dict[str, Any]]] = None
    provider: Optional[str] = "iota-collaboration"


class SendMessageResponse(BaseSchema):
    """發送消息響應Schema"""
    message_id: Optional[str] = None
    success: bool
    provider: str


class SummarizeChannelRequest(BaseSchema):
    """頻道摘要請求Schema"""
    channel: str = Field(..., min_length=1, max_length=200)
    hours: int = Field(default=24, ge=1, le=168)
    provider: Optional[str] = "iota-collaboration"


class SummarizeChannelResponse(BaseSchema):
    """頻道摘要響應Schema"""
    summary: str
    key_points: List[str]
    provider: str


# =============================================================================
# Deployment Schemas
# =============================================================================

class DeployRequest(BaseSchema):
    """部署請求Schema"""
    artifact_path: str = Field(..., min_length=1, max_length=500)
    environment: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., min_length=1, max_length=100)
    config_overrides: Optional[Dict[str, Any]] = None
    provider: Optional[str] = "omicron-deployment"


class DeployResponse(BaseSchema):
    """部署響應Schema"""
    deployment_id: str
    url: Optional[str] = None
    status: str
    provider: str


class DeploymentStatusResponse(BaseSchema):
    """部署狀態響應Schema"""
    deployment_id: str
    status: str
    url: Optional[str] = None
    created_at: Optional[str] = None
    provider: str


# =============================================================================
# Stream Response Schemas
# =============================================================================

class StreamChunkSchema(BaseSchema):
    """流式數據塊Schema"""
    content: Union[str, Dict[str, Any]]
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
