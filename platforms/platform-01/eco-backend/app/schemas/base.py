# =============================================================================
# Base Schemas
# =============================================================================

from datetime import datetime
from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel, Field, ConfigDict


T = TypeVar("T")


class BaseSchema(BaseModel):
    """基礎Schema"""
    model_config = ConfigDict(from_attributes=True)


class TimestampSchema(BaseSchema):
    """時間戳Schema"""
    created_at: datetime
    updated_at: Optional[datetime] = None


class IDSchema(BaseSchema):
    """ID Schema"""
    id: str


class ResponseSchema(BaseSchema, Generic[T]):
    """通用響應Schema"""
    success: bool = True
    data: Optional[T] = None
    error: Optional[dict] = None
    request_id: Optional[str] = None


class PaginationSchema(BaseSchema):
    """分頁Schema"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    total: int = 0
    pages: int = 0


class PaginatedResponseSchema(ResponseSchema[List[T]], Generic[T]):
    """分頁響應Schema"""
    pagination: PaginationSchema


class ErrorDetailSchema(BaseSchema):
    """錯誤詳情Schema"""
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponseSchema(BaseSchema):
    """錯誤響應Schema"""
    success: bool = False
    error: ErrorDetailSchema
    request_id: Optional[str] = None
