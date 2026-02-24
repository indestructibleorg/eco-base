# =============================================================================
# Data Persistence Endpoints
# =============================================================================

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user_id, get_optional_user_id
from app.core.logging import get_logger
from app.core.exceptions import ProviderError
from app.db.base import get_db
from app.schemas.base import ResponseSchema
from app.schemas.platform import (
    QueryDataRequest, QueryDataResponse,
    MutateDataRequest, MutateDataResponse,
    VectorSearchRequest, VectorSearchResponse
)
from app.services.platform_integration_service import platform_integration_service

router = APIRouter()
logger = get_logger("data")


@router.post("/query", response_model=ResponseSchema[QueryDataResponse])
async def query_data(
    request: Request,
    query_request: QueryDataRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """查詢數據"""
    
    provider = query_request.provider or "alpha-persistence"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("url"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "data_query_requested",
        provider=provider,
        table=query_request.table,
        filters=query_request.filters,
        user_id=user_id,
    )
    
    # 调用平台集成框架
    result = await platform_integration_service.query_data(
        table=query_request.table,
        filters=query_request.filters,
        provider=provider,
    )
    
    if not result.success:
        raise ProviderError(provider, result.error or "Query failed")
    
    data = result.data if isinstance(result.data, list) else []
    
    return ResponseSchema(
        data=QueryDataResponse(
            data=data,
            total=len(data),
            provider=result.provider or provider,
        )
    )


@router.post("/mutate", response_model=ResponseSchema[MutateDataResponse])
async def mutate_data(
    request: Request,
    mutate_request: MutateDataRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """變更數據"""
    
    provider = mutate_request.provider or "alpha-persistence"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("url"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "data_mutate_requested",
        provider=provider,
        operation=mutate_request.operation,
        table=mutate_request.table,
        user_id=user_id,
    )
    
    # 调用平台集成框架
    result = await platform_integration_service.persist_data(
        table=mutate_request.table,
        data=mutate_request.data or {},
        provider=provider,
    )
    
    if not result.success:
        raise ProviderError(provider, result.error or "Mutate failed")
    
    return ResponseSchema(
        data=MutateDataResponse(
            affected_rows=1,
            data=mutate_request.data,
            provider=result.provider or provider,
        )
    )


@router.post("/vector-search", response_model=ResponseSchema[VectorSearchResponse])
async def vector_search(
    request: Request,
    search_request: VectorSearchRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """向量搜索"""
    
    provider = search_request.provider or "alpha-persistence"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("url"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "vector_search_requested",
        provider=provider,
        table=search_request.table,
        top_k=search_request.top_k,
        user_id=user_id,
    )
    
    # 调用平台集成框架
    result = await platform_integration_service.vector_search(
        index=search_request.table,
        vector=search_request.vector,
        top_k=search_request.top_k or 5,
        provider=provider,
    )
    
    if not result.success:
        raise ProviderError(provider, result.error or "Vector search failed")
    
    results = result.data if isinstance(result.data, list) else []
    
    return ResponseSchema(
        data=VectorSearchResponse(
            results=results,
            provider=result.provider or provider,
        )
    )
