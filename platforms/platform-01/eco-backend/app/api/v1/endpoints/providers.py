# =============================================================================
# Provider Endpoints
# =============================================================================

from typing import List
from fastapi import APIRouter, Depends, Query

from app.core.security import get_optional_user_id
from app.core.logging import get_logger
from app.schemas.base import ResponseSchema
from app.schemas.platform import (
    ProviderInfoSchema, ProviderListResponse, ProviderHealthSchema
)
from app.services.provider_service import provider_service

router = APIRouter()
logger = get_logger("providers")


@router.get("", response_model=ResponseSchema[ProviderListResponse])
async def list_providers(
    domain: str = Query(None, description="按能力領域過濾"),
    user_id: str = Depends(get_optional_user_id),
):
    """列出所有可用的提供者"""
    providers = provider_service.list_providers(domain)
    
    logger.info("providers_listed", count=len(providers), domain=domain)
    
    return ResponseSchema(
        data=ProviderListResponse(
            items=[ProviderInfoSchema(**p) for p in providers],
            total=len(providers)
        )
    )


@router.get("/health", response_model=ResponseSchema[List[ProviderHealthSchema]])
async def check_providers_health(
    user_id: str = Depends(get_optional_user_id),
):
    """檢查所有提供者的健康狀態"""
    from datetime import datetime
    
    providers = provider_service.list_providers()
    health_statuses = []
    
    for provider in providers:
        health = provider_service.check_health(provider["provider_id"])
        health_statuses.append(ProviderHealthSchema(
            provider_id=health["provider_id"],
            healthy=health["healthy"],
            error=None if health["healthy"] else "Configuration not found or circuit open",
            latency_ms=None,
            checked_at=datetime.utcnow().isoformat(),
        ))
    
    return ResponseSchema(data=health_statuses)


@router.get("/{provider_id}", response_model=ResponseSchema[ProviderInfoSchema])
async def get_provider(
    provider_id: str,
    user_id: str = Depends(get_optional_user_id),
):
    """獲取特定提供者的詳細信息"""
    provider = provider_service.get_provider(provider_id)
    
    if not provider:
        from app.core.exceptions import ResourceNotFoundError
        raise ResourceNotFoundError("Provider", provider_id)
    
    return ResponseSchema(data=ProviderInfoSchema(**provider))


@router.get("/{provider_id}/health", response_model=ResponseSchema[ProviderHealthSchema])
async def check_provider_health(
    provider_id: str,
    user_id: str = Depends(get_optional_user_id),
):
    """檢查特定提供者的健康狀態"""
    from datetime import datetime
    
    if not provider_service.get_provider(provider_id):
        from app.core.exceptions import ResourceNotFoundError
        raise ResourceNotFoundError("Provider", provider_id)
    
    health = provider_service.check_health(provider_id)
    
    return ResponseSchema(
        data=ProviderHealthSchema(
            provider_id=health["provider_id"],
            healthy=health["healthy"],
            error=None if health["healthy"] else health.get("circuit_state", "unknown"),
            latency_ms=None,
            checked_at=datetime.utcnow().isoformat(),
        )
    )
