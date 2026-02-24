# =============================================================================
# Deployment Endpoints
# =============================================================================

from fastapi import APIRouter, Depends, Request, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user_id, get_optional_user_id
from app.core.logging import get_logger
from app.core.exceptions import ProviderError
from app.db.base import get_db
from app.schemas.base import ResponseSchema
from app.schemas.platform import (
    DeployRequest, DeployResponse,
    DeploymentStatusResponse
)

router = APIRouter()
logger = get_logger("deployment")


@router.post("/deploy", response_model=ResponseSchema[DeployResponse])
async def deploy(
    request: Request,
    deploy_request: DeployRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """部署應用"""
    
    provider = deploy_request.provider or "omicron-deployment"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("token"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "deployment_requested",
        provider=provider,
        artifact=deploy_request.artifact_path,
        environment=deploy_request.environment,
        version=deploy_request.version,
        user_id=user_id,
    )
    
    # 模擬部署響應
    import uuid
    deployment_id = str(uuid.uuid4())[:8]
    
    return ResponseSchema(
        data=DeployResponse(
            deployment_id=deployment_id,
            url=f"https://{deploy_request.artifact_path}-{deployment_id}.vercel.app",
            status="building",
            provider=provider,
        )
    )


@router.get("/status/{deployment_id}", response_model=ResponseSchema[DeploymentStatusResponse])
async def get_deployment_status(
    request: Request,
    deployment_id: str = Path(..., description="部署ID"),
    provider: str = "omicron-deployment",
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """獲取部署狀態"""
    
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("token"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "deployment_status_requested",
        provider=provider,
        deployment_id=deployment_id,
        user_id=user_id,
    )
    
    return ResponseSchema(
        data=DeploymentStatusResponse(
            deployment_id=deployment_id,
            status="ready",
            url=f"https://app-{deployment_id}.vercel.app",
            created_at="2024-01-15T10:30:00Z",
            provider=provider,
        )
    )
