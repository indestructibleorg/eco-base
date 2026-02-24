# =============================================================================
# Collaboration Endpoints
# =============================================================================

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user_id, get_optional_user_id
from app.core.logging import get_logger
from app.core.exceptions import ProviderError
from app.db.base import get_db
from app.schemas.base import ResponseSchema
from app.schemas.platform import (
    SendMessageRequest, SendMessageResponse,
    SummarizeChannelRequest, SummarizeChannelResponse
)

router = APIRouter()
logger = get_logger("collaboration")


@router.post("/message", response_model=ResponseSchema[SendMessageResponse])
async def send_message(
    request: Request,
    message_request: SendMessageRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """發送消息"""
    
    provider = message_request.provider or "iota-collaboration"
    config = settings.get_provider_configs().get(provider)
    
    if not config:
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "message_send_requested",
        provider=provider,
        channel=message_request.channel,
        content_length=len(message_request.content),
        user_id=user_id,
    )
    
    return ResponseSchema(
        data=SendMessageResponse(
            message_id=f"msg_{hash(message_request.content) % 1000000}",
            success=True,
            provider=provider,
        )
    )


@router.post("/summarize", response_model=ResponseSchema[SummarizeChannelResponse])
async def summarize_channel(
    request: Request,
    summarize_request: SummarizeChannelRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """頻道摘要"""
    
    provider = summarize_request.provider or "iota-collaboration"
    config = settings.get_provider_configs().get(provider)
    
    if not config:
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "channel_summarize_requested",
        provider=provider,
        channel=summarize_request.channel,
        hours=summarize_request.hours,
        user_id=user_id,
    )
    
    return ResponseSchema(
        data=SummarizeChannelResponse(
            summary=f"Summary of #{summarize_request.channel} for the past {summarize_request.hours} hours: Team discussed deployment strategies and resolved 3 critical bugs.",
            key_points=[
                "Deployment pipeline optimization discussed",
                "Bug fixes for authentication module",
                "New feature planning for next sprint",
            ],
            provider=provider,
        )
    )
