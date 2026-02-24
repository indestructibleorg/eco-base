# =============================================================================
# Cognitive Computing Endpoints
# =============================================================================

from typing import AsyncGenerator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.core.config import settings
from app.core.security import get_current_user_id, get_optional_user_id
from app.core.logging import get_logger
from app.core.exceptions import ProviderError, ValidationError
from app.db.base import get_db
from app.schemas.base import ResponseSchema
from app.schemas.platform import (
    GenerateTextRequest, GenerateTextResponse,
    EmbedTextsRequest, EmbedTextsResponse,
    FunctionCallRequest, FunctionCallResponse,
    StreamChunkSchema
)
from app.services.platform_integration_service import platform_integration_service

router = APIRouter()
logger = get_logger("cognitive")


@router.post("/generate", response_model=ResponseSchema[GenerateTextResponse])
async def generate_text(
    request: Request,
    gen_request: GenerateTextRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """生成文本"""
    
    if gen_request.stream:
        raise ValidationError("Use /generate/stream endpoint for streaming")
    
    # 檢查提供者配置
    provider = gen_request.provider or "gamma-cognitive"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    # 调用平台集成框架
    logger.info(
        "text_generation_requested",
        provider=provider,
        prompt_length=len(gen_request.prompt),
        user_id=user_id,
    )
    
    result = await platform_integration_service.chat_completion(
        messages=[{"role": "user", "content": gen_request.prompt}],
        model=gen_request.model,
        temperature=gen_request.temperature or 0.7,
        provider=provider,
    )
    
    if not result.success:
        raise ProviderError(provider, result.error or "Generation failed")
    
    return ResponseSchema(
        data=GenerateTextResponse(
            content=result.data.get("content", "") if isinstance(result.data, dict) else str(result.data),
            model=result.data.get("model", gen_request.model or "default") if isinstance(result.data, dict) else gen_request.model or "default",
            usage=result.data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) if isinstance(result.data, dict) else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            provider=result.provider or provider,
        )
    )


@router.post("/generate/stream")
async def generate_text_stream(
    request: Request,
    gen_request: GenerateTextRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """流式生成文本"""
    
    provider = gen_request.provider or "gamma-cognitive"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    async def stream_generator() -> AsyncGenerator[str, None]:
        """流式生成器"""
        try:
            # 使用平台集成框架进行流式生成
            async for text in platform_integration_service.stream_chat_completion(
                messages=[{"role": "user", "content": gen_request.prompt}],
                model=gen_request.model,
                temperature=gen_request.temperature or 0.7,
            ):
                if text:
                    chunk = StreamChunkSchema(
                        content=text,
                        is_final=False,
                        metadata={},
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error("stream_generation_failed", error=str(e))
            # 发送错误标记
            chunk = StreamChunkSchema(
                content="",
                is_final=True,
                metadata={"error": str(e)},
            )
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
    
    logger.info(
        "text_stream_generation_requested",
        provider=provider,
        prompt_length=len(gen_request.prompt),
        user_id=user_id,
    )
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/embed", response_model=ResponseSchema[EmbedTextsResponse])
async def embed_texts(
    request: Request,
    embed_request: EmbedTextsRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """嵌入文本"""
    
    provider = embed_request.provider or "gamma-cognitive"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    # 生成模擬嵌入向量
    import random
    embeddings = [
        [random.uniform(-1, 1) for _ in range(1536)]
        for _ in embed_request.texts
    ]
    
    logger.info(
        "text_embedding_requested",
        provider=provider,
        text_count=len(embed_request.texts),
        user_id=user_id,
    )
    
    return ResponseSchema(
        data=EmbedTextsResponse(
            embeddings=embeddings,
            model="text-embedding-simulated",
            provider=provider,
        )
    )


@router.post("/function-call", response_model=ResponseSchema[FunctionCallResponse])
async def function_call(
    request: Request,
    call_request: FunctionCallRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """函數調用"""
    
    provider = call_request.provider or "gamma-cognitive"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "function_call_requested",
        provider=provider,
        prompt_length=len(call_request.prompt),
        function_count=len(call_request.functions),
        user_id=user_id,
    )
    
    # 模擬函數調用響應
    return ResponseSchema(
        data=FunctionCallResponse(
            content=None,
            function_call={
                "name": call_request.functions[0]["name"] if call_request.functions else "unknown",
                "arguments": '{"location": "Tokyo", "unit": "celsius"}'
            },
            provider=provider,
        )
    )
