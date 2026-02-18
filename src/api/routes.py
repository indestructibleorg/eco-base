"""
API Routes - OpenAI-compatible REST API + platform management endpoints.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from src.middleware.auth import verify_api_key
from src.schemas.inference import (
    BatchInferenceRequest,
    BatchInferenceStatus,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    PlatformHealth,
)
from src.schemas.models import (
    ModelCapability,
    ModelInfo,
    ModelListResponse,
    ModelLoadRequest,
    ModelRegisterRequest,
    ModelStatus,
    ModelUnloadRequest,
)
from src.schemas.auth import APIKeyCreate, APIKeyResponse

router = APIRouter()


# ── OpenAI-Compatible Inference Endpoints ────────────────────────

@router.post("/v1/chat/completions", response_model=None, tags=["Inference"])
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    auth: Dict = Depends(verify_api_key),
):
    """OpenAI-compatible chat completion endpoint."""
    app = request.app
    rate_limiter = app.state.rate_limiter
    await rate_limiter.check(auth.get("key_id", auth.get("user_id", "anon")))

    router_svc = app.state.inference_router
    health = app.state.health_checker
    health.increment_requests()

    if body.stream:
        async def stream_generator():
            async for chunk in router_svc.chat_completion_stream(body):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = await router_svc.chat_completion(body)
    return response.model_dump()


@router.post("/v1/completions", response_model=CompletionResponse, tags=["Inference"])
async def completions(
    request: Request,
    body: CompletionRequest,
    auth: Dict = Depends(verify_api_key),
):
    """OpenAI-compatible text completion endpoint."""
    app = request.app
    await app.state.rate_limiter.check(auth.get("key_id", "anon"))
    app.state.health_checker.increment_requests()
    return await app.state.inference_router.completion(body)


@router.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["Inference"])
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    auth: Dict = Depends(verify_api_key),
):
    """OpenAI-compatible embeddings endpoint."""
    app = request.app
    await app.state.rate_limiter.check(auth.get("key_id", "anon"))
    return await app.state.inference_router.embedding(body)


# ── Model Management ─────────────────────────────────────────────

@router.get("/v1/models", response_model=None, tags=["Models"])
async def list_models(
    request: Request,
    capability: Optional[str] = Query(None),
    engine: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    auth: Dict = Depends(verify_api_key),
):
    """List available models (OpenAI-compatible)."""
    registry = request.app.state.model_registry
    cap = ModelCapability(capability) if capability else None
    st = ModelStatus(status) if status else None
    models = await registry.list_models(capability=cap, engine=engine, status=st, tag=tag)

    # OpenAI-compatible format
    return {
        "object": "list",
        "data": [
            {
                "id": m.model_id,
                "object": "model",
                "created": int(m.registered_at.timestamp()),
                "owned_by": "superai",
                "capabilities": [c.value for c in m.capabilities],
                "compatible_engines": m.compatible_engines,
                "status": m.status.value,
                "parameters_billion": m.parameters_billion,
                "context_length": m.context_length,
            }
            for m in models
        ],
    }


@router.post("/v1/models/register", response_model=ModelInfo, tags=["Models"])
async def register_model(
    request: Request,
    body: ModelRegisterRequest,
    auth: Dict = Depends(verify_api_key),
):
    """Register a new model in the registry."""
    registry = request.app.state.model_registry
    return await registry.register(body)


@router.post("/v1/models/load", tags=["Models"])
async def load_model(
    request: Request,
    body: ModelLoadRequest,
    auth: Dict = Depends(verify_api_key),
):
    """Load a model onto an inference engine."""
    registry = request.app.state.model_registry
    model = await registry.get(body.model_id)
    if not model:
        raise ValueError(f"Model '{body.model_id}' not found")

    await registry.update_status(body.model_id, ModelStatus.LOADING, body.engine)
    # Engine adapter handles actual loading
    await registry.update_status(body.model_id, ModelStatus.READY, body.engine)

    return {"status": "loaded", "model_id": body.model_id, "engine": body.engine}


@router.post("/v1/models/unload", tags=["Models"])
async def unload_model(
    request: Request,
    body: ModelUnloadRequest,
    auth: Dict = Depends(verify_api_key),
):
    """Unload a model from an inference engine."""
    registry = request.app.state.model_registry
    await registry.update_status(body.model_id, ModelStatus.UNLOADING, body.engine)
    await registry.update_status(body.model_id, ModelStatus.REGISTERED, body.engine)
    return {"status": "unloaded", "model_id": body.model_id, "engine": body.engine}


@router.get("/v1/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model(
    request: Request,
    model_id: str,
    auth: Dict = Depends(verify_api_key),
):
    """Get detailed model information."""
    registry = request.app.state.model_registry
    model = await registry.get(model_id)
    if not model:
        raise ValueError(f"Model '{model_id}' not found")
    return model


@router.delete("/v1/models/{model_id}", tags=["Models"])
async def delete_model(
    request: Request,
    model_id: str,
    auth: Dict = Depends(verify_api_key),
):
    """Remove a model from the registry."""
    registry = request.app.state.model_registry
    deleted = await registry.delete(model_id)
    if not deleted:
        raise ValueError(f"Model '{model_id}' not found")
    return {"status": "deleted", "model_id": model_id}


# ── Batch Inference ──────────────────────────────────────────────

@router.post("/v1/batch", response_model=BatchInferenceStatus, tags=["Batch"])
async def submit_batch(
    request: Request,
    body: BatchInferenceRequest,
    auth: Dict = Depends(verify_api_key),
):
    """Submit a batch inference job."""
    processor = request.app.state.batch_processor
    return await processor.submit(body)


@router.get("/v1/batch/{batch_id}", response_model=BatchInferenceStatus, tags=["Batch"])
async def get_batch_status(
    request: Request,
    batch_id: str,
    auth: Dict = Depends(verify_api_key),
):
    """Get batch job status."""
    processor = request.app.state.batch_processor
    status = await processor.get_status(batch_id)
    if not status:
        raise ValueError(f"Batch '{batch_id}' not found")
    return status


@router.delete("/v1/batch/{batch_id}", tags=["Batch"])
async def cancel_batch(
    request: Request,
    batch_id: str,
    auth: Dict = Depends(verify_api_key),
):
    """Cancel a batch job."""
    processor = request.app.state.batch_processor
    cancelled = await processor.cancel(batch_id)
    return {"cancelled": cancelled, "batch_id": batch_id}


# ── Multimodal Endpoints ─────────────────────────────────────────

@router.post("/v1/vision/analyze", tags=["Multimodal"])
async def analyze_image(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Analyze an image using vision-language model."""
    pipeline = request.app.state.vision_pipeline
    return await pipeline.analyze_image(**body)


@router.post("/v1/audio/transcriptions", tags=["Multimodal"])
async def transcribe_audio(
    request: Request,
    auth: Dict = Depends(verify_api_key),
):
    """Transcribe audio to text (OpenAI-compatible)."""
    pipeline = request.app.state.audio_pipeline
    form = await request.form()
    audio_file = form.get("file")
    if audio_file:
        audio_bytes = await audio_file.read()
        return await pipeline.transcribe(
            audio_bytes=audio_bytes,
            model=form.get("model"),
            language=form.get("language"),
            response_format=form.get("response_format", "json"),
        )
    raise ValueError("Audio file required")


@router.post("/v1/audio/speech", tags=["Multimodal"])
async def synthesize_speech(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Text-to-speech synthesis (OpenAI-compatible)."""
    pipeline = request.app.state.audio_pipeline
    return await pipeline.synthesize(**body)


@router.post("/v1/images/generations", tags=["Multimodal"])
async def generate_image(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Generate images from text prompt."""
    pipeline = request.app.state.image_gen_pipeline
    return await pipeline.generate(**body)


# ── Specialized Endpoints ────────────────────────────────────────

@router.post("/v1/code/generate", tags=["Specialized"])
async def generate_code(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Generate code from natural language."""
    service = request.app.state.code_gen_service
    return await service.generate_code(**body)


@router.post("/v1/code/review", tags=["Specialized"])
async def review_code(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Review code for issues."""
    service = request.app.state.code_gen_service
    return await service.code_review(**body)


@router.post("/v1/rag/ingest", tags=["RAG"])
async def rag_ingest(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Ingest a document into RAG knowledge base."""
    pipeline = request.app.state.rag_pipeline
    return await pipeline.ingest_document(**body)


@router.post("/v1/rag/query", tags=["RAG"])
async def rag_query(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Query the RAG knowledge base."""
    pipeline = request.app.state.rag_pipeline
    return await pipeline.query(**body)


@router.post("/v1/agent/run", tags=["Agent"])
async def run_agent(
    request: Request,
    body: Dict[str, Any],
    auth: Dict = Depends(verify_api_key),
):
    """Run an agent task with tool calling."""
    service = request.app.state.agent_service
    return await service.run(**body)


# ── Platform Management ──────────────────────────────────────────

@router.get("/health", response_model=PlatformHealth, tags=["Platform"])
async def health_check(request: Request):
    """Platform health check (no auth required)."""
    return await request.app.state.health_checker.get_platform_health()


@router.get("/v1/engines", tags=["Platform"])
async def list_engines(
    request: Request,
    auth: Dict = Depends(verify_api_key),
):
    """List all inference engines and their status."""
    return request.app.state.inference_router.get_engine_status()


@router.get("/metrics", tags=["Platform"])
async def prometheus_metrics(request: Request):
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response

    metrics = request.app.state.metrics
    return Response(
        content=metrics.export(),
        media_type=metrics.content_type,
    )


@router.get("/v1/queue/stats", tags=["Platform"])
async def queue_stats(
    request: Request,
    auth: Dict = Depends(verify_api_key),
):
    """Get request queue statistics."""
    return await request.app.state.request_queue.get_stats()


# ── API Key Management ───────────────────────────────────────────

@router.post("/v1/api-keys", response_model=APIKeyResponse, tags=["Auth"])
async def create_api_key(
    request: Request,
    body: APIKeyCreate,
    auth: Dict = Depends(verify_api_key),
):
    """Create a new API key (admin only)."""
    if auth.get("role") != "admin":
        raise ValueError("Admin role required to create API keys")
    from src.middleware.auth import get_auth
    return get_auth().create_api_key(body)


@router.get("/v1/api-keys", tags=["Auth"])
async def list_api_keys(
    request: Request,
    auth: Dict = Depends(verify_api_key),
):
    """List all API keys (admin only)."""
    if auth.get("role") != "admin":
        raise ValueError("Admin role required")
    from src.middleware.auth import get_auth
    return get_auth().list_api_keys()


@router.delete("/v1/api-keys/{key_id}", tags=["Auth"])
async def revoke_api_key(
    request: Request,
    key_id: str,
    auth: Dict = Depends(verify_api_key),
):
    """Revoke an API key (admin only)."""
    if auth.get("role") != "admin":
        raise ValueError("Admin role required")
    from src.middleware.auth import get_auth
    revoked = get_auth().revoke_api_key(key_id)
    return {"revoked": revoked, "key_id": key_id}