"""
SuperAI Platform - Main Application Factory
Enterprise-grade AI inference backend with multi-engine routing.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.config.settings import get_settings
from src.core.health import HealthChecker
from src.core.queue import RequestQueue
from src.core.registry import ModelRegistry
from src.core.router import InferenceRouter
from src.engines.vllm_adapter import VLLMAdapter
from src.engines.tgi_adapter import TGIAdapter
from src.engines.sglang_adapter import SGLangAdapter
from src.engines.ollama_adapter import OllamaAdapter
from src.engines.tensorrt_adapter import TensorRTAdapter
from src.engines.lmdeploy_adapter import LMDeployAdapter
from src.middleware.error_handler import error_handler
from src.middleware.rate_limiter import RateLimiter
from src.multimodal.audio import AudioPipeline
from src.multimodal.image_gen import ImageGenerationPipeline
from src.multimodal.video import VideoUnderstandingPipeline
from src.multimodal.vision import VisionLanguagePipeline
from src.specialized.agent import AgentService
from src.specialized.batch import BatchInferenceProcessor
from src.specialized.code_gen import CodeGenerationService
from src.specialized.rag import RAGPipeline
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

logger = get_logger("superai.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: startup and shutdown."""
    settings = get_settings()
    logger.info(
        "SuperAI Platform starting",
        version=settings.app_version,
        environment=settings.environment.value,
    )

    # ── Initialize core services ─────────────────────────────
    metrics = get_metrics()
    metrics.platform_info.info({
        "version": settings.app_version,
        "environment": settings.environment.value,
    })
    app.state.metrics = metrics

    # Model Registry
    registry = ModelRegistry()
    app.state.model_registry = registry
    logger.info("Model registry initialized", models=registry.count)

    # Inference Router
    inference_router = InferenceRouter(registry)
    app.state.inference_router = inference_router

    # ── Initialize engine adapters ───────────────────────────
    adapters = {
        "vllm": VLLMAdapter(timeout=settings.engine_timeout_seconds),
        "tgi": TGIAdapter(timeout=settings.engine_timeout_seconds),
        "sglang": SGLangAdapter(timeout=settings.engine_timeout_seconds),
        "ollama": OllamaAdapter(timeout=settings.engine_timeout_seconds),
        "tensorrt-llm": TensorRTAdapter(timeout=settings.engine_timeout_seconds),
        "lmdeploy": LMDeployAdapter(timeout=settings.engine_timeout_seconds),
    }

    for name, adapter in adapters.items():
        await adapter.initialize()
        inference_router.register_adapter(name, adapter)

    app.state.adapters = adapters

    # ── Health Checker ───────────────────────────────────────
    health_checker = HealthChecker(registry)
    await health_checker.start(interval=15.0)
    app.state.health_checker = health_checker

    # ── Request Queue ────────────────────────────────────────
    request_queue = RequestQueue()
    app.state.request_queue = request_queue

    # ── Rate Limiter ─────────────────────────────────────────
    rate_limiter = RateLimiter()
    app.state.rate_limiter = rate_limiter

    # ── Multimodal Pipelines ─────────────────────────────────
    vision = VisionLanguagePipeline()
    await vision.initialize()
    app.state.vision_pipeline = vision

    image_gen = ImageGenerationPipeline()
    await image_gen.initialize()
    app.state.image_gen_pipeline = image_gen

    audio = AudioPipeline()
    await audio.initialize()
    app.state.audio_pipeline = audio

    video = VideoUnderstandingPipeline()
    await video.initialize()
    app.state.video_pipeline = video

    # ── Specialized Services ─────────────────────────────────
    code_gen = CodeGenerationService(inference_router)
    app.state.code_gen_service = code_gen

    rag = RAGPipeline(inference_router)
    await rag.initialize()
    app.state.rag_pipeline = rag

    agent = AgentService(inference_router)
    app.state.agent_service = agent

    batch = BatchInferenceProcessor(inference_router)
    app.state.batch_processor = batch

    logger.info("SuperAI Platform started successfully")

    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("SuperAI Platform shutting down")

    await health_checker.stop()
    await request_queue.stop()

    for adapter in adapters.values():
        await adapter.shutdown()

    await vision.shutdown()
    await image_gen.shutdown()
    await audio.shutdown()
    await video.shutdown()
    await rag.shutdown()

    logger.info("SuperAI Platform stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Enterprise AI Inference Platform with multi-engine routing. "
            "Supports vLLM, TGI, SGLang, Ollama, TensorRT-LLM, LMDeploy. "
            "OpenAI-compatible API with multimodal, RAG, agent, and batch capabilities."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else "/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    error_handler(app)

    # Routes
    app.include_router(api_router)

    return app


# Application instance
app = create_app()