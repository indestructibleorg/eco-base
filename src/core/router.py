"""
Inference Router - Intelligent multi-engine request dispatch.
Routes requests to optimal engine based on model, load, latency, and capabilities.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from src.config.settings import InferenceEngine, get_settings
from src.core.registry import ModelRegistry
from src.schemas.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from src.schemas.models import ModelCapability, ModelInfo, ModelStatus
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

logger = get_logger("superai.router")


class EngineEndpoint:
    """Represents a live inference engine endpoint."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        engine_type: InferenceEngine,
    ):
        self.name = name
        self.host = host
        self.port = port
        self.engine_type = engine_type
        self.healthy = False
        self.loaded_models: List[str] = []
        self.active_requests = 0
        self.total_requests = 0
        self.avg_latency_ms = 0.0
        self.last_health_check = 0.0
        self._latency_window: List[float] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def record_latency(self, latency_ms: float) -> None:
        self._latency_window.append(latency_ms)
        if len(self._latency_window) > 100:
            self._latency_window = self._latency_window[-100:]
        self.avg_latency_ms = sum(self._latency_window) / len(self._latency_window)

    @property
    def load_score(self) -> float:
        """Lower is better. Combines active requests and latency."""
        return self.active_requests * 10.0 + self.avg_latency_ms * 0.1


class InferenceRouter:
    """
    Routes inference requests to the optimal engine.

    Routing strategy:
    1. Resolve model → find compatible engines
    2. Filter healthy engines with model loaded
    3. Select engine with lowest load score (weighted: queue depth + latency)
    4. Dispatch request via engine adapter
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._settings = get_settings()
        self._metrics = get_metrics()
        self._engines: Dict[str, EngineEndpoint] = {}
        self._adapters: Dict[str, Any] = {}
        self._init_engines()

    def _init_engines(self) -> None:
        """Initialize engine endpoints from settings."""
        s = self._settings
        engine_configs = [
            ("vllm", s.vllm_host, s.vllm_port, InferenceEngine.VLLM),
            ("tgi", s.tgi_host, s.tgi_port, InferenceEngine.TGI),
            ("sglang", s.sglang_host, s.sglang_port, InferenceEngine.SGLANG),
            ("ollama", s.ollama_host, s.ollama_port, InferenceEngine.OLLAMA),
            ("tensorrt-llm", s.tensorrt_host, s.tensorrt_port, InferenceEngine.TENSORRT_LLM),
            ("lmdeploy", s.lmdeploy_host, s.lmdeploy_port, InferenceEngine.LMDEPLOY),
            ("deepspeed", s.deepspeed_host, s.deepspeed_port, InferenceEngine.DEEPSPEED),
        ]
        for name, host, port, etype in engine_configs:
            self._engines[name] = EngineEndpoint(name, host, port, etype)

    def register_adapter(self, engine_name: str, adapter: Any) -> None:
        """Register an engine adapter for request dispatch."""
        self._adapters[engine_name] = adapter
        logger.info("Engine adapter registered", engine=engine_name)

    async def _select_engine(
        self, model: ModelInfo, preferred_engine: Optional[str] = None
    ) -> Tuple[str, EngineEndpoint]:
        """Select the best engine for a given model and request."""

        # If user specified an engine, try it first
        if preferred_engine and preferred_engine in self._engines:
            ep = self._engines[preferred_engine]
            if ep.healthy and preferred_engine in model.compatible_engines:
                return preferred_engine, ep

        # Find all healthy, compatible engines
        candidates: List[Tuple[str, EngineEndpoint]] = []
        for engine_name in model.compatible_engines:
            ep = self._engines.get(engine_name)
            if ep and ep.healthy:
                candidates.append((engine_name, ep))

        if not candidates:
            # Fallback: try any healthy engine
            for name, ep in self._engines.items():
                if ep.healthy:
                    candidates.append((name, ep))

        if not candidates:
            raise RuntimeError(
                f"No healthy engine available for model '{model.model_id}'"
            )

        # Sort by load score (lower is better)
        candidates.sort(key=lambda x: x[1].load_score)
        return candidates[0]

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Route a chat completion request to the optimal engine."""
        model = await self._registry.resolve_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found in registry")

        engine_name, endpoint = await self._select_engine(model, request.engine)
        adapter = self._adapters.get(engine_name)
        if not adapter:
            raise RuntimeError(f"No adapter registered for engine '{engine_name}'")

        endpoint.active_requests += 1
        start = time.perf_counter()

        try:
            with self._metrics.track_request(engine_name, model.model_id, "/v1/chat/completions"):
                response = await adapter.chat_completion(request, model, endpoint)
                latency_ms = (time.perf_counter() - start) * 1000
                endpoint.record_latency(latency_ms)
                endpoint.total_requests += 1

                # Track tokens
                if response.usage:
                    self._metrics.tokens_generated.labels(
                        engine=engine_name, model=model.model_id, token_type="prompt"
                    ).inc(response.usage.prompt_tokens)
                    self._metrics.tokens_generated.labels(
                        engine=engine_name, model=model.model_id, token_type="completion"
                    ).inc(response.usage.completion_tokens)

                response.model = model.model_id
                return response
        finally:
            endpoint.active_requests -= 1

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Route a streaming chat completion request."""
        model = await self._registry.resolve_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found in registry")

        engine_name, endpoint = await self._select_engine(model, request.engine)
        adapter = self._adapters.get(engine_name)
        if not adapter:
            raise RuntimeError(f"No adapter registered for engine '{engine_name}'")

        endpoint.active_requests += 1
        start = time.perf_counter()

        try:
            first_token = True
            async for chunk in adapter.chat_completion_stream(request, model, endpoint):
                if first_token:
                    ttft = time.perf_counter() - start
                    self._metrics.time_to_first_token.labels(
                        engine=engine_name, model=model.model_id
                    ).observe(ttft)
                    first_token = False
                chunk.model = model.model_id
                yield chunk

            latency_ms = (time.perf_counter() - start) * 1000
            endpoint.record_latency(latency_ms)
            endpoint.total_requests += 1
        finally:
            endpoint.active_requests -= 1

    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Route a legacy completion request."""
        model = await self._registry.resolve_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found in registry")

        engine_name, endpoint = await self._select_engine(model, request.engine)
        adapter = self._adapters.get(engine_name)
        if not adapter:
            raise RuntimeError(f"No adapter registered for engine '{engine_name}'")

        endpoint.active_requests += 1
        start = time.perf_counter()

        try:
            with self._metrics.track_request(engine_name, model.model_id, "/v1/completions"):
                response = await adapter.completion(request, model, endpoint)
                latency_ms = (time.perf_counter() - start) * 1000
                endpoint.record_latency(latency_ms)
                response.model = model.model_id
                return response
        finally:
            endpoint.active_requests -= 1

    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Route an embedding request."""
        model = await self._registry.resolve_model(request.model)
        if not model:
            raise ValueError(f"Model '{request.model}' not found in registry")

        engine_name, endpoint = await self._select_engine(model, request.engine if hasattr(request, 'engine') else None)
        adapter = self._adapters.get(engine_name)
        if not adapter:
            raise RuntimeError(f"No adapter registered for engine '{engine_name}'")

        with self._metrics.track_request(engine_name, model.model_id, "/v1/embeddings"):
            response = await adapter.embedding(request, model, endpoint)
            response.model = model.model_id
            return response

    def get_engine_status(self) -> List[Dict[str, Any]]:
        """Get status of all engines."""
        return [
            {
                "engine": name,
                "healthy": ep.healthy,
                "active_requests": ep.active_requests,
                "total_requests": ep.total_requests,
                "avg_latency_ms": round(ep.avg_latency_ms, 2),
                "loaded_models": ep.loaded_models,
                "base_url": ep.base_url,
            }
            for name, ep in self._engines.items()
        ]