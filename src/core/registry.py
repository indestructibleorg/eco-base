"""
Model Registry - Central model lifecycle management.
Handles registration, discovery, loading, and unloading of models across engines.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config.settings import get_settings
from src.schemas.models import (
    ModelCapability,
    ModelFormat,
    ModelInfo,
    ModelLoadRequest,
    ModelRegisterRequest,
    ModelStatus,
    ModelUnloadRequest,
    QuantizationConfig,
    ModelHardwareRequirements,
)
from src.utils.logging import get_logger

logger = get_logger("superai.registry")


class ModelRegistry:
    """
    In-memory model registry with persistence support.
    Tracks all registered models, their status, and engine assignments.
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()
        self._settings = get_settings()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in default models."""
        defaults = [
            ModelRegisterRequest(
                model_id="llama-3.1-8b-instruct",
                display_name="Llama 3.1 8B Instruct",
                source="meta-llama/Llama-3.1-8B-Instruct",
                format=ModelFormat.SAFETENSORS,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                ],
                compatible_engines=["vllm", "tgi", "sglang", "ollama", "lmdeploy"],
                context_length=131072,
                parameters_billion=8.0,
                license="llama3.1",
                tags=["general", "instruction-following", "multilingual"],
            ),
            ModelRegisterRequest(
                model_id="llama-3.1-70b-instruct",
                display_name="Llama 3.1 70B Instruct",
                source="meta-llama/Llama-3.1-70B-Instruct",
                format=ModelFormat.SAFETENSORS,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.MATH_REASONING,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.LONG_CONTEXT,
                ],
                compatible_engines=["vllm", "tgi", "tensorrt-llm", "deepspeed"],
                context_length=131072,
                parameters_billion=70.0,
                license="llama3.1",
                tags=["general", "high-performance"],
                hardware_requirements=ModelHardwareRequirements(
                    min_gpu_memory_gb=40.0,
                    recommended_gpu_memory_gb=80.0,
                    min_gpu_count=2,
                ),
            ),
            ModelRegisterRequest(
                model_id="qwen2.5-72b-instruct",
                display_name="Qwen 2.5 72B Instruct",
                source="Qwen/Qwen2.5-72B-Instruct",
                format=ModelFormat.SAFETENSORS,
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.MATH_REASONING,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.LONG_CONTEXT,
                ],
                compatible_engines=["vllm", "sglang", "tgi", "lmdeploy"],
                context_length=131072,
                parameters_billion=72.0,
                license="apache-2.0",
                tags=["general", "multilingual", "gsm8k-95.8%"],
            ),
            ModelRegisterRequest(
                model_id="qwen2.5-vl-7b-instruct",
                display_name="Qwen 2.5 VL 7B Instruct",
                source="Qwen/Qwen2.5-VL-7B-Instruct",
                format=ModelFormat.SAFETENSORS,
                capabilities=[
                    ModelCapability.VISION_LANGUAGE,
                    ModelCapability.CHAT,
                ],
                compatible_engines=["vllm", "sglang", "lmdeploy"],
                context_length=32768,
                parameters_billion=7.0,
                license="apache-2.0",
                tags=["multimodal", "vision"],
            ),
            ModelRegisterRequest(
                model_id="deepseek-coder-v2",
                display_name="DeepSeek Coder V2",
                source="deepseek-ai/DeepSeek-Coder-V2-Instruct",
                format=ModelFormat.SAFETENSORS,
                capabilities=[
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.MATH_REASONING,
                ],
                compatible_engines=["vllm", "sglang"],
                context_length=128000,
                parameters_billion=236.0,
                license="deepseek",
                tags=["code", "math"],
            ),
            ModelRegisterRequest(
                model_id="whisper-large-v3-turbo",
                display_name="Whisper Large V3 Turbo",
                source="openai/whisper-large-v3-turbo",
                format=ModelFormat.SAFETENSORS,
                capabilities=[ModelCapability.SPEECH_RECOGNITION],
                compatible_engines=["vllm"],
                context_length=448,
                parameters_billion=0.8,
                license="apache-2.0",
                tags=["audio", "asr", "multilingual"],
            ),
            ModelRegisterRequest(
                model_id="bge-large-en-v1.5",
                display_name="BGE Large EN v1.5",
                source="BAAI/bge-large-en-v1.5",
                format=ModelFormat.SAFETENSORS,
                capabilities=[ModelCapability.EMBEDDING],
                compatible_engines=["vllm", "tgi"],
                context_length=512,
                parameters_billion=0.3,
                license="mit",
                tags=["embedding", "retrieval"],
            ),
        ]

        for req in defaults:
            self._models[req.model_id] = ModelInfo(
                model_id=req.model_id,
                display_name=req.display_name or req.model_id,
                source=req.source,
                format=req.format,
                capabilities=req.capabilities,
                compatible_engines=req.compatible_engines,
                quantization=req.quantization,
                hardware_requirements=req.hardware_requirements,
                context_length=req.context_length,
                parameters_billion=req.parameters_billion,
                license=req.license,
                status=ModelStatus.REGISTERED,
                tags=req.tags,
                metadata=req.metadata,
                registered_at=datetime.now(timezone.utc),
            )

    async def register(self, request: ModelRegisterRequest) -> ModelInfo:
        """Register a new model."""
        async with self._lock:
            if request.model_id in self._models:
                raise ValueError(f"Model '{request.model_id}' already registered")

            info = ModelInfo(
                model_id=request.model_id,
                display_name=request.display_name or request.model_id,
                source=request.source,
                format=request.format,
                capabilities=request.capabilities,
                compatible_engines=request.compatible_engines,
                quantization=request.quantization,
                hardware_requirements=request.hardware_requirements,
                context_length=request.context_length,
                parameters_billion=request.parameters_billion,
                license=request.license,
                status=ModelStatus.REGISTERED,
                tags=request.tags,
                metadata=request.metadata,
                registered_at=datetime.now(timezone.utc),
            )
            self._models[request.model_id] = info
            logger.info(
                "Model registered",
                model_id=request.model_id,
                source=request.source,
            )
            return info

    async def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return self._models.get(model_id)

    async def list_models(
        self,
        capability: Optional[ModelCapability] = None,
        engine: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tag: Optional[str] = None,
    ) -> List[ModelInfo]:
        """List models with optional filters."""
        results = list(self._models.values())
        if capability:
            results = [m for m in results if capability in m.capabilities]
        if engine:
            results = [m for m in results if engine in m.compatible_engines]
        if status:
            results = [m for m in results if m.status == status]
        if tag:
            results = [m for m in results if tag in m.tags]
        return results

    async def update_status(
        self, model_id: str, status: ModelStatus, engine: Optional[str] = None
    ) -> None:
        """Update model status."""
        async with self._lock:
            model = self._models.get(model_id)
            if not model:
                raise ValueError(f"Model '{model_id}' not found")
            model.status = status
            if engine and status == ModelStatus.READY:
                if engine not in model.loaded_on_engines:
                    model.loaded_on_engines.append(engine)
            elif engine and status in (ModelStatus.UNLOADING, ModelStatus.REGISTERED):
                if engine in model.loaded_on_engines:
                    model.loaded_on_engines.remove(engine)
            logger.info(
                "Model status updated",
                model_id=model_id,
                status=status.value,
                engine=engine,
            )

    async def resolve_model(self, model_name: str) -> Optional[ModelInfo]:
        """Resolve model name to ModelInfo. Supports aliases and partial matches."""
        if model_name == "default":
            model_name = self._settings.default_model

        # Direct match
        if model_name in self._models:
            return self._models[model_name]

        # Match by source
        for m in self._models.values():
            if m.source == model_name:
                return m

        # Partial match
        for m in self._models.values():
            if model_name.lower() in m.model_id.lower():
                return m

        return None

    async def delete(self, model_id: str) -> bool:
        """Remove a model from registry."""
        async with self._lock:
            if model_id in self._models:
                model = self._models[model_id]
                if model.loaded_on_engines:
                    raise ValueError(
                        f"Model '{model_id}' is still loaded on: {model.loaded_on_engines}"
                    )
                del self._models[model_id]
                logger.info("Model deleted", model_id=model_id)
                return True
            return False

    @property
    def count(self) -> int:
        return len(self._models)