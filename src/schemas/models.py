"""
Model registry schemas.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelFormat(str, Enum):
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    GPTQ = "gptq"
    AWQ = "awq"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    FP8 = "fp8"


class ModelCapability(str, Enum):
    TEXT_GENERATION = "text-generation"
    CHAT = "chat"
    CODE_GENERATION = "code-generation"
    MATH_REASONING = "math-reasoning"
    VISION_LANGUAGE = "vision-language"
    IMAGE_GENERATION = "image-generation"
    SPEECH_RECOGNITION = "speech-recognition"
    TEXT_TO_SPEECH = "text-to-speech"
    VIDEO_UNDERSTANDING = "video-understanding"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function-calling"
    RAG = "rag"
    LONG_CONTEXT = "long-context"


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


class QuantizationConfig(BaseModel):
    method: str = "none"
    bits: int = 16
    group_size: int = 128
    desc_act: bool = False


class ModelHardwareRequirements(BaseModel):
    min_gpu_memory_gb: float = 0.0
    recommended_gpu_memory_gb: float = 0.0
    min_cpu_memory_gb: float = 0.0
    min_gpu_count: int = 1
    supported_gpu_architectures: List[str] = ["ampere", "hopper", "ada"]
    supports_cpu_only: bool = False


class ModelRegisterRequest(BaseModel):
    model_id: str
    display_name: Optional[str] = None
    source: str  # HuggingFace repo ID or local path
    format: ModelFormat = ModelFormat.SAFETENSORS
    capabilities: List[ModelCapability] = [ModelCapability.CHAT]
    compatible_engines: List[str] = ["vllm"]
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    hardware_requirements: ModelHardwareRequirements = Field(
        default_factory=ModelHardwareRequirements
    )
    context_length: int = 4096
    parameters_billion: Optional[float] = None
    license: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class ModelInfo(BaseModel):
    model_id: str
    display_name: str
    source: str
    format: ModelFormat
    capabilities: List[ModelCapability]
    compatible_engines: List[str]
    quantization: QuantizationConfig
    hardware_requirements: ModelHardwareRequirements
    context_length: int
    parameters_billion: Optional[float]
    license: Optional[str]
    status: ModelStatus
    loaded_on_engines: List[str] = []
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    registered_at: datetime
    last_used_at: Optional[datetime] = None
    total_requests: int = 0


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
    total: int


class ModelLoadRequest(BaseModel):
    model_id: str
    engine: str = "vllm"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    extra_params: Dict[str, Any] = {}


class ModelUnloadRequest(BaseModel):
    model_id: str
    engine: str = "vllm"
    force: bool = False