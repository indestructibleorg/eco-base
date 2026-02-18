"""
Inference request/response schemas - OpenAI-compatible API format.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Chat Completion (OpenAI-compatible) ──────────────────────────

class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class ContentPart(BaseModel):
    """Multimodal content part."""
    type: Literal["text", "image_url", "audio_url", "video_url"] = "text"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    audio_url: Optional[Dict[str, str]] = None
    video_url: Optional[Dict[str, str]] = None


class ChatMessage(BaseModel):
    role: ChatRole
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "default"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=131072)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: Optional[int] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    user: Optional[str] = None
    n: int = Field(default=1, ge=1, le=16)

    # Engine routing hints
    engine: Optional[str] = None
    priority: int = Field(default=0, ge=0, le=10)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response chunk."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[UsageInfo] = None


# ── Completion (Legacy) ──────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=2048, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    echo: bool = False
    seed: Optional[int] = None
    engine: Optional[str] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[CompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Embeddings ───────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    model: str = "default"
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str = ""
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Batch Inference ──────────────────────────────────────────────

class BatchInferenceRequest(BaseModel):
    requests: List[ChatCompletionRequest]
    batch_id: str = Field(default_factory=lambda: f"batch-{uuid.uuid4().hex[:12]}")
    priority: int = Field(default=0, ge=0, le=10)
    callback_url: Optional[str] = None


class BatchInferenceStatus(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    status: Literal["pending", "processing", "completed", "failed"]
    results: Optional[List[ChatCompletionResponse]] = None


# ── Health & Status ──────────────────────────────────────────────

class EngineStatus(BaseModel):
    engine: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    loaded_models: List[str] = []
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    queue_depth: int = 0


class PlatformHealth(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    engines: List[EngineStatus]
    total_requests_served: int = 0
    active_connections: int = 0