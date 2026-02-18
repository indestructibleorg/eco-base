"""
TensorRT-LLM Engine Adapter
NVIDIA deep optimization with FP8/FP4 quantization, kernel fusion, Tensor Core utilization.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from src.engines.base import BaseEngineAdapter
from src.schemas.inference import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatRole,
    ChoiceDelta,
    UsageInfo,
)
from src.schemas.models import ModelInfo
from src.utils.logging import get_logger

logger = get_logger("superai.engines.tensorrt")


class TensorRTAdapter(BaseEngineAdapter):
    """
    Adapter for TensorRT-LLM inference engine.

    TensorRT-LLM features utilized:
    - Deep kernel fusion for minimal launch overhead
    - FP8/FP4 quantization on Hopper/Ada architectures
    - In-flight batching (continuous batching variant)
    - Paged KV-cache with efficient memory management
    - Multi-GPU tensor/pipeline parallelism
    - Triton Inference Server integration
    """

    def __init__(self, timeout: float = 120.0):
        super().__init__("tensorrt-llm", timeout)

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> ChatCompletionResponse:
        # TensorRT-LLM with Triton uses OpenAI-compatible API via trtllm-serve
        payload = {
            "model": model.source,
            "messages": self._build_messages_payload(request),
            **self._build_generation_params(request),
            "stream": False,
        }

        resp = await self._client.post(
            f"{endpoint.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        choices = [
            ChatCompletionChoice(
                index=c.get("index", i),
                message=ChatMessage(
                    role=ChatRole(c["message"]["role"]),
                    content=c["message"].get("content", ""),
                    tool_calls=c["message"].get("tool_calls"),
                ),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for i, c in enumerate(data.get("choices", []))
        ]

        usage_data = data.get("usage", {})
        return ChatCompletionResponse(
            id=data.get("id", ""),
            model=model.model_id,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        payload = {
            "model": model.source,
            "messages": self._build_messages_payload(request),
            **self._build_generation_params(request),
            "stream": True,
        }

        async with self._client.stream(
            "POST",
            f"{endpoint.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = []
                for c in data.get("choices", []):
                    delta = c.get("delta", {})
                    choices.append(
                        ChatCompletionStreamChoice(
                            index=c.get("index", 0),
                            delta=ChoiceDelta(
                                role=delta.get("role"),
                                content=delta.get("content"),
                            ),
                            finish_reason=c.get("finish_reason"),
                        )
                    )

                yield ChatCompletionStreamResponse(
                    id=data.get("id", ""),
                    model=model.model_id,
                    choices=choices,
                )

    async def generate_triton(
        self,
        prompt: str,
        model: ModelInfo,
        endpoint: Any,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """
        Direct Triton Inference Server generate endpoint for
        maximum throughput with TensorRT-LLM backend.
        """
        payload = {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        resp = await self._client.post(
            f"{endpoint.base_url}/v2/models/{model.source}/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()