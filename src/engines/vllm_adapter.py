"""
vLLM Engine Adapter
Leverages PagedAttention, continuous batching, and prefix caching.
Supports OpenAI-compatible API natively.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict

import httpx

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
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    UsageInfo,
)
from src.schemas.models import ModelInfo, ModelLoadRequest
from src.utils.logging import get_logger

logger = get_logger("superai.engines.vllm")


class VLLMAdapter(BaseEngineAdapter):
    """
    Adapter for vLLM inference engine.

    vLLM features utilized:
    - PagedAttention for efficient KV-cache management
    - Continuous batching for high throughput
    - Prefix caching for multi-turn conversations
    - Tensor parallelism for large models
    - OpenAI-compatible API (native)
    """

    def __init__(self, timeout: float = 120.0):
        super().__init__("vllm", timeout)

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> ChatCompletionResponse:
        """Forward to vLLM's OpenAI-compatible /v1/chat/completions."""
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
            system_fingerprint=data.get("system_fingerprint"),
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Stream from vLLM's SSE endpoint."""
        payload = {
            "model": model.source,
            "messages": self._build_messages_payload(request),
            **self._build_generation_params(request),
            "stream": True,
            "stream_options": {"include_usage": True},
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
                                tool_calls=delta.get("tool_calls"),
                            ),
                            finish_reason=c.get("finish_reason"),
                        )
                    )

                usage = None
                if "usage" in data and data["usage"]:
                    u = data["usage"]
                    usage = UsageInfo(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                        total_tokens=u.get("total_tokens", 0),
                    )

                yield ChatCompletionStreamResponse(
                    id=data.get("id", ""),
                    model=model.model_id,
                    choices=choices,
                    usage=usage,
                )

    async def embedding(
        self,
        request: EmbeddingRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> EmbeddingResponse:
        """Forward to vLLM's /v1/embeddings endpoint."""
        payload = {
            "model": model.source,
            "input": request.input,
            "encoding_format": request.encoding_format,
        }

        resp = await self._client.post(
            f"{endpoint.base_url}/v1/embeddings",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = [
            EmbeddingData(
                index=e.get("index", i),
                embedding=e.get("embedding", []),
            )
            for i, e in enumerate(data.get("data", []))
        ]

        usage_data = data.get("usage", {})
        return EmbeddingResponse(
            data=embeddings,
            model=model.model_id,
            usage=UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )

    async def load_model(self, request: ModelLoadRequest, endpoint: Any) -> bool:
        """
        vLLM typically requires restart to load new models.
        For dynamic loading, use the model management API if available.
        """
        logger.info(
            "vLLM model load requested (requires engine restart for new models)",
            model_id=request.model_id,
            engine_url=endpoint.base_url,
        )
        return True