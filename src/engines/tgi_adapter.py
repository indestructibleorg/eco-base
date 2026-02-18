"""
TGI (Text Generation Inference) Engine Adapter
HuggingFace's production inference server with Flash Attention and quantization.
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

logger = get_logger("superai.engines.tgi")


class TGIAdapter(BaseEngineAdapter):
    """
    Adapter for HuggingFace Text Generation Inference.

    TGI features utilized:
    - Flash Attention 2 for efficient attention computation
    - Continuous batching with token streaming
    - GPTQ/AWQ/bitsandbytes quantization support
    - Tensor parallelism for multi-GPU
    - OpenAI-compatible API
    - Watermarking and safety features
    """

    def __init__(self, timeout: float = 120.0):
        super().__init__("tgi", timeout)

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> ChatCompletionResponse:
        payload = {
            "model": model.source,
            "messages": self._build_messages_payload(request),
            **self._build_generation_params(request),
            "stream": False,
        }

        # TGI uses repetition_penalty instead of frequency/presence
        if request.repetition_penalty != 1.0:
            payload["repetition_penalty"] = request.repetition_penalty

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

        if request.repetition_penalty != 1.0:
            payload["repetition_penalty"] = request.repetition_penalty

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

                yield ChatCompletionStreamResponse(
                    id=data.get("id", ""),
                    model=model.model_id,
                    choices=choices,
                )

    async def _generate_raw(
        self, prompt: str, params: dict, endpoint: Any
    ) -> dict:
        """Use TGI's native /generate endpoint for advanced features."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": params.get("max_tokens", 2048),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
                "repetition_penalty": params.get("repetition_penalty", 1.0),
                "do_sample": params.get("temperature", 0.7) > 0,
                "return_full_text": False,
            },
        }
        if params.get("stop"):
            payload["parameters"]["stop_sequences"] = (
                params["stop"] if isinstance(params["stop"], list) else [params["stop"]]
            )

        resp = await self._client.post(
            f"{endpoint.base_url}/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()