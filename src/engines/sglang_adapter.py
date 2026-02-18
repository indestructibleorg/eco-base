"""
SGLang Engine Adapter
RadixAttention prefix caching, structured generation, 6.4x throughput boost.
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

logger = get_logger("superai.engines.sglang")


class SGLangAdapter(BaseEngineAdapter):
    """
    Adapter for SGLang inference engine.

    SGLang features utilized:
    - RadixAttention for automatic prefix caching (multi-turn optimization)
    - Structured generation with constrained decoding (JSON, regex)
    - Compressed finite state machine for grammar-guided generation
    - Up to 6.4x throughput improvement on structured output tasks
    - OpenAI-compatible API
    """

    def __init__(self, timeout: float = 120.0):
        super().__init__("sglang", timeout)

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

        # SGLang-specific: structured output via regex/json_schema
        if request.response_format and request.response_format.type == "json_schema":
            if request.response_format.json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.response_format.json_schema,
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

        if request.response_format and request.response_format.type == "json_schema":
            if request.response_format.json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.response_format.json_schema,
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

                yield ChatCompletionStreamResponse(
                    id=data.get("id", ""),
                    model=model.model_id,
                    choices=choices,
                )

    async def generate_structured(
        self,
        prompt: str,
        json_schema: dict,
        model: ModelInfo,
        endpoint: Any,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict:
        """
        SGLang-native structured generation using constrained decoding.
        Guarantees output conforms to the provided JSON schema.
        """
        payload = {
            "model": model.source,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
        }

        resp = await self._client.post(
            f"{endpoint.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return json.loads(content)