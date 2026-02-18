"""
LMDeploy Engine Adapter
Dual-engine design with TurboMind + PyTorch, persistent batch processing, KV-cache optimization.
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

logger = get_logger("superai.engines.lmdeploy")


class LMDeployAdapter(BaseEngineAdapter):
    """
    Adapter for LMDeploy inference engine.

    LMDeploy features utilized:
    - TurboMind engine for high-performance C++ inference
    - PyTorch engine for broad model compatibility
    - Persistent batch processing for interactive inference
    - KV-cache quantization (INT4/INT8) for memory efficiency
    - W4A16 quantization with minimal accuracy loss
    - OpenAI-compatible API via lmdeploy serve
    """

    def __init__(self, timeout: float = 120.0):
        super().__init__("lmdeploy", timeout)

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

    async def interactive_session(
        self,
        session_id: str,
        prompt: str,
        model: ModelInfo,
        endpoint: Any,
        max_tokens: int = 2048,
    ) -> str:
        """
        LMDeploy persistent batch session for interactive inference.
        Maintains KV-cache across turns for efficient multi-turn conversations.
        """
        payload = {
            "session_id": session_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "interactive_mode": True,
        }

        resp = await self._client.post(
            f"{endpoint.base_url}/v1/chat/interactive",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "")