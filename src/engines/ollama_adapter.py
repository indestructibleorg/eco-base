"""
Ollama Engine Adapter
Local deployment with extreme simplicity. GGUF quantization support.
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
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    UsageInfo,
)
from src.schemas.models import ModelInfo, ModelLoadRequest
from src.utils.logging import get_logger

logger = get_logger("superai.engines.ollama")


class OllamaAdapter(BaseEngineAdapter):
    """
    Adapter for Ollama inference engine.

    Ollama features utilized:
    - One-command model pull and serve
    - GGUF quantization (Q4_K_M, Q5_K_M, Q8_0)
    - CPU and GPU inference
    - OpenAI-compatible API (/v1/chat/completions)
    - Native API (/api/chat, /api/generate)
    - Model management (/api/pull, /api/delete)
    """

    def __init__(self, timeout: float = 120.0):
        super().__init__("ollama", timeout)

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> ChatCompletionResponse:
        # Use Ollama's OpenAI-compatible endpoint
        payload = {
            "model": self._resolve_ollama_model(model),
            "messages": self._build_messages_payload(request),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens or 2048,
            },
        }

        if request.stop:
            payload["options"]["stop"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.top_k > 0:
            payload["options"]["top_k"] = request.top_k
        if request.repetition_penalty != 1.0:
            payload["options"]["repeat_penalty"] = request.repetition_penalty

        resp = await self._client.post(
            f"{endpoint.base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        usage_data = data.get("usage", {})

        # Ollama returns eval_count and prompt_eval_count
        prompt_tokens = usage_data.get("prompt_tokens", data.get("prompt_eval_count", 0))
        completion_tokens = usage_data.get("completion_tokens", data.get("eval_count", 0))

        return ChatCompletionResponse(
            model=model.model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=message.get("content", ""),
                    ),
                    finish_reason="stop" if data.get("done", True) else "length",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        payload = {
            "model": self._resolve_ollama_model(model),
            "messages": self._build_messages_payload(request),
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens or 2048,
            },
        }

        if request.stop:
            payload["options"]["stop"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        async with self._client.stream(
            "POST",
            f"{endpoint.base_url}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                content = message.get("content", "")
                done = data.get("done", False)

                yield ChatCompletionStreamResponse(
                    model=model.model_id,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=ChoiceDelta(
                                role="assistant" if not done else None,
                                content=content if not done else None,
                            ),
                            finish_reason="stop" if done else None,
                        )
                    ],
                )

    async def embedding(
        self,
        request: EmbeddingRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> EmbeddingResponse:
        inputs = request.input if isinstance(request.input, list) else [request.input]

        payload = {
            "model": self._resolve_ollama_model(model),
            "input": inputs,
        }

        resp = await self._client.post(
            f"{endpoint.base_url}/api/embed",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings_data = data.get("embeddings", [])
        return EmbeddingResponse(
            data=[
                EmbeddingData(index=i, embedding=emb)
                for i, emb in enumerate(embeddings_data)
            ],
            model=model.model_id,
            usage=UsageInfo(prompt_tokens=data.get("prompt_eval_count", 0)),
        )

    async def load_model(self, request: ModelLoadRequest, endpoint: Any) -> bool:
        """Pull a model via Ollama's /api/pull endpoint."""
        payload = {
            "name": request.model_id,
            "stream": False,
        }
        try:
            resp = await self._client.post(
                f"{endpoint.base_url}/api/pull",
                json=payload,
                timeout=600.0,
            )
            resp.raise_for_status()
            logger.info("Ollama model pulled", model_id=request.model_id)
            return True
        except Exception as e:
            logger.error("Ollama model pull failed", model_id=request.model_id, error=str(e))
            return False

    async def unload_model(self, model_id: str, endpoint: Any) -> bool:
        """Delete a model via Ollama's /api/delete endpoint."""
        try:
            resp = await self._client.delete(
                f"{endpoint.base_url}/api/delete",
                json={"name": model_id},
            )
            return resp.status_code < 400
        except Exception as e:
            logger.error("Ollama model delete failed", model_id=model_id, error=str(e))
            return False

    def _resolve_ollama_model(self, model: ModelInfo) -> str:
        """Map model info to Ollama model name."""
        ollama_map = {
            "llama-3.1-8b-instruct": "llama3.1:8b",
            "llama-3.1-70b-instruct": "llama3.1:70b",
            "qwen2.5-72b-instruct": "qwen2.5:72b",
            "deepseek-coder-v2": "deepseek-coder-v2",
        }
        return ollama_map.get(model.model_id, model.source.split("/")[-1].lower())