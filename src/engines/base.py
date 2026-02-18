"""
Base Engine Adapter - Abstract interface for all inference engine adapters.
"""
from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from src.schemas.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from src.schemas.models import ModelInfo, ModelLoadRequest
from src.utils.logging import get_logger

logger = get_logger("superai.engines.base")


class BaseEngineAdapter(abc.ABC):
    """
    Abstract base class for inference engine adapters.
    Each adapter translates the unified OpenAI-compatible API to
    the engine's native protocol.
    """

    def __init__(self, engine_name: str, timeout: float = 120.0):
        self.engine_name = engine_name
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize HTTP client and connections."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
        logger.info("Engine adapter initialized", engine=self.engine_name)

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
        logger.info("Engine adapter shut down", engine=self.engine_name)

    @abc.abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> ChatCompletionResponse:
        """Execute a chat completion request."""
        ...

    @abc.abstractmethod
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Execute a streaming chat completion request."""
        ...

    async def completion(
        self,
        request: CompletionRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> CompletionResponse:
        """Execute a legacy completion request. Default: convert to chat."""
        from src.schemas.inference import ChatMessage, ChatRole

        chat_req = ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role=ChatRole.USER, content=request.prompt if isinstance(request.prompt, str) else request.prompt[0])],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
            seed=request.seed,
        )
        chat_resp = await self.chat_completion(chat_req, model, endpoint)

        from src.schemas.inference import CompletionChoice
        choices = [
            CompletionChoice(
                index=c.index,
                text=c.message.content if isinstance(c.message.content, str) else "",
                finish_reason=c.finish_reason,
            )
            for c in chat_resp.choices
        ]
        return CompletionResponse(
            id=chat_resp.id.replace("chatcmpl", "cmpl"),
            model=chat_resp.model,
            choices=choices,
            usage=chat_resp.usage,
        )

    async def embedding(
        self,
        request: EmbeddingRequest,
        model: ModelInfo,
        endpoint: Any,
    ) -> EmbeddingResponse:
        """Execute an embedding request."""
        raise NotImplementedError(
            f"Engine '{self.engine_name}' does not support embeddings"
        )

    async def load_model(self, request: ModelLoadRequest, endpoint: Any) -> bool:
        """Load a model onto the engine."""
        raise NotImplementedError(
            f"Engine '{self.engine_name}' does not support dynamic model loading"
        )

    async def unload_model(self, model_id: str, endpoint: Any) -> bool:
        """Unload a model from the engine."""
        raise NotImplementedError(
            f"Engine '{self.engine_name}' does not support dynamic model unloading"
        )

    def _build_messages_payload(self, request: ChatCompletionRequest) -> List[Dict]:
        """Convert ChatMessage list to dict payload."""
        messages = []
        for msg in request.messages:
            m: Dict[str, Any] = {"role": msg.role.value}
            if isinstance(msg.content, str):
                m["content"] = msg.content
            elif isinstance(msg.content, list):
                m["content"] = [
                    part.model_dump(exclude_none=True) for part in msg.content
                ]
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            messages.append(m)
        return messages

    def _build_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Extract common generation parameters."""
        params: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "n": request.n,
        }
        if request.stop:
            params["stop"] = request.stop
        if request.presence_penalty != 0.0:
            params["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty != 0.0:
            params["frequency_penalty"] = request.frequency_penalty
        if request.seed is not None:
            params["seed"] = request.seed
        if request.top_k > 0:
            params["top_k"] = request.top_k
        if request.response_format:
            params["response_format"] = request.response_format.model_dump(exclude_none=True)
        if request.tools:
            params["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        return params