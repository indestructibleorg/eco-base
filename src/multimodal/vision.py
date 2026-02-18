"""
Vision-Language Model Pipeline
Supports Qwen-VL, LLaVA, InternVL via vLLM/SGLang backends.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import get_settings
from src.schemas.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    ContentPart,
)
from src.utils.logging import get_logger

logger = get_logger("superai.multimodal.vision")


class VisionLanguagePipeline:
    """
    Pipeline for vision-language model inference.

    Supported models:
    - Qwen2.5-VL: 256K context, dynamic resolution, video understanding
    - LLaVA-NeXT: Strong visual reasoning, SGLang optimized
    - InternVL3.5: Multi-modal understanding, exceeds GPT-4o on some benchmarks

    Deployment via vLLM or SGLang with multimodal support.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=120.0)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def analyze_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        prompt: str = "Describe this image in detail.",
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Analyze an image using a vision-language model.

        Args:
            image_url: URL of the image to analyze
            image_base64: Base64-encoded image data
            prompt: Text prompt for the analysis
            model: Model to use (defaults to settings.vision_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        if not image_url and not image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")

        image_content: Dict[str, Any] = {}
        if image_url:
            image_content = {"type": "image_url", "image_url": {"url": image_url}}
        elif image_base64:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        target_model = model or self._settings.vision_model
        engine_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}"

        payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = await self._client.post(
            f"{engine_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "model": target_model,
            "analysis": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
        }

    async def analyze_multiple_images(
        self,
        images: List[Dict[str, str]],
        prompt: str = "Compare and describe these images.",
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Analyze multiple images in a single request."""
        content_parts = []
        for img in images:
            if "url" in img:
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": img["url"]}}
                )
            elif "base64" in img:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"},
                })
        content_parts.append({"type": "text", "text": prompt})

        target_model = model or self._settings.vision_model
        engine_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}"

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        resp = await self._client.post(
            f"{engine_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "model": target_model,
            "analysis": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
        }

    async def document_ocr(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Extract text from document images using VLM."""
        prompt = (
            f"Extract all text from this document image. "
            f"Maintain the original formatting and structure. "
            f"Language hint: {language}. "
            f"Output the extracted text only, no explanations."
        )
        return await self.analyze_image(
            image_url=image_url,
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=8192,
            temperature=0.1,
        )