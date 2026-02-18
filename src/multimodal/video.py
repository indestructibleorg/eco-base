"""
Video Understanding Pipeline
Frame extraction, temporal analysis, video QA.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger("superai.multimodal.video")


class VideoUnderstandingPipeline:
    """
    Pipeline for video understanding inference.

    Approaches:
    - Frame sampling + VLM analysis (Qwen-VL, InternVL)
    - Dedicated video models (VideoLLaMA, Video-ChatGPT)
    - Temporal reasoning across sampled frames
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=300.0)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def analyze_video(
        self,
        video_url: Optional[str] = None,
        video_base64: Optional[str] = None,
        frames: Optional[List[str]] = None,
        prompt: str = "Describe what happens in this video.",
        model: Optional[str] = None,
        max_frames: int = 16,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Analyze video content.

        Supports:
        - Direct video URL (for models with native video support)
        - Pre-extracted frames as base64 images
        - Automatic frame sampling from video

        Args:
            video_url: URL of video file
            video_base64: Base64-encoded video
            frames: List of base64-encoded frame images
            prompt: Analysis prompt
            model: VLM model to use
            max_frames: Maximum frames to sample
            max_tokens: Maximum response tokens
        """
        target_model = model or self._settings.vision_model
        engine_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}"

        content_parts = []

        if video_url:
            # For models with native video support (Qwen2.5-VL)
            content_parts.append({
                "type": "video_url",
                "video_url": {"url": video_url},
            })
        elif frames:
            # Use pre-extracted frames
            for frame_b64 in frames[:max_frames]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                })
        elif video_base64:
            content_parts.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
            })
        else:
            raise ValueError("Provide video_url, video_base64, or frames")

        content_parts.append({"type": "text", "text": prompt})

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
            "frames_analyzed": len([p for p in content_parts if p["type"] != "text"]),
            "usage": data.get("usage", {}),
        }

    async def extract_keyframes(
        self,
        video_url: str,
        num_frames: int = 8,
        strategy: str = "uniform",
    ) -> List[str]:
        """
        Extract keyframes from video for analysis.

        Strategies:
        - uniform: Evenly spaced frames
        - scene_change: Frames at scene boundaries
        - motion: Frames with highest motion
        """
        # Delegate to a frame extraction service
        payload = {
            "video_url": video_url,
            "num_frames": num_frames,
            "strategy": strategy,
        }

        engine_url = f"http://{self._settings.vllm_host}:8191"
        try:
            resp = await self._client.post(
                f"{engine_url}/extract_frames",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("frames", [])
        except Exception as e:
            logger.warning("Frame extraction service unavailable", error=str(e))
            return []

    async def video_qa(
        self,
        video_url: str,
        question: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a question about a video."""
        frames = await self.extract_keyframes(video_url, num_frames=16)

        if frames:
            return await self.analyze_video(
                frames=frames,
                prompt=f"Based on the video frames, answer: {question}",
                model=model,
            )
        else:
            return await self.analyze_video(
                video_url=video_url,
                prompt=question,
                model=model,
            )