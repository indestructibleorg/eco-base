"""
Audio Pipeline - ASR (Whisper) + TTS (CosyVoice/Fish-Speech).
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger("superai.multimodal.audio")


class AudioPipeline:
    """
    Pipeline for audio inference.

    ASR (Speech-to-Text):
    - Whisper Large V3 Turbo: sub-second latency, 100+ languages
    - WhisperKit: device-side real-time inference

    TTS (Text-to-Speech):
    - CosyVoice: multi-language, streaming generation
    - Fish-Speech: high-quality voice cloning
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=120.0)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def transcribe(
        self,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_url: URL of audio file
            audio_base64: Base64-encoded audio
            audio_bytes: Raw audio bytes
            model: Whisper model variant
            language: Source language hint (ISO 639-1)
            task: 'transcribe' or 'translate' (to English)
            response_format: 'json', 'text', 'srt', 'vtt', 'verbose_json'
            timestamp_granularities: ['word', 'segment']
        """
        target_model = model or self._settings.whisper_model
        engine_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}"

        # Build multipart form data for OpenAI-compatible /v1/audio/transcriptions
        files = {}
        data = {
            "model": target_model,
            "response_format": response_format,
        }

        if language:
            data["language"] = language
        if task == "translate":
            data["task"] = "translate"
        if timestamp_granularities:
            data["timestamp_granularities[]"] = timestamp_granularities

        if audio_bytes:
            files["file"] = ("audio.wav", audio_bytes, "audio/wav")
        elif audio_base64:
            audio_data = base64.b64decode(audio_base64)
            files["file"] = ("audio.wav", audio_data, "audio/wav")
        elif audio_url:
            # Download audio first
            audio_resp = await self._client.get(audio_url)
            audio_resp.raise_for_status()
            files["file"] = ("audio.wav", audio_resp.content, "audio/wav")
        else:
            raise ValueError("One of audio_url, audio_base64, or audio_bytes required")

        resp = await self._client.post(
            f"{engine_url}/v1/audio/transcriptions",
            data=data,
            files=files,
        )
        resp.raise_for_status()

        if response_format == "text":
            return {"text": resp.text, "model": target_model}
        return resp.json()

    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "default",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice preset or speaker ID
            speed: Speech speed multiplier
            response_format: Output format (mp3, wav, opus, flac)
        """
        target_model = model or self._settings.tts_model
        engine_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}"

        payload = {
            "model": target_model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
        }

        resp = await self._client.post(
            f"{engine_url}/v1/audio/speech",
            json=payload,
        )
        resp.raise_for_status()

        audio_b64 = base64.b64encode(resp.content).decode("utf-8")
        return {
            "audio_base64": audio_b64,
            "format": response_format,
            "model": target_model,
            "size_bytes": len(resp.content),
        }

    async def voice_clone(
        self,
        text: str,
        reference_audio_base64: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clone a voice from reference audio and synthesize new text."""
        target_model = model or self._settings.tts_model

        payload = {
            "model": target_model,
            "input": text,
            "reference_audio": reference_audio_base64,
            "task": "voice_clone",
        }

        engine_url = f"http://{self._settings.vllm_host}:{self._settings.vllm_port}"
        resp = await self._client.post(
            f"{engine_url}/v1/audio/speech",
            json=payload,
        )
        resp.raise_for_status()

        audio_b64 = base64.b64encode(resp.content).decode("utf-8")
        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "model": target_model,
        }