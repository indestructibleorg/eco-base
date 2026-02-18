"""
Image Generation Pipeline
Stable Diffusion XL, FLUX via ComfyUI/diffusers backends.
"""
from __future__ import annotations

import base64
import io
import uuid
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger("superai.multimodal.image_gen")


class ImageGenerationPipeline:
    """
    Pipeline for image generation inference.

    Supported backends:
    - ComfyUI: Node-based workflow, production-grade, 10x faster than A1111
    - diffusers: HuggingFace native, flexible pipeline composition
    - AUTOMATIC1111: Broad extension ecosystem

    Models: SDXL, FLUX.1, Stable Diffusion 3
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=300.0)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: int = -1,
        num_images: int = 1,
        scheduler: str = "euler_a",
        backend: str = "comfyui",
    ) -> Dict[str, Any]:
        """
        Generate images from text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in generation
            model: Checkpoint model to use
            width/height: Output dimensions
            steps: Denoising steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (-1 for random)
            num_images: Number of images to generate
            scheduler: Sampling scheduler
            backend: 'comfyui' or 'diffusers'
        """
        target_model = model or self._settings.image_gen_model

        if backend == "comfyui":
            return await self._generate_comfyui(
                prompt, negative_prompt, target_model,
                width, height, steps, cfg_scale, seed, num_images, scheduler,
            )
        else:
            return await self._generate_diffusers(
                prompt, negative_prompt, target_model,
                width, height, steps, cfg_scale, seed, num_images,
            )

    async def _generate_comfyui(
        self,
        prompt: str,
        negative_prompt: str,
        model: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        num_images: int,
        scheduler: str,
    ) -> Dict[str, Any]:
        """Generate via ComfyUI API with workflow."""
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed if seed >= 0 else int(uuid.uuid4().int % 2**32),
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": scheduler,
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": model},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": num_images,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "superai", "images": ["8", 0]},
            },
        }

        # Queue the workflow
        comfyui_url = f"http://{self._settings.vllm_host}:8188"
        client_id = str(uuid.uuid4())

        resp = await self._client.post(
            f"{comfyui_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "prompt_id": data.get("prompt_id", ""),
            "model": model,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "scheduler": scheduler,
            },
            "status": "queued",
            "num_images": num_images,
        }

    async def _generate_diffusers(
        self,
        prompt: str,
        negative_prompt: str,
        model: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        num_images: int,
    ) -> Dict[str, Any]:
        """Generate via diffusers-based API server."""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "seed": seed if seed >= 0 else None,
            "num_images": num_images,
        }

        diffusers_url = f"http://{self._settings.vllm_host}:8190"
        resp = await self._client.post(
            f"{diffusers_url}/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def img2img(
        self,
        prompt: str,
        image_base64: str,
        strength: float = 0.75,
        negative_prompt: str = "",
        steps: int = 30,
        cfg_scale: float = 7.0,
    ) -> Dict[str, Any]:
        """Image-to-image generation."""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_image": image_base64,
            "strength": strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
        }

        diffusers_url = f"http://{self._settings.vllm_host}:8190"
        resp = await self._client.post(
            f"{diffusers_url}/img2img",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()