"""
Inference engine adapters.
Each adapter translates OpenAI-compatible requests to engine-specific protocols.
"""
from .base import BaseEngineAdapter
from .vllm_adapter import VLLMAdapter
from .tgi_adapter import TGIAdapter
from .sglang_adapter import SGLangAdapter
from .ollama_adapter import OllamaAdapter
from .tensorrt_adapter import TensorRTAdapter
from .lmdeploy_adapter import LMDeployAdapter

__all__ = [
    "BaseEngineAdapter",
    "VLLMAdapter",
    "TGIAdapter",
    "SGLangAdapter",
    "OllamaAdapter",
    "TensorRTAdapter",
    "LMDeployAdapter",
]