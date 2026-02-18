"""
SuperAI Platform - Central Configuration
Unified settings for all services via environment variables.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class InferenceEngine(str, Enum):
    VLLM = "vllm"
    TGI = "tgi"
    SGLANG = "sglang"
    OLLAMA = "ollama"
    TENSORRT_LLM = "tensorrt-llm"
    LMDEPLOY = "lmdeploy"
    DEEPSPEED = "deepspeed"
    LLAMACPP = "llama.cpp"


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────
    app_name: str = "SuperAI Platform"
    app_version: str = "1.0.0"
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # ── Security ─────────────────────────────────────────────────
    secret_key: str = Field(default="change-me-in-production-use-openssl-rand-hex-32")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440
    api_key_header: str = "X-API-Key"
    cors_origins: List[str] = ["*"]
    rate_limit_per_minute: int = 600

    # ── Database ─────────────────────────────────────────────────
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "superai"
    postgres_password: str = "superai_secret"
    postgres_db: str = "superai"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ────────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Inference Engines ────────────────────────────────────────
    default_engine: InferenceEngine = InferenceEngine.VLLM
    engine_timeout_seconds: int = 120
    max_concurrent_requests: int = 256
    request_queue_size: int = 4096

    # vLLM
    vllm_host: str = "vllm"
    vllm_port: int = 8001
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 32768
    vllm_tensor_parallel_size: int = 1

    # TGI
    tgi_host: str = "tgi"
    tgi_port: int = 8002
    tgi_max_input_length: int = 4096
    tgi_max_total_tokens: int = 8192

    # SGLang
    sglang_host: str = "sglang"
    sglang_port: int = 8003

    # Ollama
    ollama_host: str = "ollama"
    ollama_port: int = 11434

    # TensorRT-LLM
    tensorrt_host: str = "tensorrt-llm"
    tensorrt_port: int = 8004

    # LMDeploy
    lmdeploy_host: str = "lmdeploy"
    lmdeploy_port: int = 8005

    # DeepSpeed
    deepspeed_host: str = "deepspeed"
    deepspeed_port: int = 8006

    # ── Model Registry ───────────────────────────────────────────
    model_cache_dir: str = "/models"
    default_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_loaded_models: int = 10

    # ── Multimodal ───────────────────────────────────────────────
    vision_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    image_gen_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    whisper_model: str = "openai/whisper-large-v3-turbo"
    tts_model: str = "fishaudio/fish-speech-1.5"

    # ── RAG ──────────────────────────────────────────────────────
    vector_db_host: str = "milvus"
    vector_db_port: int = 19530
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5

    # ── Monitoring ───────────────────────────────────────────────
    prometheus_port: int = 9090
    grafana_port: int = 3000
    enable_tracing: bool = True
    jaeger_host: str = "jaeger"
    jaeger_port: int = 6831

    # ── Storage ──────────────────────────────────────────────────
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "superai-artifacts"

    @field_validator("vllm_gpu_memory_utilization")
    @classmethod
    def validate_gpu_util(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return v

    class Config:
        env_prefix = "SUPERAI_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()