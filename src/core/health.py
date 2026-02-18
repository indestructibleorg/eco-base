"""
Health Checker - Continuous engine health monitoring.
Probes all registered engines and updates their status for routing decisions.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import get_settings
from src.core.registry import ModelRegistry
from src.schemas.inference import EngineStatus, PlatformHealth
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

logger = get_logger("superai.health")


class HealthChecker:
    """
    Periodically probes inference engines and aggregates platform health.
    Supports OpenAI-compatible /health and /v1/models endpoints.
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._settings = get_settings()
        self._metrics = get_metrics()
        self._engine_statuses: Dict[str, EngineStatus] = {}
        self._start_time = time.time()
        self._total_requests = 0
        self._active_connections = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self, interval: float = 15.0) -> None:
        """Start periodic health checking."""
        self._http_client = httpx.AsyncClient(timeout=5.0)
        self._running = True
        self._task = asyncio.create_task(self._check_loop(interval))
        logger.info("Health checker started", interval_seconds=interval)

    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Health checker stopped")

    async def _check_loop(self, interval: float) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._check_all_engines()
            except Exception as e:
                logger.error("Health check cycle failed", error=str(e))
            await asyncio.sleep(interval)

    async def _check_all_engines(self) -> None:
        """Check all engine endpoints concurrently."""
        s = self._settings
        engines = {
            "vllm": (s.vllm_host, s.vllm_port),
            "tgi": (s.tgi_host, s.tgi_port),
            "sglang": (s.sglang_host, s.sglang_port),
            "ollama": (s.ollama_host, s.ollama_port),
            "tensorrt-llm": (s.tensorrt_host, s.tensorrt_port),
            "lmdeploy": (s.lmdeploy_host, s.lmdeploy_port),
            "deepspeed": (s.deepspeed_host, s.deepspeed_port),
        }

        tasks = [
            self._check_engine(name, host, port)
            for name, (host, port) in engines.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (name, _), result in zip(engines.items(), results):
            if isinstance(result, Exception):
                self._engine_statuses[name] = EngineStatus(
                    engine=name, status="unhealthy"
                )
                self._metrics.engine_status.labels(engine=name).set(0)
            elif isinstance(result, EngineStatus):
                self._engine_statuses[name] = result
                health_val = 1 if result.status == "healthy" else 0
                self._metrics.engine_status.labels(engine=name).set(health_val)

    async def _check_engine(
        self, name: str, host: str, port: int
    ) -> EngineStatus:
        """Check a single engine endpoint."""
        base_url = f"http://{host}:{port}"
        status = EngineStatus(engine=name, status="unknown")

        # Try multiple health endpoints
        health_endpoints = ["/health", "/v1/models", "/api/tags", "/"]
        for endpoint in health_endpoints:
            try:
                resp = await self._http_client.get(f"{base_url}{endpoint}")
                if resp.status_code < 500:
                    status.status = "healthy"

                    # Try to extract model list
                    if endpoint == "/v1/models":
                        try:
                            data = resp.json()
                            if "data" in data:
                                status.loaded_models = [
                                    m.get("id", "") for m in data["data"]
                                ]
                        except Exception:
                            pass

                    # Ollama-specific
                    elif endpoint == "/api/tags":
                        try:
                            data = resp.json()
                            if "models" in data:
                                status.loaded_models = [
                                    m.get("name", "") for m in data["models"]
                                ]
                        except Exception:
                            pass

                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
            except Exception as e:
                logger.debug(
                    "Health check endpoint failed",
                    engine=name,
                    endpoint=endpoint,
                    error=str(e),
                )
                continue

        if status.status == "unknown":
            status.status = "unhealthy"

        # Try GPU metrics endpoint
        try:
            resp = await self._http_client.get(f"{base_url}/metrics")
            if resp.status_code == 200:
                self._parse_gpu_metrics(name, resp.text)
        except Exception:
            pass

        return status

    def _parse_gpu_metrics(self, engine: str, metrics_text: str) -> None:
        """Parse Prometheus metrics for GPU info."""
        for line in metrics_text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            try:
                if "gpu_memory_used" in line.lower() or "vllm:gpu_cache" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        value = float(parts[-1])
                        self._metrics.gpu_memory_used.labels(
                            engine=engine, gpu_id="0"
                        ).set(value)
            except (ValueError, IndexError):
                continue

    async def get_platform_health(self) -> PlatformHealth:
        """Get aggregated platform health."""
        engine_list = list(self._engine_statuses.values())
        healthy_count = sum(1 for e in engine_list if e.status == "healthy")
        total = len(engine_list)

        if total == 0:
            overall = "unhealthy"
        elif healthy_count == total:
            overall = "healthy"
        elif healthy_count > 0:
            overall = "degraded"
        else:
            overall = "unhealthy"

        return PlatformHealth(
            status=overall,
            version=self._settings.app_version,
            uptime_seconds=time.time() - self._start_time,
            engines=engine_list,
            total_requests_served=self._total_requests,
            active_connections=self._active_connections,
        )

    async def get_engine_health(self, engine: str) -> Optional[EngineStatus]:
        """Get health of a specific engine."""
        return self._engine_statuses.get(engine)

    def is_engine_healthy(self, engine: str) -> bool:
        """Quick check if an engine is healthy."""
        status = self._engine_statuses.get(engine)
        return status is not None and status.status == "healthy"

    def increment_requests(self) -> None:
        self._total_requests += 1

    def set_active_connections(self, count: int) -> None:
        self._active_connections = count