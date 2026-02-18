"""
Prometheus metrics collector for inference observability.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


class MetricsCollector:
    """Centralized Prometheus metrics for the SuperAI platform."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._init_metrics()

    def _init_metrics(self) -> None:
        # ── Platform info ────────────────────────────────────────
        self.platform_info = Info(
            "superai_platform",
            "Platform build information",
            registry=self.registry,
        )

        # ── Request metrics ──────────────────────────────────────
        self.requests_total = Counter(
            "superai_requests_total",
            "Total inference requests",
            ["engine", "model", "endpoint", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "superai_request_duration_seconds",
            "Request duration in seconds",
            ["engine", "model", "endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry,
        )

        self.time_to_first_token = Histogram(
            "superai_ttft_seconds",
            "Time to first token in seconds",
            ["engine", "model"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )

        self.tokens_generated = Counter(
            "superai_tokens_generated_total",
            "Total tokens generated",
            ["engine", "model", "token_type"],
            registry=self.registry,
        )

        self.tokens_per_second = Histogram(
            "superai_tokens_per_second",
            "Token generation throughput",
            ["engine", "model"],
            buckets=(1, 5, 10, 25, 50, 100, 200, 500, 1000),
            registry=self.registry,
        )

        # ── Queue metrics ────────────────────────────────────────
        self.queue_depth = Gauge(
            "superai_queue_depth",
            "Current request queue depth",
            ["engine"],
            registry=self.registry,
        )

        self.active_requests = Gauge(
            "superai_active_requests",
            "Currently processing requests",
            ["engine", "model"],
            registry=self.registry,
        )

        # ── Engine metrics ───────────────────────────────────────
        self.engine_status = Gauge(
            "superai_engine_status",
            "Engine health status (1=healthy, 0=unhealthy)",
            ["engine"],
            registry=self.registry,
        )

        self.gpu_utilization = Gauge(
            "superai_gpu_utilization_percent",
            "GPU utilization percentage",
            ["engine", "gpu_id"],
            registry=self.registry,
        )

        self.gpu_memory_used = Gauge(
            "superai_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["engine", "gpu_id"],
            registry=self.registry,
        )

        self.gpu_memory_total = Gauge(
            "superai_gpu_memory_total_bytes",
            "GPU total memory in bytes",
            ["engine", "gpu_id"],
            registry=self.registry,
        )

        # ── Model metrics ────────────────────────────────────────
        self.loaded_models = Gauge(
            "superai_loaded_models",
            "Number of currently loaded models",
            ["engine"],
            registry=self.registry,
        )

        self.model_load_duration = Histogram(
            "superai_model_load_duration_seconds",
            "Model loading time in seconds",
            ["engine", "model"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
            registry=self.registry,
        )

        # ── Error metrics ────────────────────────────────────────
        self.errors_total = Counter(
            "superai_errors_total",
            "Total errors",
            ["engine", "error_type"],
            registry=self.registry,
        )

        # ── Batch metrics ────────────────────────────────────────
        self.batch_size = Histogram(
            "superai_batch_size",
            "Batch sizes for batch inference",
            ["engine"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            registry=self.registry,
        )

    @contextmanager
    def track_request(
        self, engine: str, model: str, endpoint: str
    ) -> Generator[None, None, None]:
        """Context manager to track request duration and status."""
        self.active_requests.labels(engine=engine, model=model).inc()
        start = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            self.errors_total.labels(engine=engine, error_type="request_failed").inc()
            raise
        finally:
            duration = time.perf_counter() - start
            self.request_duration.labels(
                engine=engine, model=model, endpoint=endpoint
            ).observe(duration)
            self.requests_total.labels(
                engine=engine, model=model, endpoint=endpoint, status=status
            ).inc()
            self.active_requests.labels(engine=engine, model=model).dec()

    def export(self) -> bytes:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


# Singleton
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics