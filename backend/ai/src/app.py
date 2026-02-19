"""IndestructibleEco AI Generation Service -- FastAPI + Uvicorn.

Runtime: Python 3.11 + FastAPI + Uvicorn
Ports: 8000 (gRPC internal) + 8001 (HTTP)
Vector alignment: quantum-bert-xxl-v1, dim 1024-4096, tol 0.0001-0.005
Queuing: Celery + Redis for async inference jobs
Engine management: 7 adapters with connection pools + circuit breakers
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .routes import router as api_router
from .governance import GovernanceEngine


def _resolve_engine_endpoints() -> Dict[str, str]:
    """Resolve engine endpoints from ECO_* env vars."""
    return {
        "vllm": os.environ.get("ECO_VLLM_URL", "http://localhost:8100"),
        "tgi": os.environ.get("ECO_TGI_URL", "http://localhost:8101"),
        "ollama": os.environ.get("ECO_OLLAMA_URL", "http://localhost:11434"),
        "sglang": os.environ.get("ECO_SGLANG_URL", "http://localhost:8102"),
        "tensorrt": os.environ.get("ECO_TENSORRT_URL", "http://localhost:8103"),
        "deepspeed": os.environ.get("ECO_DEEPSPEED_URL", "http://localhost:8104"),
        "lmdeploy": os.environ.get("ECO_LMDEPLOY_URL", "http://localhost:8105"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    app.state.governance = GovernanceEngine()

    from .services.engine_manager import EngineManager

    endpoints = _resolve_engine_endpoints()
    engine_mgr = EngineManager(
        endpoints=endpoints,
        failure_threshold=3,
        recovery_timeout=30.0,
    )
    await engine_mgr.initialize()
    app.state.engine_manager = engine_mgr

    yield

    await engine_mgr.shutdown()


app = FastAPI(
    title="IndestructibleEco AI Service",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health(request: Request):
    uptime = time.time() - request.app.state.start_time
    engine_mgr = getattr(request.app.state, "engine_manager", None)
    available_engines = engine_mgr.list_available_engines() if engine_mgr else []

    return {
        "status": "healthy",
        "service": "ai",
        "version": "1.0.0",
        "uri": "indestructibleeco://backend/ai/health",
        "urn": f"urn:indestructibleeco:backend:ai:health:{uuid.uuid1()}",
        "uptime_seconds": round(uptime, 2),
        "engines": available_engines,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/engines")
async def engine_health(request: Request):
    """Detailed engine health with circuit breaker states."""
    engine_mgr = getattr(request.app.state, "engine_manager", None)
    if not engine_mgr:
        return {"error": "engine manager not initialized"}
    return engine_mgr.get_health()


@app.get("/metrics")
async def metrics(request: Request):
    uptime = time.time() - request.app.state.start_time
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF)

    engine_mgr = getattr(request.app.state, "engine_manager", None)
    engine_lines = []
    if engine_mgr:
        for name, h in engine_mgr._health.items():
            engine_lines.extend([
                f'eco_engine_requests_total{{engine="{name}"}} {h.total_requests}',
                f'eco_engine_errors_total{{engine="{name}"}} {h.total_errors}',
                f'eco_engine_latency_ms{{engine="{name}"}} {h.latency_ms:.2f}',
                f'eco_engine_available{{engine="{name}"}} {1 if h.is_available else 0}',
            ])

    lines = [
        "# HELP eco_uptime_seconds AI service uptime in seconds",
        "# TYPE eco_uptime_seconds gauge",
        f"eco_uptime_seconds {uptime:.2f}",
        "# HELP eco_memory_maxrss_bytes Maximum resident set size",
        "# TYPE eco_memory_maxrss_bytes gauge",
        f"eco_memory_maxrss_bytes {mem.ru_maxrss * 1024}",
        "# HELP eco_engine_requests_total Total requests per engine",
        "# TYPE eco_engine_requests_total counter",
        "# HELP eco_engine_errors_total Total errors per engine",
        "# TYPE eco_engine_errors_total counter",
    ] + engine_lines

    return JSONResponse(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )
