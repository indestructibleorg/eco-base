"""FastAPI application for the Runtime Platform.

Exposes endpoints for workflow management, ETL pipeline execution,
scheduler registration, and health checks.
"""

from __future__ import annotations

import uuid
from typing import Any

from runtime_platform.domain.entities import ETLPipeline, Job, Step, Workflow
from runtime_platform.domain.value_objects import JobStatus, StepType, ScheduleFrequency
from runtime_platform.engines.workflow_engine import WorkflowEngine
from runtime_platform.engines.etl_engine import ETLEngine
from runtime_platform.engines.scheduler import Scheduler


# -- In-memory stores --------------------------------------------------------

_workflows: dict[str, Workflow] = {}
_workflow_engine = WorkflowEngine()
_etl_engine = ETLEngine()
_scheduler = Scheduler()


def _reset_state() -> None:
    """Reset all in-memory state (used by tests)."""
    global _workflows, _workflow_engine, _etl_engine, _scheduler
    _workflows.clear()
    _workflow_engine = WorkflowEngine()
    _etl_engine = ETLEngine()
    _scheduler = Scheduler()


# -- Application factory -----------------------------------------------------

def create_app() -> Any:
    """Build and return the FastAPI application.

    Imports FastAPI lazily so the module can be loaded without FastAPI installed
    (the domain, engines, and tests don't require it).
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise RuntimeError(
            "FastAPI is required for the presentation layer. "
            "Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Runtime Platform",
        version="1.0.0",
        description="Execution engine, ETL pipeline orchestration, workflow management, and task scheduling.",
    )

    # -- Request / Response models -------------------------------------------

    class StepRequest(BaseModel):
        name: str
        type: str = "shell"
        command: str
        timeout: float = 60.0

    class JobRequest(BaseModel):
        name: str
        steps: list[StepRequest] = []
        depends_on: list[str] = []

    class WorkflowRequest(BaseModel):
        name: str
        jobs: list[JobRequest] = []

    class ETLRequest(BaseModel):
        source: str
        target: str
        transforms: list[str] = []

    class SchedulerRegisterRequest(BaseModel):
        name: str
        cron_expr: str
        frequency: str = "cron"

    # -- Endpoints -----------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
            "platform": "runtime-platform",
            "version": "1.0.0",
        }

    @app.post("/workflows")
    async def create_and_run_workflow(req: WorkflowRequest) -> dict[str, Any]:
        workflow = Workflow(
            id=f"wf-{uuid.uuid4().hex[:8]}",
            name=req.name,
            jobs=[
                Job(
                    id=f"job-{uuid.uuid4().hex[:8]}",
                    name=jr.name,
                    steps=[
                        Step(
                            name=sr.name,
                            type=StepType(sr.type),
                            command=sr.command,
                            timeout=sr.timeout,
                        )
                        for sr in jr.steps
                    ],
                    depends_on=jr.depends_on,
                )
                for jr in req.jobs
            ],
        )

        _workflows[workflow.id] = workflow

        try:
            result = await _workflow_engine.run_workflow(workflow)
            return {"workflow_id": workflow.id, **result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str) -> dict[str, Any]:
        wf = _workflows.get(workflow_id)
        if wf is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        return {
            "id": wf.id,
            "name": wf.name,
            "status": wf.status.value,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "status": j.status.value,
                    "depends_on": j.depends_on,
                    "steps": [
                        {"name": s.name, "type": s.type.value, "command": s.command}
                        for s in j.steps
                    ],
                }
                for j in wf.jobs
            ],
        }

    @app.post("/etl/run")
    async def run_etl_pipeline(req: ETLRequest) -> dict[str, Any]:
        """Create and run a simple ETL pipeline with identity functions."""
        # For the API, we provide simple pass-through functions
        async def extract(source: str) -> list[dict]:
            return [{"source": source, "row": i} for i in range(10)]

        async def transform(data: list) -> list:
            return [dict(row, transformed=True) for row in data]

        async def load(target: str, data: list) -> int:
            return len(data)

        pipeline = ETLPipeline(
            source=req.source,
            target=req.target,
            transforms=req.transforms,
            extract_fn=extract,
            transform_fns=[transform],
            load_fn=load,
        )

        try:
            result = await _etl_engine.run_pipeline(pipeline)
            return {
                "pipeline_id": result.pipeline_id,
                "success": result.success,
                "rows_extracted": result.rows_extracted,
                "rows_transformed": result.rows_transformed,
                "rows_loaded": result.rows_loaded,
                "duration_seconds": result.total_duration,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/scheduler/register")
    async def register_scheduler_job(req: SchedulerRegisterRequest) -> dict[str, Any]:
        """Register a no-op scheduled job (real callable would be injected via DI)."""
        async def noop() -> str:
            return f"Executed {req.name}"

        try:
            freq = ScheduleFrequency(req.frequency)
            job = _scheduler.register_job(
                req.name, req.cron_expr, noop, frequency=freq
            )
            return {
                "name": job.name,
                "cron_expr": job.cron_expr,
                "frequency": job.frequency.value,
                "enabled": job.enabled,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app
