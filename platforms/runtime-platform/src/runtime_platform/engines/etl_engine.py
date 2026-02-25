"""ETLEngine — extract, transform, load pipeline execution.

Runs a three-phase pipeline where each phase is an async callable.
Tracks row counts, timing, and errors through an ETLResult dataclass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from platform_shared.protocols.engine import EngineStatus

from runtime_platform.domain.entities import ETLPipeline
from runtime_platform.domain.value_objects import ETLPhase, JobStatus
from runtime_platform.domain.events import ETLCompleted
from runtime_platform.domain.exceptions import ETLPipelineError


@dataclass(slots=True)
class ETLResult:
    """Outcome of an ETL pipeline run."""

    pipeline_id: str
    success: bool = False
    rows_extracted: int = 0
    rows_transformed: int = 0
    rows_loaded: int = 0
    extract_duration: float = 0.0
    transform_duration: float = 0.0
    load_duration: float = 0.0
    total_duration: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return self.rows_loaded


class ETLEngine:
    """Executes ETL pipelines through extract -> transform -> load phases.

    Each phase receives data from the previous phase. The extract phase receives
    the pipeline source, transforms chain data through, and load writes to target.
    """

    def __init__(self) -> None:
        self._status = EngineStatus.IDLE
        self._events: list[Any] = []
        self._results: dict[str, ETLResult] = {}

    @property
    def name(self) -> str:
        return "etl-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def events(self) -> list[Any]:
        return list(self._events)

    @property
    def results(self) -> dict[str, ETLResult]:
        return dict(self._results)

    async def start(self) -> None:
        self._status = EngineStatus.RUNNING

    async def stop(self) -> None:
        self._status = EngineStatus.STOPPED

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Protocol-compatible execute — wraps run_pipeline."""
        pipeline = payload.get("pipeline")
        if not isinstance(pipeline, ETLPipeline):
            raise ETLPipelineError("Payload must contain a 'pipeline' key with an ETLPipeline")
        result = await self.run_pipeline(pipeline)
        return {
            "pipeline_id": result.pipeline_id,
            "success": result.success,
            "rows_loaded": result.rows_loaded,
        }

    async def run_pipeline(self, pipeline: ETLPipeline) -> ETLResult:
        """Execute the full extract -> transform -> load pipeline."""
        self._status = EngineStatus.RUNNING
        result = ETLResult(pipeline_id=pipeline.id)
        total_start = time.monotonic()

        try:
            # --- EXTRACT ---
            pipeline.status = ETLPhase.EXTRACT
            data = await self._run_extract(pipeline, result)

            # --- TRANSFORM ---
            pipeline.status = ETLPhase.TRANSFORM
            data = await self._run_transforms(pipeline, result, data)

            # --- LOAD ---
            pipeline.status = ETLPhase.LOAD
            await self._run_load(pipeline, result, data)

            result.success = True
            pipeline.status = JobStatus.COMPLETED

        except ETLPipelineError:
            result.success = False
            pipeline.status = JobStatus.FAILED
            raise
        except Exception as exc:
            result.success = False
            result.errors.append(str(exc))
            pipeline.status = JobStatus.FAILED
            raise ETLPipelineError(
                str(exc), pipeline_id=pipeline.id, phase="unknown"
            ) from exc
        finally:
            result.total_duration = time.monotonic() - total_start
            self._results[pipeline.id] = result
            self._status = EngineStatus.IDLE

            self._events.append(ETLCompleted(
                pipeline_id=pipeline.id,
                source=pipeline.source,
                target=pipeline.target,
                rows_extracted=result.rows_extracted,
                rows_transformed=result.rows_transformed,
                rows_loaded=result.rows_loaded,
                success=result.success,
                error="; ".join(result.errors) if result.errors else "",
                duration_seconds=result.total_duration,
            ))

        return result

    async def _run_extract(self, pipeline: ETLPipeline, result: ETLResult) -> Any:
        """Run the extract phase."""
        if pipeline.extract_fn is None:
            raise ETLPipelineError(
                "No extract function configured",
                pipeline_id=pipeline.id,
                phase="extract",
            )

        start = time.monotonic()
        try:
            data = await pipeline.extract_fn(pipeline.source)
            result.extract_duration = time.monotonic() - start

            # Track row count if data is iterable with length
            if isinstance(data, (list, tuple)):
                result.rows_extracted = len(data)
            elif hasattr(data, "__len__"):
                result.rows_extracted = len(data)
            else:
                result.rows_extracted = 1

            return data

        except Exception as exc:
            result.extract_duration = time.monotonic() - start
            result.errors.append(f"Extract failed: {exc}")
            raise ETLPipelineError(
                f"Extract phase failed: {exc}",
                pipeline_id=pipeline.id,
                phase="extract",
            ) from exc

    async def _run_transforms(
        self, pipeline: ETLPipeline, result: ETLResult, data: Any
    ) -> Any:
        """Chain data through all transform functions."""
        start = time.monotonic()
        try:
            for i, transform_fn in enumerate(pipeline.transform_fns):
                data = await transform_fn(data)

            result.transform_duration = time.monotonic() - start

            if isinstance(data, (list, tuple)):
                result.rows_transformed = len(data)
            elif hasattr(data, "__len__"):
                result.rows_transformed = len(data)
            else:
                result.rows_transformed = 1

            return data

        except Exception as exc:
            result.transform_duration = time.monotonic() - start
            result.errors.append(f"Transform [{i}] failed: {exc}")
            raise ETLPipelineError(
                f"Transform phase failed at step {i}: {exc}",
                pipeline_id=pipeline.id,
                phase="transform",
            ) from exc

    async def _run_load(self, pipeline: ETLPipeline, result: ETLResult, data: Any) -> None:
        """Run the load phase."""
        if pipeline.load_fn is None:
            raise ETLPipelineError(
                "No load function configured",
                pipeline_id=pipeline.id,
                phase="load",
            )

        start = time.monotonic()
        try:
            load_result = await pipeline.load_fn(pipeline.target, data)
            result.load_duration = time.monotonic() - start

            # If load returns a count, use it; otherwise use transformed count
            if isinstance(load_result, int):
                result.rows_loaded = load_result
            elif isinstance(data, (list, tuple)):
                result.rows_loaded = len(data)
            elif hasattr(data, "__len__"):
                result.rows_loaded = len(data)
            else:
                result.rows_loaded = 1

        except Exception as exc:
            result.load_duration = time.monotonic() - start
            result.errors.append(f"Load failed: {exc}")
            raise ETLPipelineError(
                f"Load phase failed: {exc}",
                pipeline_id=pipeline.id,
                phase="load",
            ) from exc
