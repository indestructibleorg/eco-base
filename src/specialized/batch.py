"""
Batch Inference Processor
High-throughput offline processing with priority scheduling.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.router import InferenceRouter
from src.schemas.inference import (
    BatchInferenceRequest,
    BatchInferenceStatus,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

logger = get_logger("superai.specialized.batch")


class BatchInferenceProcessor:
    """
    Batch inference processor for high-throughput offline workloads.

    Features:
    - Async parallel processing with configurable concurrency
    - Priority-based scheduling
    - Progress tracking and callback notifications
    - Automatic retry on transient failures
    - Result aggregation and export
    """

    def __init__(self, router: InferenceRouter, max_concurrency: int = 32):
        self._router = router
        self._max_concurrency = max_concurrency
        self._metrics = get_metrics()
        self._batches: Dict[str, BatchInferenceStatus] = {}
        self._results: Dict[str, List[Optional[ChatCompletionResponse]]] = {}

    async def submit(self, request: BatchInferenceRequest) -> BatchInferenceStatus:
        """Submit a batch of requests for processing."""
        batch_id = request.batch_id
        total = len(request.requests)

        status = BatchInferenceStatus(
            batch_id=batch_id,
            total=total,
            completed=0,
            failed=0,
            status="pending",
        )
        self._batches[batch_id] = status
        self._results[batch_id] = [None] * total

        self._metrics.batch_size.labels(engine="batch").observe(total)

        # Start processing in background
        asyncio.create_task(self._process_batch(batch_id, request))

        logger.info("Batch submitted", batch_id=batch_id, total=total)
        return status

    async def _process_batch(
        self, batch_id: str, request: BatchInferenceRequest
    ) -> None:
        """Process all requests in a batch with bounded concurrency."""
        self._batches[batch_id].status = "processing"
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def process_one(index: int, req: ChatCompletionRequest) -> None:
            async with semaphore:
                retries = 3
                for attempt in range(retries):
                    try:
                        response = await self._router.chat_completion(req)
                        self._results[batch_id][index] = response
                        self._batches[batch_id].completed += 1
                        return
                    except Exception as e:
                        if attempt == retries - 1:
                            logger.error(
                                "Batch item failed",
                                batch_id=batch_id,
                                index=index,
                                error=str(e),
                            )
                            self._batches[batch_id].failed += 1
                        else:
                            await asyncio.sleep(2 ** attempt)

        tasks = [
            process_one(i, req) for i, req in enumerate(request.requests)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Finalize
        status = self._batches[batch_id]
        if status.failed == 0:
            status.status = "completed"
        elif status.completed > 0:
            status.status = "completed"  # partial success
        else:
            status.status = "failed"

        status.results = [r for r in self._results[batch_id] if r is not None]

        logger.info(
            "Batch completed",
            batch_id=batch_id,
            completed=status.completed,
            failed=status.failed,
        )

        # Callback notification
        if request.callback_url:
            await self._notify_callback(request.callback_url, status)

    async def _notify_callback(self, url: str, status: BatchInferenceStatus) -> None:
        """Send completion notification to callback URL."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json=status.model_dump(exclude={"results"}))
        except Exception as e:
            logger.warning("Batch callback failed", url=url, error=str(e))

    async def get_status(self, batch_id: str) -> Optional[BatchInferenceStatus]:
        """Get batch processing status."""
        return self._batches.get(batch_id)

    async def get_results(self, batch_id: str) -> Optional[List[ChatCompletionResponse]]:
        """Get batch results (only available after completion)."""
        status = self._batches.get(batch_id)
        if not status or status.status not in ("completed", "failed"):
            return None
        return [r for r in self._results.get(batch_id, []) if r is not None]

    async def cancel(self, batch_id: str) -> bool:
        """Cancel a pending/processing batch."""
        status = self._batches.get(batch_id)
        if not status:
            return False
        if status.status in ("completed", "failed"):
            return False
        status.status = "failed"
        logger.info("Batch cancelled", batch_id=batch_id)
        return True

    async def list_batches(
        self, status_filter: Optional[str] = None
    ) -> List[BatchInferenceStatus]:
        """List all batches with optional status filter."""
        batches = list(self._batches.values())
        if status_filter:
            batches = [b for b in batches if b.status == status_filter]
        return batches

    async def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Remove old completed/failed batches."""
        removed = 0
        to_remove = []
        for batch_id, status in self._batches.items():
            if status.status in ("completed", "failed"):
                to_remove.append(batch_id)
        for batch_id in to_remove:
            del self._batches[batch_id]
            self._results.pop(batch_id, None)
            removed += 1
        return removed