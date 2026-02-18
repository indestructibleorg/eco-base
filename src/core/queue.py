"""
Request Queue - Redis-backed priority queue for inference requests.
Provides backpressure, priority scheduling, and batch aggregation.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

logger = get_logger("superai.queue")


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 5
    LOW = 8
    BATCH = 10


@dataclass
class QueuedRequest:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = Priority.NORMAL
    engine: Optional[str] = None
    model: str = "default"
    created_at: float = field(default_factory=time.time)
    timeout: float = 120.0
    callback: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.timeout

    def to_json(self) -> str:
        return json.dumps({
            "request_id": self.request_id,
            "payload": self.payload,
            "priority": self.priority,
            "engine": self.engine,
            "model": self.model,
            "created_at": self.created_at,
            "timeout": self.timeout,
            "callback": self.callback,
        })

    @classmethod
    def from_json(cls, data: str) -> QueuedRequest:
        d = json.loads(data)
        return cls(**d)


class RequestQueue:
    """
    Async priority queue with Redis backend for distributed deployments.
    Falls back to in-memory heapq when Redis is unavailable.
    """

    def __init__(self, redis_client: Optional[Any] = None):
        self._settings = get_settings()
        self._metrics = get_metrics()
        self._redis = redis_client
        self._local_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self._settings.request_queue_size
        )
        self._pending: Dict[str, QueuedRequest] = {}
        self._results: Dict[str, Any] = {}
        self._result_events: Dict[str, asyncio.Event] = {}
        self._running = False
        self._workers: List[asyncio.Task] = []

    @property
    def depth(self) -> int:
        return self._local_queue.qsize()

    async def enqueue(self, request: QueuedRequest) -> str:
        """Add a request to the queue. Returns request_id."""
        if self._local_queue.full():
            raise RuntimeError("Request queue is full. Try again later.")

        self._pending[request.request_id] = request
        self._result_events[request.request_id] = asyncio.Event()

        # Priority queue uses (priority, timestamp, request_id) for ordering
        await self._local_queue.put(
            (request.priority, request.created_at, request.request_id)
        )

        # Redis backup for distributed mode
        if self._redis:
            try:
                await self._redis.zadd(
                    "superai:queue",
                    {request.to_json(): request.priority + (request.created_at / 1e12)},
                )
            except Exception as e:
                logger.warning("Redis enqueue failed, using local only", error=str(e))

        self._metrics.queue_depth.labels(engine=request.engine or "any").inc()
        logger.debug(
            "Request enqueued",
            request_id=request.request_id,
            priority=request.priority,
            model=request.model,
            queue_depth=self.depth,
        )
        return request.request_id

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get the highest priority request from the queue."""
        try:
            priority, timestamp, request_id = await asyncio.wait_for(
                self._local_queue.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            return None

        request = self._pending.get(request_id)
        if not request:
            return None

        if request.is_expired:
            logger.warning("Request expired", request_id=request_id)
            self._complete(request_id, error="Request timed out in queue")
            return None

        self._metrics.queue_depth.labels(engine=request.engine or "any").dec()
        return request

    def _complete(self, request_id: str, result: Any = None, error: Optional[str] = None) -> None:
        """Mark a request as completed."""
        if error:
            self._results[request_id] = {"error": error}
        else:
            self._results[request_id] = result

        event = self._result_events.get(request_id)
        if event:
            event.set()

        self._pending.pop(request_id, None)

    async def complete(self, request_id: str, result: Any) -> None:
        """Complete a request with a result."""
        self._complete(request_id, result=result)

    async def fail(self, request_id: str, error: str) -> None:
        """Fail a request with an error."""
        self._complete(request_id, error=error)

    async def wait_for_result(self, request_id: str, timeout: float = 120.0) -> Any:
        """Wait for a request result."""
        event = self._result_events.get(request_id)
        if not event:
            raise ValueError(f"Unknown request: {request_id}")

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._complete(request_id, error="Request timed out waiting for result")

        result = self._results.pop(request_id, None)
        self._result_events.pop(request_id, None)

        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])

        return result

    async def start_workers(
        self,
        handler: Callable[[QueuedRequest], Coroutine[Any, Any, Any]],
        num_workers: int = 4,
    ) -> None:
        """Start background workers to process the queue."""
        self._running = True
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(handler, worker_id=i))
            self._workers.append(task)
        logger.info("Queue workers started", num_workers=num_workers)

    async def _worker(
        self,
        handler: Callable[[QueuedRequest], Coroutine[Any, Any, Any]],
        worker_id: int,
    ) -> None:
        """Worker loop that processes queued requests."""
        logger.info("Queue worker started", worker_id=worker_id)
        while self._running:
            request = await self.dequeue()
            if not request:
                continue

            try:
                result = await handler(request)
                await self.complete(request.request_id, result)
            except Exception as e:
                logger.error(
                    "Queue worker error",
                    worker_id=worker_id,
                    request_id=request.request_id,
                    error=str(e),
                )
                await self.fail(request.request_id, str(e))

    async def stop(self) -> None:
        """Stop all workers."""
        self._running = False
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Queue workers stopped")

    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "depth": self.depth,
            "pending": len(self._pending),
            "max_size": self._settings.request_queue_size,
            "workers": len(self._workers),
            "running": self._running,
        }