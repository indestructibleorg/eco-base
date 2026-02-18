"""Unit tests for request queue."""
import pytest
import asyncio

from src.core.queue import RequestQueue, QueuedRequest, Priority


@pytest.fixture
def queue():
    return RequestQueue()


class TestRequestQueue:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, queue):
        req = QueuedRequest(
            payload={"test": True},
            model="test-model",
            priority=Priority.NORMAL,
        )
        request_id = await queue.enqueue(req)
        assert request_id == req.request_id
        assert queue.depth == 1

        dequeued = await queue.dequeue()
        assert dequeued is not None
        assert dequeued.request_id == request_id
        assert queue.depth == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        high = QueuedRequest(payload={}, priority=Priority.HIGH)
        low = QueuedRequest(payload={}, priority=Priority.LOW)
        normal = QueuedRequest(payload={}, priority=Priority.NORMAL)

        await queue.enqueue(low)
        await queue.enqueue(high)
        await queue.enqueue(normal)

        first = await queue.dequeue()
        assert first.priority == Priority.HIGH

        second = await queue.dequeue()
        assert second.priority == Priority.NORMAL

        third = await queue.dequeue()
        assert third.priority == Priority.LOW

    @pytest.mark.asyncio
    async def test_complete_and_wait(self, queue):
        req = QueuedRequest(payload={"q": "test"})
        await queue.enqueue(req)

        async def process():
            await asyncio.sleep(0.1)
            await queue.complete(req.request_id, {"answer": "done"})

        asyncio.create_task(process())
        result = await queue.wait_for_result(req.request_id, timeout=5.0)
        assert result == {"answer": "done"}

    @pytest.mark.asyncio
    async def test_fail_request(self, queue):
        req = QueuedRequest(payload={})
        await queue.enqueue(req)

        async def fail():
            await asyncio.sleep(0.1)
            await queue.fail(req.request_id, "test error")

        asyncio.create_task(fail())

        with pytest.raises(RuntimeError, match="test error"):
            await queue.wait_for_result(req.request_id, timeout=5.0)

    @pytest.mark.asyncio
    async def test_expired_request(self, queue):
        req = QueuedRequest(payload={}, timeout=0.0)  # Already expired
        await queue.enqueue(req)

        dequeued = await queue.dequeue()
        assert dequeued is None  # Expired requests are skipped

    @pytest.mark.asyncio
    async def test_stats(self, queue):
        stats = await queue.get_stats()
        assert stats["depth"] == 0
        assert stats["pending"] == 0
        assert stats["running"] is False

        await queue.enqueue(QueuedRequest(payload={}))
        stats = await queue.get_stats()
        assert stats["depth"] == 1
        assert stats["pending"] == 1

    @pytest.mark.asyncio
    async def test_serialization(self):
        req = QueuedRequest(
            payload={"key": "value"},
            model="test",
            priority=Priority.HIGH,
        )
        json_str = req.to_json()
        restored = QueuedRequest.from_json(json_str)
        assert restored.request_id == req.request_id
        assert restored.payload == req.payload
        assert restored.priority == req.priority
        assert restored.model == req.model