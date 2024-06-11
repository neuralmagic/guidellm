import pytest
from guidellm.core import BenchmarkRequest
from guidellm.request import RequestGenerator


class DummyRequestGenerator(RequestGenerator):
    def __init__(self, mode: str = "async", async_queue_size: int = 50):
        super().__init__(mode, async_queue_size)
        self.count = 0

    def create_item(self) -> BenchmarkRequest:
        self.count += 1
        return BenchmarkRequest(prompt=f"Request {self.count}")


@pytest.mark.asyncio
async def test_request_generator_async():
    generator = DummyRequestGenerator(mode="async", async_queue_size=10)
    generator.start()

    # Collect a few items from the generator
    items = []
    for _ in range(5):
        items.append(await generator._queue.get())
        generator._queue.task_done()

    generator.stop()
    await generator._populating_task

    assert len(items) == 5
    assert items[0].prompt == "Request 1"
    assert items[-1].prompt == "Request 5"


def test_request_generator_sync():
    generator = DummyRequestGenerator(mode="sync", async_queue_size=10)

    items = []
    for _ in range(5):
        items.append(next(generator))

    assert len(items) == 5
    assert items[0].prompt == "Request 1"
    assert items[-1].prompt == "Request 5"


def test_request_generator_start_stop():
    generator = DummyRequestGenerator(mode="async", async_queue_size=10)

    # Start the generator
    generator.start()
    assert generator._populating_task is not None

    # Stop the generator
    generator.stop()
    assert generator._stop_event.is_set()
