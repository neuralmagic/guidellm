import pytest
import asyncio
from guidellm.scheduler import Task


@pytest.mark.asyncio
async def test_task_run_async():
    async def sample_func(x, y):
        return x + y

    params = {"x": 1, "y": 2}
    task = Task(func=sample_func, params=params)
    result = await task.run_async()
    assert result == 3


def test_task_run_sync():
    def sample_func(x, y):
        return x + y

    params = {"x": 1, "y": 2}
    task = Task(func=sample_func, params=params)
    result = task.run_sync()
    assert result == 3


@pytest.mark.asyncio
async def test_task_cancel():
    async def sample_func(x, y):
        await asyncio.sleep(1)
        return x + y

    params = {"x": 1, "y": 2}
    task = Task(func=sample_func, params=params)

    asyncio.create_task(task.run_async())
    await asyncio.sleep(0.1)
    task.cancel()

    result = await task.run_async()
    assert result is None


def test_task_is_cancelled():
    def sample_func(x, y):
        return x + y

    params = {"x": 1, "y": 2}
    task = Task(func=sample_func, params=params)

    assert not task.is_cancelled()
    task.cancel()
    assert task.is_cancelled()
