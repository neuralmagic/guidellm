import pytest
import asyncio
from unittest.mock import Mock, patch, call
from guidellm.scheduler.scheduler import Scheduler
from guidellm.scheduler.load_generator import LoadGenerationModes
from guidellm.core.result import BenchmarkResultSet, BenchmarkResult, BenchmarkError
from guidellm.request import BenchmarkRequest
from guidellm.backend import Backend


@pytest.fixture
def mock_request_generator():
    mock = Mock(spec=BenchmarkRequest)
    mock.__iter__ = Mock(
        return_value=iter([BenchmarkRequest(prompt="test prompt") for _ in range(10)])
    )
    return mock


@pytest.fixture
def mock_backend():
    mock = Mock(spec=Backend)
    mock.submit = Mock(
        return_value=BenchmarkResult(
            prompt="test prompt", generated_text="test output", start_time=0, end_time=1
        )
    )
    return mock


@pytest.fixture
def mock_load_generator():
    with patch("guidellm.scheduler.scheduler.LoadGenerator") as MockLoadGenerator:
        instance = MockLoadGenerator.return_value
        instance.times = Mock(return_value=iter([0.1 * i for i in range(10)]))
        yield instance


def test_scheduler_sync(mock_request_generator, mock_backend):
    scheduler = Scheduler(
        request_generator=mock_request_generator,
        backend=mock_backend,
        load_gen_mode=LoadGenerationModes.SYNCHRONOUS,
        max_requests=5,
    )

    result_set = scheduler.run()
    assert isinstance(result_set, BenchmarkResultSet)
    assert len(result_set.results) == 5


def test_scheduler_async_max_requests(
    mock_request_generator, mock_backend, mock_load_generator
):
    scheduler = Scheduler(
        request_generator=mock_request_generator,
        backend=mock_backend,
        load_gen_mode=LoadGenerationModes.CONSTANT,
        load_gen_rate=2.0,
        max_requests=5,
    )

    result_set = scheduler.run()
    assert isinstance(result_set, BenchmarkResultSet)
    assert len(result_set.results) == 5


def test_scheduler_async_max_duration(
    mock_request_generator, mock_backend, mock_load_generator
):
    scheduler = Scheduler(
        request_generator=mock_request_generator,
        backend=mock_backend,
        load_gen_mode=LoadGenerationModes.CONSTANT,
        load_gen_rate=2.0,
        max_duration=1.0,
    )

    result_set = scheduler.run()
    assert isinstance(result_set, BenchmarkResultSet)
    assert len(result_set.results) > 0


@pytest.mark.asyncio
async def test_scheduler_async_cancelled(
    mock_request_generator, mock_backend, mock_load_generator
):
    scheduler = Scheduler(
        request_generator=mock_request_generator,
        backend=mock_backend,
        load_gen_mode=LoadGenerationModes.CONSTANT,
        load_gen_rate=2.0,
        max_duration=0.1,
    )

    result_set = await scheduler._run_async()
    assert isinstance(result_set, BenchmarkResultSet)


if __name__ == "__main__":
    pytest.main()
