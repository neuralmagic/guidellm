import random
from unittest.mock import create_autospec

import pytest

from guidellm.backend import Backend
from guidellm.core import (
    TextGenerationBenchmark,
    TextGenerationRequest,
    TextGenerationResult,
)
from guidellm.request import RequestGenerator
from guidellm.scheduler import (
    LoadGenerator,
    Scheduler,
    SchedulerResult,
)


@pytest.mark.smoke()
def test_scheduler_result_default_intialization():
    benchmark = create_autospec(TextGenerationBenchmark, instance=True)
    scheduler_result = SchedulerResult(
        completed=False,
        count_total=0,
        count_completed=0,
        benchmark=benchmark,
    )

    assert scheduler_result.completed is False
    assert scheduler_result.count_total == 0
    assert scheduler_result.count_completed == 0
    assert scheduler_result.benchmark == benchmark
    assert scheduler_result.current_result is None


@pytest.mark.smoke()
def test_scheduler_result_initialization():
    benchmark = create_autospec(TextGenerationBenchmark, instance=True)
    result = TextGenerationResult(
        request=TextGenerationRequest(prompt="prompt"), output="Test output"
    )
    scheduler_result = SchedulerResult(
        completed=False,
        count_total=10,
        count_completed=5,
        benchmark=benchmark,
        current_result=result,
    )

    assert scheduler_result.completed is False
    assert scheduler_result.count_total == 10
    assert scheduler_result.count_completed == 5
    assert scheduler_result.benchmark == benchmark
    assert scheduler_result.current_result == result


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("mode", "rate", "max_number", "max_duration"),
    [
        ("synchronous", None, 10, None),
        ("throughput", 5.0, None, 60.0),
        ("poisson", 10.0, 100, None),
        ("constant", 1.0, None, 120.0),
    ],
)
def test_scheduler_initialization(mode, rate, max_number, max_duration):
    generator = create_autospec(RequestGenerator, instance=True)
    backend = create_autospec(Backend, instance=True)
    scheduler = Scheduler(
        generator,
        backend,
        mode=mode,
        rate=rate,
        max_number=max_number,
        max_duration=max_duration,
    )

    assert scheduler.generator == generator
    assert scheduler.backend == backend
    assert scheduler.mode == mode
    assert scheduler.rate == rate
    assert scheduler.max_number == max_number
    assert scheduler.max_duration == max_duration
    assert isinstance(scheduler.load_generator, LoadGenerator)
    assert scheduler.benchmark_mode in {"synchronous", "asynchronous", "throughput"}


@pytest.mark.sanity()
@pytest.mark.parametrize(
    ("mode", "rate", "max_number", "max_duration"),
    [
        # invalid modes
        ("invalid_mode", None, 10, None),
        # invalid max settings
        ("synchronous", None, None, None),
        ("synchronous", None, -1, 10),
        ("synchronous", None, 10, -1),
        # invalid rate settings
        ("constant", -1, None, 10),
        ("constant", None, None, 10),
        ("poisson", -1, None, 10),
        ("poisson", None, None, 10),
    ],
)
def test_scheduler_invalid_initialization(
    mode,
    rate,
    max_number,
    max_duration,
):
    generator = create_autospec(RequestGenerator, instance=True)
    backend = create_autospec(Backend, instance=True)

    with pytest.raises(ValueError):
        Scheduler(
            generator,
            backend,
            mode=mode,
            rate=rate,
            max_number=max_number,
            max_duration=max_duration,
        )


@pytest.mark.sanity()
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "mode",
    [
        "synchronous",
        "throughput",
        "poisson",
        "constant",
    ],
)
async def test_scheduler_run_number(mode, mock_backend):
    rate = 10.0
    max_number = 20
    generator = create_autospec(RequestGenerator, instance=True)

    # Mock the request generator and backend submit behavior
    generator.__iter__.return_value = iter(
        [TextGenerationRequest(prompt="Test", type_=random.choice(["text", "chat"]))]
        * (max_number * 2)
    )

    scheduler = Scheduler(
        generator,
        mock_backend,
        mode=mode,
        rate=rate,
        max_number=max_number,
    )

    run_count = 0
    count_completed = 0
    received_init = False
    received_final = False
    async for result in scheduler.run():
        run_count += 1

        assert run_count <= max_number + 2
        assert result.count_total == max_number
        assert result.benchmark is not None
        assert isinstance(result.benchmark, TextGenerationBenchmark)

        if result.current_result is not None:
            count_completed += 1

        if run_count == 1:
            assert not received_init
            assert not received_final
            assert count_completed == 0
            assert result.count_completed == 0
            assert not result.completed
            assert result.current_result is None
            received_init = True
        elif run_count - 2 == max_number:
            assert received_init
            assert not received_final
            assert count_completed == max_number
            assert result.count_completed == max_number
            assert result.completed
            assert result.current_result is None
            received_final = True
        else:
            assert received_init
            assert not received_final
            assert count_completed == run_count - 1
            assert result.count_completed == run_count - 1
            assert not result.completed
            assert result.current_result is not None
            assert isinstance(result.current_result, TextGenerationResult)

    assert received_init
    assert received_final
    assert count_completed == max_number
