import asyncio
import time
from unittest.mock import AsyncMock, create_autospec

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
def test_scheduler_result():
    benchmark = create_autospec(TextGenerationBenchmark, instance=True)
    result = TextGenerationResult(
        request=TextGenerationRequest(prompt="prompt"), output="Test output"
    )
    scheduler_result = SchedulerResult(
        completed=True,
        count_total=10,
        count_completed=5,
        benchmark=benchmark,
        current_result=result,
    )

    assert scheduler_result.completed is True
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
def test_scheduler_instantiation(mode, rate, max_number, max_duration):
    generator = create_autospec(RequestGenerator, instance=True)
    worker = create_autospec(Backend, instance=True)
    scheduler = Scheduler(
        generator,
        worker,
        mode=mode,
        rate=rate,
        max_number=max_number,
        max_duration=max_duration,
    )

    assert scheduler.generator == generator
    assert scheduler.worker == worker
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
def test_scheduler_invalid_instantiation(
    mode,
    rate,
    max_number,
    max_duration,
):
    generator = create_autospec(RequestGenerator, instance=True)
    worker = create_autospec(Backend, instance=True)

    with pytest.raises(ValueError):
        Scheduler(
            generator,
            worker,
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
async def test_scheduler_run_number(mode):
    rate = 10.0
    max_number = 20
    generator = create_autospec(RequestGenerator, instance=True)
    worker = create_autospec(Backend, instance=True)

    # Mock the request generator and backend submit behavior
    generator.__iter__.return_value = iter(
        [TextGenerationRequest(prompt="Test")] * (max_number * 2)
    )
    worker.submit = AsyncMock()

    def _submit(req):
        res = TextGenerationResult(request=req, output="Output")
        res.start(prompt=req.prompt)
        res.output_token("token")
        res.end()
        return res

    worker.submit.side_effect = _submit

    scheduler = Scheduler(
        generator,
        worker,
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


@pytest.mark.sanity()
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "mode",
    [
        "synchronous",
        "constant",
    ],
)
@pytest.mark.flaky(reruns=5)
async def test_scheduler_run_duration(mode):
    rate = 10
    max_duration = 2
    generator = create_autospec(RequestGenerator, instance=True)
    worker = create_autospec(Backend, instance=True)

    # Mock the request generator and backend submit behavior
    generator.__iter__.return_value = iter(
        [TextGenerationRequest(prompt="Test")] * (rate * max_duration * 100)
    )
    worker.submit = AsyncMock()

    async def _submit(req):
        await asyncio.sleep(0.1)
        res = TextGenerationResult(request=req, output="Output")
        res.start(prompt=req.prompt)
        res.output_token("token")
        res.end()
        return res

    worker.submit.side_effect = _submit

    scheduler = Scheduler(
        generator,
        worker,
        mode=mode,
        rate=rate,
        max_duration=max_duration,
    )

    run_count = 0
    count_completed = 0
    received_init = False
    received_final = False
    start_time = time.time()
    async for result in scheduler.run():
        run_count += 1

        assert run_count <= max_duration * rate + 2
        assert result.count_total == max_duration
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
        elif time.time() - start_time >= max_duration:
            assert received_init
            assert not received_final
            assert result.count_completed == max_duration
            assert result.completed
            assert result.current_result is None
            received_final = True
        else:
            assert received_init
            assert not received_final
            assert result.count_completed == round(time.time() - start_time)
            assert not result.completed
            assert result.current_result is not None
            assert isinstance(result.current_result, TextGenerationResult)

    assert received_init
    assert received_final
    end_time = time.time()
    assert pytest.approx(end_time - start_time, abs=0.1) == max_duration
    assert pytest.approx(count_completed, abs=5) == max_duration * rate
