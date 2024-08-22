from typing import List, Optional, Union
from unittest.mock import create_autospec, patch

import pytest

from guidellm.backend import Backend
from guidellm.config import settings
from guidellm.core import (
    TextGenerationBenchmarkReport,
)
from guidellm.executor import (
    Executor,
    ExecutorResult,
    Profile,
    ProfileGenerationMode,
    ProfileGenerator,
)
from guidellm.request import RequestGenerator
from guidellm.scheduler import Scheduler, SchedulerResult


@pytest.fixture()
def mock_scheduler():
    with patch("guidellm.executor.base.Scheduler") as mock_scheduler:

        def scheduler_constructor(*args, **kwargs):
            mock_instance = create_autospec(Scheduler, instance=True)
            mock_instance.args = args
            mock_instance.kwargs = kwargs
            num_requests = kwargs.get("max_number", 10)

            async def run():
                benchmark = create_autospec(
                    TextGenerationBenchmarkReport, instance=True
                )
                benchmark.completed_request_rate = kwargs.get("rate", None)
                yield SchedulerResult(
                    completed=False,
                    count_total=10,
                    count_completed=0,
                    benchmark=benchmark,
                    current_result=None,
                )

                for index in range(num_requests):
                    yield SchedulerResult(
                        completed=False,
                        count_total=10,
                        count_completed=index + 1,
                        benchmark=benchmark,
                        current_result=create_autospec(
                            TextGenerationBenchmarkReport, instance=True
                        ),
                    )

                yield SchedulerResult(
                    completed=True,
                    count_total=num_requests,
                    count_completed=num_requests,
                    benchmark=benchmark,
                    current_result=None,
                )

            mock_instance.run.side_effect = run

            return mock_instance

        mock_scheduler.side_effect = scheduler_constructor
        yield mock_scheduler


@pytest.mark.smoke()
def test_executor_result_instantiation():
    report = create_autospec(TextGenerationBenchmarkReport, instance=True)
    scheduler_result = create_autospec(SchedulerResult, instance=True)
    executor_result = ExecutorResult(
        completed=True,
        count_total=10,
        count_completed=5,
        generation_modes=["synchronous", "throughput", "constant"],
        report=report,
        scheduler_result=scheduler_result,
    )

    assert executor_result.completed is True
    assert executor_result.count_total == 10
    assert executor_result.count_completed == 5
    assert executor_result.report == report
    assert executor_result.scheduler_result == scheduler_result


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("mode", "rate"),
    [
        ("sweep", None),
        ("synchronous", None),
        ("throughput", None),
        ("constant", 10),
        ("constant", [10, 20, 30]),
        ("poisson", 10),
        ("poisson", [10, 20, 30]),
    ],
)
def test_executor_instantiation(mode, rate):
    backend = create_autospec(Backend, instance=True)
    request_generator = create_autospec(RequestGenerator, instance=True)
    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode=mode,
        rate=rate,
        max_number=100,
        max_duration=60.0,
    )

    assert executor.backend == backend
    assert executor.request_generator == request_generator
    assert executor.profile_generator is not None
    assert isinstance(executor.profile_generator, ProfileGenerator)
    assert executor.profile_generator.mode == mode
    assert (
        executor.profile_generator.rates == rate
        if not rate or isinstance(rate, list)
        else [rate]
    )
    assert executor.max_number == 100
    assert executor.max_duration == 60.0


def _check_executor_result(
    result: ExecutorResult,
    num_profiles: int,
    num_requests: int,
    mode: ProfileGenerationMode,
    rate: Optional[Union[float, List[float]]],
    expected_complete: bool,
    expected_completed: int,
    expected_scheduler_result: bool,
    expected_current: bool,
    expected_profile_index: Optional[int] = None,
):
    if mode == "sweep":
        generation_modes = ["synchronous", "throughput"] + ["constant"] * (
            num_profiles - 2
        )
        generation_rates = [None, None] + [
            count + 2 for count in range(num_profiles - 2)
        ]
    elif mode in ("throughput", "synchronous"):
        generation_modes = [mode]
        generation_rates = [None]
    else:
        generation_modes = [mode] * num_profiles
        generation_rates = [rate] if not rate or isinstance(rate, list) else [rate]  # type: ignore  # noqa: PGH003

    # check completed and progress
    assert result.completed == expected_complete
    assert result.count_total == num_profiles
    assert result.count_completed == expected_completed

    # check modes
    assert result.generation_modes is not None
    assert len(result.generation_modes) == num_profiles
    assert result.generation_modes == generation_modes

    # check report
    assert result.report is not None
    assert isinstance(result.report, TextGenerationBenchmarkReport)
    assert len(result.report.benchmarks) == expected_completed
    assert "mode" in result.report.args
    assert result.report.args["mode"] == mode
    assert "rate" in result.report.args
    assert (
        result.report.args["rate"] == rate
        if rate is None or isinstance(rate, list)
        else [rate]
    )
    assert "max_number" in result.report.args
    assert result.report.args["max_number"] == num_requests
    assert "max_duration" in result.report.args
    assert result.report.args["max_duration"] is None

    # check scheduler result
    if expected_scheduler_result:
        assert result.scheduler_result is not None
        assert isinstance(result.scheduler_result, SchedulerResult)
    else:
        assert result.scheduler_result is None

    # check current index and profile
    if expected_current:
        assert result.current_index is not None
        assert result.current_index == expected_profile_index
        assert result.current_profile is not None
        assert isinstance(result.current_profile, Profile)
        assert (
            result.current_profile.load_gen_mode
            == generation_modes[expected_profile_index]
        )
        assert (
            result.current_profile.load_gen_rate
            == generation_rates[expected_profile_index]
        )
    else:
        assert result.current_index is None
        assert result.current_profile is None


async def _run_executor_tests(
    executor: Executor,
    num_profiles: int,
    num_requests: int,
    mode: ProfileGenerationMode,
    rate: Optional[Union[float, List[float]]],
):
    iterator = executor.run()

    if mode == "sweep":
        benchmark_rates = [1.0, float(num_profiles)] + [
            float(count + 2) for count in range(num_profiles - 2)
        ]
    elif mode in ("throughput", "synchronous"):
        benchmark_rates = [1.0]
    else:
        benchmark_rates = [rate] if not rate or isinstance(rate, list) else [rate]  # type: ignore  # noqa: PGH003

    result = await iterator.__anext__()
    _check_executor_result(
        result=result,
        num_profiles=num_profiles,
        num_requests=num_requests,
        mode=mode,
        rate=rate,
        expected_complete=False,
        expected_completed=0,
        expected_scheduler_result=False,
        expected_current=False,
    )

    for benchmark_index in range(num_profiles):
        result = await iterator.__anext__()
        _check_executor_result(
            result=result,
            num_profiles=num_profiles,
            num_requests=num_requests,
            mode=mode,
            rate=rate,
            expected_complete=False,
            expected_completed=benchmark_index,
            expected_scheduler_result=True,
            expected_current=True,
            expected_profile_index=benchmark_index,
        )

        for _ in range(num_requests):
            result = await iterator.__anext__()
            _check_executor_result(
                result=result,
                num_profiles=num_profiles,
                num_requests=num_requests,
                mode=mode,
                rate=rate,
                expected_complete=False,
                expected_completed=benchmark_index,
                expected_scheduler_result=True,
                expected_current=True,
                expected_profile_index=benchmark_index,
            )

        result = await iterator.__anext__()
        result.scheduler_result.benchmark.completed_request_rate = benchmark_rates[  # type: ignore  # noqa: PGH003
            benchmark_index
        ]
        _check_executor_result(
            result=result,
            num_profiles=num_profiles,
            num_requests=num_requests,
            mode=mode,
            rate=rate,
            expected_complete=False,
            expected_completed=benchmark_index + 1,
            expected_scheduler_result=True,
            expected_current=True,
            expected_profile_index=benchmark_index,
        )

    result = await iterator.__anext__()
    _check_executor_result(
        result=result,
        num_profiles=num_profiles,
        num_requests=num_requests,
        mode=mode,
        rate=rate,
        expected_complete=True,
        expected_completed=num_profiles,
        expected_scheduler_result=False,
        expected_current=False,
    )


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_executor_run_sweep(mock_scheduler):
    num_requests = 15

    backend = create_autospec(Backend, instance=True)
    request_generator = create_autospec(RequestGenerator, instance=True)
    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode="sweep",
        rate=None,
        max_number=num_requests,
    )

    await _run_executor_tests(
        executor, settings.num_sweep_profiles, num_requests, "sweep", None
    )


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_executor_run_synchronous(mock_scheduler):
    num_requests = 15

    backend = create_autospec(Backend, instance=True)
    request_generator = create_autospec(RequestGenerator, instance=True)
    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode="synchronous",
        rate=None,
        max_number=num_requests,
    )

    await _run_executor_tests(executor, 1, num_requests, "synchronous", None)


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_executor_run_throughput(mock_scheduler):
    num_requests = 15

    backend = create_autospec(Backend, instance=True)
    request_generator = create_autospec(RequestGenerator, instance=True)
    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode="throughput",
        rate=None,
        max_number=num_requests,
    )

    await _run_executor_tests(executor, 1, num_requests, "throughput", None)


@pytest.mark.smoke()
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("mode", "rate"),
    [
        ("constant", 10),
        ("constant", [10, 20, 30]),
        ("poisson", 10),
        ("poisson", [10, 20, 30]),
    ],
)
async def test_executor_run_constant_poisson(mock_scheduler, mode, rate):
    num_requests = 15

    backend = create_autospec(Backend, instance=True)
    request_generator = create_autospec(RequestGenerator, instance=True)
    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode=mode,
        rate=rate,
        max_number=num_requests,
    )

    await _run_executor_tests(
        executor, len(rate) if isinstance(rate, list) else 1, num_requests, mode, rate
    )
