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
    with patch("guidellm.executor.executor.Scheduler") as mock_scheduler:

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


def _check_executor_result_base(
    result: ExecutorResult,
    expected_completed: bool,
    expected_count_total: int,
    expected_count_completed: int,
    expected_generation_modes: List[ProfileGenerationMode],
):
    assert result.completed == expected_completed
    assert result.count_total == expected_count_total
    assert result.count_completed == expected_count_completed
    assert result.generation_modes == expected_generation_modes


def _check_executor_result_report(
    result: ExecutorResult,
    mode: ProfileGenerationMode,
    rate: Optional[Union[float, List[float]]],
    max_number: Optional[int],
    max_duration: Optional[float],
    benchmarks_count: int,
):
    assert result.report is not None
    assert isinstance(result.report, TextGenerationBenchmarkReport)

    # check args
    for expected in (
        "backend_type",
        "target",
        "model",
        "data_type",
        "data",
        "tokenizer",
        "mode",
        "rate",
        "max_number",
        "max_duration",
    ):
        assert expected in result.report.args

    assert result.report.args["mode"] == mode
    assert (
        result.report.args["rate"] == rate
        if rate is None or not isinstance(rate, (float, int))
        else [rate]
    )
    assert result.report.args["max_number"] == max_number
    assert result.report.args["max_duration"] == max_duration

    # check benchmarks
    assert len(result.report.benchmarks) == benchmarks_count
    for benchmark in result.report.benchmarks:
        assert isinstance(benchmark, TextGenerationBenchmarkReport)


def _check_executor_result_scheduler(
    result: ExecutorResult,
    expected_scheduler_result: bool,
    expected_generation_modes: List[ProfileGenerationMode],
    expected_index: Optional[int],
    expected_profile_mode: Optional[ProfileGenerationMode],
    expected_profile_rate: Optional[float],
):
    if not expected_scheduler_result:
        assert result.scheduler_result is None
        assert result.current_index is None
        assert result.current_profile is None

        return

    assert result.scheduler_result is not None
    assert isinstance(result.scheduler_result, SchedulerResult)
    assert result.current_index == expected_index
    assert result.current_profile is not None
    assert isinstance(result.current_profile, Profile)
    assert result.current_profile.load_gen_mode == expected_profile_mode
    assert result.current_profile.load_gen_rate == expected_profile_rate
    assert (
        result.current_profile.load_gen_mode
        == expected_generation_modes[expected_index]  # type: ignore
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

    num_profiles = 2 + settings.num_sweep_profiles
    generation_modes = ["synchronous", "throughput"] + [
        "constant"
    ] * settings.num_sweep_profiles
    generation_rates = [None, None] + list(range(2, settings.num_sweep_profiles + 2))
    output_rates = [1, settings.num_sweep_profiles + 1] + list(
        range(2, settings.num_sweep_profiles + 2)
    )

    iterator = executor.run()

    # Check start result
    result = await iterator.__anext__()
    _check_executor_result_base(
        result=result,
        expected_completed=False,
        expected_count_total=num_profiles,
        expected_count_completed=0,
        expected_generation_modes=generation_modes,  # type: ignore
    )
    _check_executor_result_report(
        result=result,
        mode="sweep",
        rate=None,
        max_number=num_requests,
        max_duration=None,
        benchmarks_count=0,
    )
    _check_executor_result_scheduler(
        result=result,
        expected_scheduler_result=False,
        expected_generation_modes=generation_modes,  # type: ignore
        expected_index=None,
        expected_profile_mode=None,
        expected_profile_rate=None,
    )

    for scheduler_index in range(num_profiles):
        for request_index in range(num_requests + 2):
            result = await iterator.__anext__()
            _check_executor_result_base(
                result=result,
                expected_completed=False,
                expected_count_total=num_profiles,
                expected_count_completed=scheduler_index
                if request_index < num_requests + 1
                else scheduler_index + 1,
                expected_generation_modes=generation_modes,  # type: ignore
            )
            _check_executor_result_report(
                result=result,
                mode="sweep",
                rate=None,
                max_number=num_requests,
                max_duration=None,
                benchmarks_count=scheduler_index
                if request_index < num_requests + 1
                else scheduler_index + 1,
            )
            _check_executor_result_scheduler(
                result=result,
                expected_scheduler_result=True,
                expected_generation_modes=generation_modes,  # type: ignore
                expected_index=scheduler_index,
                expected_profile_mode=generation_modes[scheduler_index],  # type: ignore
                expected_profile_rate=generation_rates[scheduler_index],
            )
        # set the rate for the benchmark for sweep profile generation
        result.report.benchmarks[-1].completed_request_rate = output_rates[  # type: ignore
            scheduler_index
        ]
        result.report.benchmarks[-1].request_count = num_requests  # type: ignore

    # Check end result
    result = await iterator.__anext__()
    _check_executor_result_base(
        result=result,
        expected_completed=True,
        expected_count_total=num_profiles,
        expected_count_completed=num_profiles,
        expected_generation_modes=generation_modes,  # type: ignore
    )
    _check_executor_result_report(
        result=result,
        mode="sweep",
        rate=None,
        max_number=num_requests,
        max_duration=None,
        benchmarks_count=num_profiles,
    )
    _check_executor_result_scheduler(
        result=result,
        expected_scheduler_result=False,
        expected_generation_modes=generation_modes,  # type: ignore
        expected_index=None,
        expected_profile_mode=None,
        expected_profile_rate=None,
    )


@pytest.mark.smoke()
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "mode",
    [
        "synchronous",
        "throughput",
    ],
)
async def test_executor_run_non_rate_modes(mock_scheduler, mode):
    num_requests = 15

    backend = create_autospec(Backend, instance=True)
    request_generator = create_autospec(RequestGenerator, instance=True)
    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode=mode,
        rate=None,
        max_number=num_requests,
    )

    iterator = executor.run()

    # Check start result
    result = await iterator.__anext__()
    _check_executor_result_base(
        result=result,
        expected_completed=False,
        expected_count_total=1,
        expected_count_completed=0,
        expected_generation_modes=[mode],
    )
    _check_executor_result_report(
        result=result,
        mode=mode,
        rate=None,
        max_number=num_requests,
        max_duration=None,
        benchmarks_count=0,
    )
    _check_executor_result_scheduler(
        result=result,
        expected_scheduler_result=False,
        expected_generation_modes=[mode],
        expected_index=None,
        expected_profile_mode=None,
        expected_profile_rate=None,
    )

    for request_index in range(num_requests + 2):
        result = await iterator.__anext__()
        _check_executor_result_base(
            result=result,
            expected_completed=False,
            expected_count_total=1,
            expected_count_completed=0 if request_index < num_requests + 1 else 1,
            expected_generation_modes=[mode],
        )
        _check_executor_result_report(
            result=result,
            mode=mode,
            rate=None,
            max_number=num_requests,
            max_duration=None,
            benchmarks_count=0 if request_index < num_requests + 1 else 1,
        )
        _check_executor_result_scheduler(
            result=result,
            expected_scheduler_result=True,
            expected_generation_modes=[mode],
            expected_index=0,
            expected_profile_mode=mode,
            expected_profile_rate=None,
        )

    # Check end result
    result = await iterator.__anext__()
    _check_executor_result_base(
        result=result,
        expected_completed=True,
        expected_count_total=1,
        expected_count_completed=1,
        expected_generation_modes=[mode],
    )
    _check_executor_result_report(
        result=result,
        mode=mode,
        rate=None,
        max_number=num_requests,
        max_duration=None,
        benchmarks_count=1,
    )
    _check_executor_result_scheduler(
        result=result,
        expected_scheduler_result=False,
        expected_generation_modes=[mode],
        expected_index=None,
        expected_profile_mode=None,
        expected_profile_rate=None,
    )


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
async def test_executor_run_rate_modes(mock_scheduler, mode, rate):
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

    num_profiles = len(rate) if isinstance(rate, list) else 1
    generation_modes = [mode] * num_profiles
    generation_rates = rate if isinstance(rate, list) else [rate]

    iterator = executor.run()

    # Check start result
    result = await iterator.__anext__()
    _check_executor_result_base(
        result=result,
        expected_completed=False,
        expected_count_total=num_profiles,
        expected_count_completed=0,
        expected_generation_modes=generation_modes,
    )
    _check_executor_result_report(
        result=result,
        mode=mode,
        rate=rate,
        max_number=num_requests,
        max_duration=None,
        benchmarks_count=0,
    )
    _check_executor_result_scheduler(
        result=result,
        expected_scheduler_result=False,
        expected_generation_modes=generation_modes,
        expected_index=None,
        expected_profile_mode=None,
        expected_profile_rate=None,
    )

    for scheduler_index in range(num_profiles):
        for request_index in range(num_requests + 2):
            result = await iterator.__anext__()
            _check_executor_result_base(
                result=result,
                expected_completed=False,
                expected_count_total=num_profiles,
                expected_count_completed=scheduler_index
                if request_index < num_requests + 1
                else scheduler_index + 1,
                expected_generation_modes=generation_modes,
            )
            _check_executor_result_report(
                result=result,
                mode=mode,
                rate=rate,
                max_number=num_requests,
                max_duration=None,
                benchmarks_count=scheduler_index
                if request_index < num_requests + 1
                else scheduler_index + 1,
            )
            _check_executor_result_scheduler(
                result=result,
                expected_scheduler_result=True,
                expected_generation_modes=generation_modes,
                expected_index=scheduler_index,
                expected_profile_mode=generation_modes[scheduler_index],
                expected_profile_rate=generation_rates[scheduler_index],
            )

    # Check end result
    result = await iterator.__anext__()
    _check_executor_result_base(
        result=result,
        expected_completed=True,
        expected_count_total=num_profiles,
        expected_count_completed=num_profiles,
        expected_generation_modes=generation_modes,
    )
    _check_executor_result_report(
        result=result,
        mode=mode,
        rate=rate,
        max_number=num_requests,
        max_duration=None,
        benchmarks_count=num_profiles,
    )
    _check_executor_result_scheduler(
        result=result,
        expected_scheduler_result=False,
        expected_generation_modes=generation_modes,
        expected_index=None,
        expected_profile_mode=None,
        expected_profile_rate=None,
    )
