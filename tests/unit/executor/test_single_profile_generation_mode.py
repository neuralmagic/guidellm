import pytest

from guidellm.core import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.executor import Executor, ProfileGenerationMode
from guidellm.scheduler import LoadGenerationMode
from tests import dummy


@pytest.mark.parametrize(
    "load_gen_mode",
    [
        LoadGenerationMode.SYNCHRONOUS,
        LoadGenerationMode.POISSON,
        LoadGenerationMode.CONSTANT,
    ],
)
def test_executor_single_profile_generator_benchmark_report(
    mocker, openai_backend_factory, load_gen_mode
):
    scheduler_run_patch = mocker.patch(
        "guidellm.scheduler.scheduler.Scheduler.run",
        return_value=TextGenerationBenchmark(mode="test", rate=1.0),
    )
    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased"
    )
    rates = [1.0]
    if load_gen_mode == LoadGenerationMode.SYNCHRONOUS:
        rates = None
    profile_generator_kwargs = {"load_gen_mode": load_gen_mode, "rates": rates}

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=ProfileGenerationMode.FIXED_RATE,
        profile_args=profile_generator_kwargs,
        max_requests=1,
        max_duration=None,
    )

    report: TextGenerationBenchmarkReport = executor.run()

    assert scheduler_run_patch.call_count == 1
    assert len(report.benchmarks) == 1
    assert report.benchmarks[0].mode == "test"
