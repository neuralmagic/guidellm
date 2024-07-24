import time

import pytest

from guidellm.backend import OpenAIBackend
from guidellm.core import TextGenerationBenchmarkReport
from guidellm.executor import Executor, ProfileGenerationMode
from guidellm.scheduler import LoadGenerationMode
from tests import dummy


@pytest.mark.sanity
def test_executor_openai_single_report_generation_sync_mode(
    openai_backend_factory, openai_completion_create_patch
):
    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased"
    )
    profile_generation_mode = ProfileGenerationMode.FIXED_RATE
    profile_generator_kwargs = {
        "load_gen_mode": LoadGenerationMode.SYNCHRONOUS,
    }

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_kwargs,
        max_requests=1,
        max_duration=2,
    )

    report: TextGenerationBenchmarkReport = executor.run()

    assert isinstance(executor.backend, OpenAIBackend)
    assert len(report.benchmarks) == 1
    assert len(report.benchmarks[0].results) == 1
    assert report.benchmarks[0].results[0].output == " ".join(
        item.content for item in openai_completion_create_patch
    )


@pytest.mark.sanity
def test_executor_openai_single_report_generation_constant_mode_infinite(
    openai_backend_factory,
):
    """
    Test without max duration defined.

    Does not matter how many requests is specified,
    the execution DOES NOT have any duration limitations.
    """

    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased"
    )
    profile_generation_mode = ProfileGenerationMode.FIXED_RATE
    profile_generator_kwargs = {
        "load_gen_mode": LoadGenerationMode.CONSTANT,
        "rates": [1.0],
    }

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_kwargs,
        max_requests=2,
        max_duration=None,  # not specified for no limitations
    )

    report: TextGenerationBenchmarkReport = executor.run()

    assert isinstance(executor.backend, OpenAIBackend)
    assert len(report.benchmarks) == 1
    assert len(report.benchmarks[0].errors) == 0


@pytest.mark.sanity
def test_executor_openai_single_report_generation_constant_mode_limited(
    openai_backend_factory,
):
    """
    Test with max duration defined.
    """

    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased"
    )
    profile_generation_mode = ProfileGenerationMode.FIXED_RATE
    profile_generator_kwargs = {
        "load_gen_mode": LoadGenerationMode.CONSTANT,
        "rates": [1.0],
    }

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_kwargs,
        max_requests=2,
        max_duration=3,
    )

    report: TextGenerationBenchmarkReport = executor.run()

    assert isinstance(executor.backend, OpenAIBackend)
    assert len(report.benchmarks) == 1
    assert len(report.benchmarks[0].results) == 2


@pytest.mark.sanity
def test_executor_openai_single_report_generation_constant_mode_failed(
    mocker, openai_backend_factory
):
    """
    Test max duration immediate tasks iteration break up
    because of the `time.time() - start_time >= self._max_duration`.
    """

    mocker.patch("guidellm.backend.Backend.submit", side_effect=Exception)

    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased"
    )
    profile_generation_mode = ProfileGenerationMode.FIXED_RATE
    profile_generator_kwargs = {
        "load_gen_mode": LoadGenerationMode.CONSTANT,
        "rates": [1.0],
    }

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_kwargs,
        max_requests=3,
        max_duration=None,
    )

    report: TextGenerationBenchmarkReport = executor.run()

    assert isinstance(executor.backend, OpenAIBackend)
    assert len(report.benchmarks) == 1
    assert len(report.benchmarks[0].errors) == 3


@pytest.mark.sanity
def test_executor_openai_single_report_generation_constant_mode_cancelled_reports(
    openai_backend_factory,
):
    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased"
    )
    profile_generation_mode = ProfileGenerationMode.FIXED_RATE
    profile_generator_kwargs = {
        "load_gen_mode": LoadGenerationMode.CONSTANT,
        "rates": [1.0],
    }

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_kwargs,
        max_requests=5,
        max_duration=3,
    )

    start_time: float = time.perf_counter()
    report: TextGenerationBenchmarkReport = executor.run()
    end_time: float = time.perf_counter() - start_time

    assert isinstance(executor.backend, OpenAIBackend)
    assert len(report.benchmarks) == 1
    assert len(report.benchmarks[0].errors) > 0
    assert round(end_time) == 3
