import time
from typing import Callable

import pytest
from guidellm.backend import OpenAIBackend
from guidellm.core import TextGenerationBenchmark
from guidellm.scheduler import LoadGenerationMode, Scheduler

from tests import dummy


@pytest.mark.sanity()
@pytest.mark.parametrize("max_requests", [1, 2, 3])
def test_scheduler_max_requests_limitation(
    openai_backend_factory: Callable[..., OpenAIBackend],
    max_requests: int,
):
    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased",
    )

    scheduler = Scheduler(
        request_generator=request_genrator,
        backend=openai_backend_factory(),
        load_gen_mode=LoadGenerationMode.CONSTANT,
        load_gen_rate=1.0,
        max_requests=max_requests,
        max_duration=None,
    )

    benchmark: TextGenerationBenchmark = scheduler.run()

    assert len(benchmark.results) == max_requests
    assert benchmark.errors == []


@pytest.mark.sanity()
@pytest.mark.parametrize("max_duration", [1, 2, 3])
def test_scheduler_max_duration_limitation(
    openai_backend_factory: Callable[..., OpenAIBackend],
    max_duration: int,
):
    request_genrator = dummy.services.TestRequestGenerator(
        tokenizer="bert-base-uncased",
    )

    scheduler = Scheduler(
        request_generator=request_genrator,
        backend=openai_backend_factory(),
        load_gen_mode=LoadGenerationMode.CONSTANT,
        load_gen_rate=1.0,
        max_requests=None,
        max_duration=max_duration,
    )

    start_time = time.perf_counter()
    scheduler.run()
    end_time = time.perf_counter() - start_time

    assert round(end_time) == max_duration
