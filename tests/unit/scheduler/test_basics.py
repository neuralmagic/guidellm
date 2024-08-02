from typing import Callable

import pytest
from guidellm.backend import OpenAIBackend
from guidellm.scheduler import LoadGenerationMode, Scheduler

from tests import dummy


@pytest.mark.parametrize(
    ("load_gen_mode", "max_requests", "max_duration", "load_gen_rate"),
    [
        # Sync load generation mode payload
        (LoadGenerationMode.SYNCHRONOUS, None, None, None),
        (LoadGenerationMode.SYNCHRONOUS, 1, -1, 1.0),
        (LoadGenerationMode.SYNCHRONOUS, -1, 1, 1.0),
        (LoadGenerationMode.SYNCHRONOUS, None, -1, 1.0),
        # Constant load generation mode payload
        (LoadGenerationMode.CONSTANT, None, None, 1.0),
        (LoadGenerationMode.CONSTANT, -1, 1, 1.0),
        (LoadGenerationMode.CONSTANT, 1, 1, None),
        (LoadGenerationMode.CONSTANT, 1, 0, None),
        (LoadGenerationMode.CONSTANT, 0, 0, None),
        # Poisson load generation mode payload
        (LoadGenerationMode.POISSON, None, None, 1.0),
        (LoadGenerationMode.POISSON, -1, 1, 1.0),
        (LoadGenerationMode.POISSON, 1, 1, None),
        (LoadGenerationMode.POISSON, 1, 0, None),
        (LoadGenerationMode.POISSON, 0, 0, None),
    ],
)
def test_scheduler_invalid_parameters(
    openai_backend_factory: Callable[..., OpenAIBackend],
    load_gen_mode,
    max_requests,
    max_duration,
    load_gen_rate,
):
    """
    Test scheduler initializer parameters validation.
    """
    with pytest.raises(ValueError):
        Scheduler(
            request_generator=dummy.services.TestRequestGenerator(),
            backend=openai_backend_factory(),
            load_gen_mode=load_gen_mode,
            load_gen_rate=load_gen_rate,
            max_requests=max_requests,
            max_duration=max_duration,
        )
