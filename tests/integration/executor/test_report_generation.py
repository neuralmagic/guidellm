import pytest

from domain.backend.base import BackendEngine
from domain.core.result import TextGenerationBenchmarkReport
from domain.executor import Executor, ProfileGenerationMode, SingleProfileGenerator
from tests.dummy.services import TestRequestGenerator


@pytest.mark.parametrize(
    "profile_generation_mode",
    [
        ProfileGenerationMode.SYNCHRONOUS,
        ProfileGenerationMode.CONSTANT,
        ProfileGenerationMode.POISSON,
    ],
)
def test_executor_openai_unsupported_generation_modes(
    openai_backend_factory, profile_generation_mode
):
    """
    Execution configuration:
        * Profile generation modes: sync,
    """

    request_genrator = TestRequestGenerator(tokenizer="bert-base-uncased")
    profile_generator_args = {"rate_type": profile_generation_mode, "rate": 1.0}

    with pytest.raises(ValueError):
        Executor(
            backend=openai_backend_factory(),
            request_generator=request_genrator,
            profile_mode=profile_generation_mode,
            profile_args=profile_generator_args,
            max_requests=1,
            max_duration=120,
        )


def test_executor_openai_single_report_generation(openai_backend_factory):
    """
    Check OpenAI Single Report Generation.
    """

    request_genrator = TestRequestGenerator(tokenizer="bert-base-uncased")
    profile_generation_mode = ProfileGenerationMode.SINGLE
    profile_generator_args = {"rate_type": profile_generation_mode, "rate": 1.0}

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_args,
        max_requests=1,
        max_duration=120,
    )

    report: TextGenerationBenchmarkReport = executor.run()

    assert report


# TODO: test sweep profile generator
