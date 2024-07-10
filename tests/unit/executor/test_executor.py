import pytest

from domain.backend.base import Backend, BackendEngine
from domain.executor import Executor, ProfileGenerationMode
from tests.dummy.services import TestRequestGenerator


@pytest.mark.parametrize(
    "profile_generation_mode,backend_engine,report_len",
    [
        (ProfileGenerationMode.SINGLE, BackendEngine.OPENAI_SERVER, 1),
    ],
)
def test_executor(
    mocker, openai_backend_factory, profile_generation_mode, backend_engine, report_len
):
    """
    Ensure that the executor works correctly with the profile generator
    """

    scheduler_mock = mocker.patch("guidellm.scheduler.scheduler.Scheduler.run")
    request_genrator = TestRequestGenerator(tokenizer="bert-base-uncased")
    profile_generator_args = {"rate_type": profile_generation_mode, "rate": 1.0}

    executor = Executor(
        backend=openai_backend_factory(),
        request_generator=request_genrator,
        profile_mode=profile_generation_mode,
        profile_args=profile_generator_args,
        max_requests=1,
        max_duration=120,
    )
    report = executor.run()

    assert isinstance(executor.backend, Backend._registry[backend_engine])
    assert len(report) == report_len
    assert scheduler_mock.call_count == 1
