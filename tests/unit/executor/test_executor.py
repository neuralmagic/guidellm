import pytest
from unittest.mock import MagicMock, patch
from guidellm.backend.base import Backend
from guidellm.executor.executor import Executor
from guidellm.executor.profile_generator import Profile, ProfileGenerator
from guidellm.request.base import RequestGenerator
from guidellm.scheduler.load_generator import LoadGenerationModes

def test_executor_creation():
    mock_request_generator = MagicMock(spec=RequestGenerator)
    mock_backend = MagicMock(spec=Backend)
    rate_type = "sweep"
    profile_args = None
    max_requests = None,
    max_duration = None,
    executor = Executor(mock_request_generator, mock_backend, rate_type, profile_args, max_requests, max_duration);
    assert executor.request_generator == mock_request_generator
    assert executor.backend == mock_backend
    assert executor.max_requests == max_requests
    assert executor.max_duration == max_duration


@pytest.fixture
def mock_request_generator():
    return MagicMock(spec=RequestGenerator)

@pytest.fixture
def mock_backend():
    return MagicMock(spec=Backend)

@pytest.fixture
def mock_scheduler():
    with patch('guidellm.executor.executor.Scheduler') as MockScheduler:
        yield MockScheduler

def test_executor_run(mock_request_generator, mock_backend, mock_scheduler):

    mock_profile_generator = MagicMock(spec=ProfileGenerator)
    profiles = [
        Profile(load_gen_mode=LoadGenerationModes.CONSTANT, load_gen_rate=1.0),
        Profile(load_gen_mode=LoadGenerationModes.CONSTANT, load_gen_rate=2.0),
        None
    ]
    mock_profile_generator.next_profile.side_effect = profiles
    
    with patch('guidellm.executor.executor.ProfileGenerator.create_generator', return_value=mock_profile_generator):
        executor = Executor(
            request_generator=mock_request_generator,
            backend=mock_backend,
            rate_type="constant",
            profile_args={"rate_type": "constant", "rate": [1.0, 2.0]},
            max_requests=10,
            max_duration=100
        )

        mock_benchmark = MagicMock()
        mock_scheduler.return_value.run.return_value = mock_benchmark

        report = executor.run()


        assert mock_scheduler.call_count == 2
        assert len(report.benchmarks) == 2
        assert report.benchmarks[0] == mock_benchmark
        assert report.benchmarks[1] == mock_benchmark
        calls = mock_scheduler.call_args_list
        assert calls[0][1]['load_gen_mode'] == LoadGenerationModes.CONSTANT
        assert calls[0][1]['load_gen_rate'] == 1.0
        assert calls[1][1]['load_gen_mode'] == LoadGenerationModes.CONSTANT
        assert calls[1][1]['load_gen_rate'] == 2.0
