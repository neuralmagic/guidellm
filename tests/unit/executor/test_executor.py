from unittest.mock import MagicMock
from src.guidellm.backend.base import Backend
from src.guidellm.executor.executor import Executor
from src.guidellm.request.base import RequestGenerator

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
