import asyncio
import contextlib
import inspect
import time
from multiprocessing import Barrier, Event, Queue
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from typing import Any, Generic, Optional, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guidellm.scheduler import (
    LastCompletionRequestTimings,
    ScheduledRequestInfo,
    WorkerProcess,
)
from guidellm.scheduler.objects import (
    BackendInterface,
    RequestTimings,
)
from guidellm.scheduler.strategy import ScheduledRequestTimings


class MockRequestTimings(RequestTimings):
    """Mock timing implementation for testing."""


class MockBackend(BackendInterface):
    """Mock backend for testing worker functionality."""

    def __init__(self, delay: float = 0.1, should_fail: bool = False):
        self.delay = delay
        self.should_fail = should_fail
        self.process_startup_called = False
        self.validate_called = False
        self.process_shutdown_called = False
        self.resolve_called = False

    @property
    def processes_limit(self) -> Optional[int]:
        return None

    @property
    def requests_limit(self) -> Optional[int]:
        return None

    def info(self) -> dict[str, Any]:
        return {"type": "mock", "delay": self.delay}

    async def process_startup(self):
        await asyncio.sleep(self.delay)
        self.process_startup_called = True

    async def validate(self):
        await asyncio.sleep(self.delay)
        self.validate_called = True
        if self.should_fail:
            raise RuntimeError("Mock validation failed")

    async def process_shutdown(self):
        await asyncio.sleep(self.delay)
        self.process_shutdown_called = True

    async def resolve(self, request, request_info, request_history):
        self.resolve_called = True
        await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("Mock resolve failed")
        yield f"response_for_{request}", request_info


class TestWorkerProcess:
    """Test suite for WorkerProcess class."""

    @pytest.fixture
    def worker_process(self):
        """Create a WorkerProcess instance for testing."""
        # TODO: ensure a new WorkerProcess is created for each test
        backend = MockBackend()
        request_timings = LastCompletionRequestTimings()

        return WorkerProcess(
            local_rank=0,
            local_world_size=2,
            async_limit=5,
            startup_barrier=Barrier(2),
            shutdown_event=Event(),
            error_event=Event(),
            requests_queue=Queue(),
            updates_queue=Queue(),
            backend=backend,
            request_timings=request_timings,
            poll_intervals=0.01,
        )

    @pytest.mark.smoke
    def test_class_signatures(self, worker_process: WorkerProcess):
        """Test inheritance and type relationships."""
        # Class
        assert isinstance(worker_process, Generic)
        assert issubclass(WorkerProcess, Generic)

        # Generics
        orig_bases = getattr(WorkerProcess, "__orig_bases__", ())
        assert len(orig_bases) > 0
        generic_base = next(
            (
                base
                for base in orig_bases
                if hasattr(base, "__origin__") and base.__origin__ is Generic
            ),
            None,
        )
        assert generic_base is not None
        type_args = getattr(generic_base, "__args__", ())
        assert len(type_args) == 4  # BackendT, RequestT, RequestTimingsT, ResponseT

        # Function signatures
        run_sig = inspect.signature(WorkerProcess.run)
        assert len(run_sig.parameters) == 1
        assert "self" in run_sig.parameters

        run_async_sig = inspect.signature(WorkerProcess.run_async)
        assert len(run_async_sig.parameters) == 1
        assert "self" in run_async_sig.parameters

        stop_processing_sig = inspect.signature(WorkerProcess.run_async_stop_processing)
        assert len(stop_processing_sig.parameters) == 1
        assert "self" in stop_processing_sig.parameters

        requests_processing_sig = inspect.signature(
            WorkerProcess.run_async_requests_processing
        )
        assert len(requests_processing_sig.parameters) == 1
        assert "self" in requests_processing_sig.parameters

    @pytest.mark.smoke
    def test_initialization(self, worker_process: WorkerProcess):
        """Test basic initialization of WorkerProcess."""
        # worker info
        assert worker_process.local_rank == 0
        assert worker_process.local_world_size == 2
        assert worker_process.async_limit == 5

        # process synchronization
        assert isinstance(worker_process.startup_barrier, ProcessingBarrier)
        assert isinstance(worker_process.shutdown_event, ProcessingEvent)
        assert isinstance(worker_process.error_event, ProcessingEvent)
        assert hasattr(worker_process.requests_queue, "put")
        assert hasattr(worker_process.requests_queue, "get")
        assert hasattr(worker_process.updates_queue, "put")
        assert hasattr(worker_process.updates_queue, "get")

        # local synchronization
        assert worker_process.pending_requests_queue is None
        assert worker_process.pending_updates_queue is None

        # request processing
        assert isinstance(worker_process.backend, MockBackend)
        assert worker_process.poll_intervals == 0.01
        assert isinstance(worker_process.request_timings, LastCompletionRequestTimings)

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test that invalid initialization raises appropriate errors."""
        # TODO: add parameterized tests for this rather than defining lists within the method

        # Test with missing required parameters
        with pytest.raises(TypeError):
            WorkerProcess()

        # Create a complete set of valid parameters
        backend = MockBackend()
        request_timings = LastCompletionRequestTimings()
        barrier = Barrier(2)
        shutdown_event = Event()
        error_event = Event()
        requests_queue = Queue()
        updates_queue = Queue()

        # Test missing each required parameter one by one
        required_params = [
            "local_rank",
            "local_world_size",
            "async_limit",
            "startup_barrier",
            "shutdown_event",
            "error_event",
            "requests_queue",
            "updates_queue",
            "backend",
            "request_timings",
            "poll_intervals",
        ]

        for param_to_remove in required_params:
            kwargs = {
                "local_rank": 0,
                "local_world_size": 2,
                "async_limit": 5,
                "startup_barrier": barrier,
                "shutdown_event": shutdown_event,
                "error_event": error_event,
                "requests_queue": requests_queue,
                "updates_queue": updates_queue,
                "backend": backend,
                "request_timings": request_timings,
                "poll_intervals": 0.01,
            }

            del kwargs[param_to_remove]

            with pytest.raises(TypeError):
                WorkerProcess(**kwargs)

    @pytest.mark.smoke
    @patch("asyncio.run")
    def test_run(self, mock_asyncio_run, worker_process: WorkerProcess):
        """Test that run method functions as expected (calls run_async, handles exceptions)"""
        # Test successful execution
        with patch.object(
            worker_process, "run_async", new_callable=AsyncMock
        ) as mock_run_async:
            worker_process.run()
            mock_asyncio_run.assert_called_once()
            mock_run_async.assert_called_once()

        mock_asyncio_run.reset_mock()

        # Test exception during execution
        test_exception = RuntimeError("Test error in run_async")
        with patch.object(
            worker_process, "run_async", new_callable=AsyncMock
        ) as mock_run_async:
            mock_asyncio_run.side_effect = test_exception

            with pytest.raises(
                RuntimeError, match="Worker process 0 encountered an error"
            ):
                worker_process.run()

            assert worker_process.error_event.is_set()
