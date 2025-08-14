import asyncio
import inspect
import math
import os
import queue
import threading
import time
from collections import defaultdict
from multiprocessing import get_context
from queue import Empty
from typing import Any, Generic, Optional

import culsans
import pytest

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    BackendInterface,
    ConcurrentStrategy,
    MaxNumberConstraint,
    MeasuredRequestTimings,
    ScheduledRequestInfo,
    SchedulerState,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerProcessGroup,
    worker_group,
)
from guidellm.utils import MsgpackEncoding


class MockWorker:
    """Picklable mock worker used to validate create_processes logic."""

    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(
        self,
        local_rank,
        local_world_size,
        async_limit,
        startup_barrier,
        shutdown_event,
        error_event,
        requests_queue,
        updates_queue,
        backend,
        request_timings,
        poll_intervals,
    ):
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.async_limit = async_limit
        self.startup_barrier = startup_barrier
        self.shutdown_event = shutdown_event
        self.error_event = error_event
        self.requests_queue = requests_queue
        self.updates_queue = updates_queue
        self.backend = backend
        self.request_timings = request_timings
        self.poll_intervals = poll_intervals

    def run(self):
        try:
            # Access parameters to ensure they're usable and wait for barrier
            shutdown_is_set = self.shutdown_event.is_set()
            error_is_set = self.error_event.is_set()
            backend_info = self.backend.info()

            self.startup_barrier.wait()

            # Publish diagnostics back to parent for assertions
            payload = (
                "diag",
                self.local_rank,
                {
                    "child_pid": os.getpid(),
                    "local_rank": self.local_rank,
                    "local_world_size": self.local_world_size,
                    "async_limit": self.async_limit,
                    "backend_info": backend_info,
                    "shutdown_is_set": shutdown_is_set,
                    "error_is_set": error_is_set,
                    "passed_barrier": True,
                    "request_timings_type": type(self.request_timings).__name__,
                },
            )
            self.updates_queue.put(payload)
        except Exception as err:  # noqa: BLE001
            try:
                self.error_event.set()
                self.updates_queue.put(("error", self.local_rank, repr(err)))
            finally:
                raise


class MockWorkerProcessor(MockWorker):
    def run(self):
        self.startup_barrier.wait()

        while not self.shutdown_event.is_set() and not self.error_event.is_set():
            try:
                request_msg = self.requests_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            request, request_info = MsgpackEncoding.decode(request_msg)
            request_info.status = "in_progress"
            self.updates_queue.put(
                MsgpackEncoding.encode((None, request, request_info))
            )
            time.sleep(0.01)
            request_info.status = "completed"
            response = f"response_for_{request}"
            self.updates_queue.put(
                MsgpackEncoding.encode((response, request, request_info))
            )


class MockRequestTimings(MeasuredRequestTimings):
    """Mock timing implementation for testing."""


class MockBackend(BackendInterface):
    """Mock backend for testing worker group functionality."""

    def __init__(
        self,
        processes_limit_value: Optional[int] = None,
        requests_limit_value: Optional[int] = None,
    ):
        self._processes_limit = processes_limit_value
        self._requests_limit = requests_limit_value

    @property
    def processes_limit(self) -> Optional[int]:
        return self._processes_limit

    @property
    def requests_limit(self) -> Optional[int]:
        return self._requests_limit

    def info(self) -> dict[str, Any]:
        return {"type": "mock"}

    async def process_startup(self):
        pass

    async def validate(self):
        pass

    async def process_shutdown(self):
        pass

    async def resolve(self, request, request_info, request_history):
        yield f"response_for_{request}"


class TestWorkerProcessGroup:
    """Test suite for WorkerProcessGroup class."""

    @pytest.fixture
    def worker_process_group(self):
        """Create a WorkerProcessGroup instance for testing."""
        backend = MockBackend()
        requests = ["request1", "request2", "request3"]
        strategy = SynchronousStrategy()
        constraints = {"max_requests": MaxNumberConstraint(max_num=10)}

        return WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=strategy,
            constraints=constraints,
        )

    @pytest.mark.smoke
    def test_class_signatures(self, worker_process_group: WorkerProcessGroup):
        """Test inheritance and type relationships."""
        # Class
        assert isinstance(worker_process_group, Generic)
        assert issubclass(WorkerProcessGroup, Generic)

        # Generics
        orig_bases = getattr(WorkerProcessGroup, "__orig_bases__", ())
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
        assert len(type_args) == 3

        # Function signatures
        create_processes_sig = inspect.signature(WorkerProcessGroup.create_processes)
        assert len(create_processes_sig.parameters) == 1
        assert "self" in create_processes_sig.parameters

        start_sig = inspect.signature(WorkerProcessGroup.start)
        assert len(start_sig.parameters) == 2
        assert "self" in start_sig.parameters
        assert "start_time" in start_sig.parameters

        request_updates_sig = inspect.signature(WorkerProcessGroup.request_updates)
        assert len(request_updates_sig.parameters) == 1
        assert "self" in request_updates_sig.parameters

        shutdown_sig = inspect.signature(WorkerProcessGroup.shutdown)
        assert len(shutdown_sig.parameters) == 1
        assert "self" in shutdown_sig.parameters

    @pytest.mark.smoke
    def test_initialization(self, worker_process_group: WorkerProcessGroup):
        """Test basic initialization of WorkerProcessGroup."""
        # Core attributes
        assert isinstance(worker_process_group.backend, MockBackend)
        expected_requests = ["request1", "request2", "request3"]
        assert list(worker_process_group.requests) == expected_requests
        assert isinstance(worker_process_group.strategy, SynchronousStrategy)
        assert isinstance(worker_process_group.constraints, dict)
        assert "max_requests" in worker_process_group.constraints
        constraint = worker_process_group.constraints["max_requests"]
        assert isinstance(constraint, MaxNumberConstraint)

        # Multiprocessing attributes (should be None initially)
        assert worker_process_group.mp_context is None
        assert worker_process_group.processes is None

        # Synchronization primitives (should be None initially)
        assert worker_process_group.startup_barrier is None
        assert worker_process_group.shutdown_event is None
        assert worker_process_group.error_event is None

        # Queues (should be None initially)
        assert worker_process_group.requests_queue is None
        assert worker_process_group.updates_queue is None
        assert worker_process_group.pending_updates_queue is None
        assert worker_process_group.pending_updates_complete is None

        # Scheduler state and tasks (should be None initially)
        assert worker_process_group.state_update_lock is None
        assert worker_process_group.scheduler_state is None
        assert worker_process_group.populate_requests_task is None
        assert worker_process_group.populate_updates_task is None

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test that invalid initialization raises appropriate errors."""
        # Test with missing required parameters
        with pytest.raises(TypeError):
            WorkerProcessGroup()

        # Create a complete set of valid parameters
        backend = MockBackend()
        requests = ["request1", "request2"]
        strategy = SynchronousStrategy()
        constraints = {"max_requests": MaxNumberConstraint(max_num=10)}

        # Test missing each required parameter one by one
        required_params = [
            "backend",
            "requests",
            "strategy",
            "constraints",
        ]

        for param_to_remove in required_params:
            kwargs = {
                "backend": backend,
                "requests": requests,
                "strategy": strategy,
                "constraints": constraints,
            }

            del kwargs[param_to_remove]

            with pytest.raises(TypeError):
                WorkerProcessGroup(**kwargs)

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("strategy", "expected_num_procs", "expected_max_conc"),
        [
            (SynchronousStrategy(), 1, 1),
            (ConcurrentStrategy(streams=3), 3, 3),
            (ThroughputStrategy(max_concurrency=6), 3, 6),
            (AsyncConstantStrategy(rate=100.0), 3, 12),
            (AsyncPoissonStrategy(rate=100.0), 3, 12),
        ],
    )
    async def test_create_processes(
        self,
        monkeypatch,
        strategy,
        expected_num_procs,
        expected_max_conc,
    ):
        # Patch required mock settings
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 3, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 12, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(worker_group, "WorkerProcess", MockWorker, raising=True)

        # Setup group to test
        backend = MockBackend()
        requests = [f"r{i}" for i in range(10)]
        constraints = {"max_requests": MaxNumberConstraint(max_num=100)}
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=strategy,
            constraints=constraints,
        )

        # Run within a reasonable time limit
        try:
            await asyncio.wait_for(group.create_processes(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("create_processes() timed out after 5 seconds")

        # Check expected attributes are created
        assert group.mp_context is not None
        assert hasattr(group.mp_context, "Barrier")
        assert hasattr(group.mp_context, "Event")
        assert hasattr(group.mp_context, "Queue")
        assert group.processes is not None
        assert len(group.processes) == expected_num_procs

        # Validate processes ran correctly
        diags: dict[int, dict] = {}
        for _ in range(expected_num_procs):
            kind, rank, payload = group.updates_queue.get(timeout=3)
            if kind == "error":
                pytest.fail(f"Worker {rank} reported error: {payload}")
            assert kind == "diag"
            diags[rank] = payload

        # Verify returned processes state
        main_pid = os.getpid()
        assert len(diags) == expected_num_procs
        for rank, payload in diags.items():
            assert payload["local_rank"] == rank
            assert payload["local_world_size"] == expected_num_procs
            assert payload["passed_barrier"] is True
            assert payload["shutdown_is_set"] is False
            assert payload["error_is_set"] is False
            assert isinstance(payload["backend_info"], dict)
            assert payload["child_pid"] != main_pid
        per_proc = math.ceil(expected_max_conc / expected_num_procs)
        expected_last = expected_max_conc - per_proc * (expected_num_procs - 1)
        for rank, payload in diags.items():
            exp_limit = per_proc if rank < expected_num_procs - 1 else expected_last
            assert payload["async_limit"] == exp_limit

        exceptions = await group.shutdown()
        assert len(exceptions) == 0, f"Shutdown encountered exceptions: {exceptions}"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_start(self, monkeypatch):
        # Patch required mock settings
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(worker_group, "WorkerProcess", MockWorker, raising=True)

        # Setup group and mimic create_processes
        backend = MockBackend()
        requests = [f"r{i}" for i in range(5)]  # to few requests, test new iter logic
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=SynchronousStrategy(),
            constraints={"max_num": MaxNumberConstraint(max_num=10)},
        )
        group.mp_context = get_context("fork")
        group.startup_barrier = group.mp_context.Barrier(2)
        group.shutdown_event = group.mp_context.Event()
        group.error_event = group.mp_context.Event()
        group.requests_queue = group.mp_context.Queue()
        group.updates_queue = group.mp_context.Queue()
        group.pending_updates_queue = culsans.Queue()
        group.pending_updates_complete = threading.Event()
        group.processes = [None]

        # Validate function runs and returns at start_time
        start_time = time.time() + 0.2
        await asyncio.wait_for(group.start(start_time), timeout=3.0)
        end_time = time.time()
        assert end_time == pytest.approx(start_time, abs=0.01)

        # Validate instance state
        assert group.state_update_lock is not None
        assert hasattr(group.state_update_lock, "acquire")
        assert group.scheduler_state is not None
        assert group.scheduler_state.num_processes == 1
        assert group.scheduler_state.start_time == start_time
        assert isinstance(group.populate_requests_task, asyncio.Task)
        assert isinstance(group.populate_updates_task, asyncio.Task)

        # Pull the queued requests
        await asyncio.sleep(0.1)
        sent_requests = []
        while True:
            await asyncio.sleep(0)
            try:
                req = group.requests_queue.get(timeout=1.0)
                sent_requests.append(req)
            except Empty:
                break
        assert len(sent_requests) == 10

        # Enqueue lifecycle updates
        for req in requests + requests:
            group.updates_queue.put(
                MsgpackEncoding.encode(
                    (
                        None,
                        req,
                        ScheduledRequestInfo[MockRequestTimings](
                            request_id=str(req),
                            status="in_progress",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=start_time,
                        ),
                    )
                )
            )
            group.updates_queue.put(
                MsgpackEncoding.encode(
                    (
                        None,
                        req,
                        ScheduledRequestInfo[MockRequestTimings](
                            request_id=str(req),
                            status="completed",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=start_time,
                        ),
                    )
                )
            )
            await asyncio.sleep(0)

        # Drain 3 updates per request (queued, started, completed)
        await asyncio.sleep(0.1)
        updates = []
        for _ in range(3 * 10):
            try:
                update = await asyncio.wait_for(
                    group.pending_updates_queue.async_get(), timeout=1.0
                )
                updates.append(update)
            except asyncio.TimeoutError:
                break
        assert len(updates) == 3 * 10

        # Ensure tasks finish
        if not group.populate_requests_task.done():
            await asyncio.wait_for(group.populate_requests_task, timeout=1.0)
        if not group.populate_updates_task.done():
            await asyncio.wait_for(group.populate_updates_task, timeout=1.0)

        # Clean up resources
        group.processes = None
        exceptions = await group.shutdown()
        assert len(exceptions) == 0, f"Shutdown encountered exceptions: {exceptions}"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_start_cancel_requests_handling(self, monkeypatch):
        """Test the start() method's async tasks handle shutdown correctly"""
        # Patch required mock settings
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(worker_group, "WorkerProcess", MockWorker, raising=True)

        # Setup group and mimic create_processes
        backend = MockBackend()
        requests = [f"req_{i}" for i in range(10)]
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=SynchronousStrategy(),
            constraints={},
        )
        group.mp_context = get_context("fork")
        group.startup_barrier = group.mp_context.Barrier(2)
        group.shutdown_event = group.mp_context.Event()
        group.error_event = group.mp_context.Event()
        group.requests_queue = group.mp_context.Queue(maxsize=1)  # Ensure saturated
        group.updates_queue = group.mp_context.Queue()
        group.pending_updates_queue = culsans.Queue()
        group.pending_updates_complete = threading.Event()
        group.processes = [None]

        # Validate function runs and returns at start_time
        start_time = time.time() + 0.1
        await asyncio.wait_for(group.start(start_time), timeout=3.0)
        end_time = time.time()
        assert end_time == pytest.approx(start_time, abs=0.01)

        # Verify tasks are running
        assert isinstance(group.populate_requests_task, asyncio.Task)
        assert isinstance(group.populate_updates_task, asyncio.Task)
        assert not group.populate_requests_task.done()
        assert not group.populate_updates_task.done()

        def _process_request():
            req, req_info = MsgpackEncoding.decode(
                group.requests_queue.get(timeout=1.0)
            )
            group.updates_queue.put(
                MsgpackEncoding.encode(
                    (
                        None,
                        req,
                        ScheduledRequestInfo[MockRequestTimings](
                            request_id=str(req),
                            status="in_progress",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=start_time,
                        ),
                    )
                )
            )
            group.updates_queue.put(
                MsgpackEncoding.encode(
                    (
                        None,
                        req,
                        ScheduledRequestInfo[MockRequestTimings](
                            request_id=str(req),
                            status="completed",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=start_time,
                        ),
                    )
                )
            )

        # Pull a few requests and push updates to ensure saturation
        for _ in range(3):
            await asyncio.sleep(0)
            _process_request()

        # Check that we've received all expected updates so far
        updates_by_request = defaultdict(list)
        while True:
            try:
                resp, req, req_info, state = await asyncio.wait_for(
                    group.pending_updates_queue.async_get(),
                    timeout=0.1,
                )
                updates_by_request[req].append(req_info.status)
            except asyncio.TimeoutError:
                break
        for index, (_, statuses) in enumerate(updates_by_request.items()):
            if index < 3:
                assert statuses == ["queued", "in_progress", "completed"]
            else:
                assert statuses == ["queued"]

        # Test that shutdown event stops the tasks
        group.shutdown_event.set()
        await asyncio.sleep(0.1)  # allow propagation
        assert group.pending_requests_complete.is_set()
        assert group.populate_requests_task.done()
        await asyncio.sleep(0.1)  # allow processing
        assert group.pending_updates_complete.is_set()
        assert group.populate_updates_task.done()

        # Check all expected pending updates and statuses processed
        while True:
            try:
                resp, req, req_info, state = await asyncio.wait_for(
                    group.pending_updates_queue.async_get(), timeout=0.1
                )
                updates_by_request[req].append(req_info.status)
            except asyncio.TimeoutError:
                break

        for index, (_, statuses) in enumerate(updates_by_request.items()):
            if index < 3:
                assert statuses == ["queued", "in_progress", "completed"]
            else:
                assert statuses == ["queued", "in_progress", "cancelled"]

        # Clean up resources
        group.processes = None
        exceptions = await group.shutdown()
        assert len(exceptions) == 0, f"Shutdown encountered exceptions: {exceptions}"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_request_updates(self, monkeypatch):
        """Test the request_updates async iterator functionality."""
        # Configure settings for controlled testing
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(
            worker_group, "WorkerProcess", MockWorkerProcessor, raising=True
        )

        # Setup group
        backend = MockBackend()
        requests = [f"req_{index}" for index in range(20)]
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=SynchronousStrategy(),
            constraints={"max_num": MaxNumberConstraint(max_num=10)},
        )

        # Mimic create_processes to set required state
        await group.create_processes()
        await group.start(time.time() + 0.05)

        # Collect all updates from request_updates iterator
        received_updates = defaultdict(list)
        received_responses = []
        count = 0
        async for resp, req, req_info, state in group.request_updates():
            assert isinstance(req_info, ScheduledRequestInfo)
            assert isinstance(state, SchedulerState)
            received_updates[req].append(req_info.status)
            if resp is not None:
                received_responses.append(resp)
            count += 1

        # Check we have all expected updates (10 requests)
        assert len(received_updates) == 10
        for index, (req, statuses, resp) in enumerate(
            zip(received_updates.keys(), received_updates.values(), received_responses)
        ):
            assert req == f"req_{index}"
            assert resp == f"response_for_req_{index}"
            assert statuses == ["queued", "in_progress", "completed"]

        # Cleanup
        await group.shutdown()
