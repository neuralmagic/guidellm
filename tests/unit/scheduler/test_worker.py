import asyncio
import contextlib
import inspect
import math
import threading
import time
from collections import defaultdict
from multiprocessing import Barrier, Event, Queue
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from queue import Empty
from typing import Any, Callable, Generic, Literal, Optional
from unittest.mock import AsyncMock, patch

import pytest

from guidellm.scheduler import (
    BackendInterface,
    LastCompletionRequestTimings,
    MeasuredRequestTimings,
    ScheduledRequestInfo,
    ScheduledRequestTimings,
    WorkerProcess,
)
from guidellm.scheduler.strategy import (
    ConstantRateRequestTimings,
    NoDelayRequestTimings,
    PoissonRateRequestTimings,
)
from guidellm.utils import MsgpackEncoding, random


class MockRequestTimings(MeasuredRequestTimings):
    """Mock timing implementation for testing."""


class MockBackend(BackendInterface):
    """Mock backend for testing worker functionality."""

    def __init__(
        self,
        delay: float = 0.01,
        should_fail: bool = False,
        request_error_rate: float = 0.0,
    ):
        self.delay = delay
        self.should_fail = should_fail
        self.request_error_rate = request_error_rate
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
        await asyncio.sleep(0.1)
        self.process_shutdown_called = True

    async def resolve(self, request, request_info, request_history):
        self.resolve_called = True
        await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("Mock resolve failed")
        if self.request_error_rate > 0.0 and random.random() < self.request_error_rate:
            raise RuntimeError("Mock resolve failed")
        yield f"response_for_{request}", request_info


class TestWorkerProcess:
    """Test suite for WorkerProcess class."""

    @pytest.fixture
    def worker_process(self):
        """Create a WorkerProcess instance for testing."""
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
        # Note: consider parameterizing this to improve coverage and readability

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
        """
        Test that run method functions as expected (calls run_async, handles errors)
        """
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

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("stop_action", "req_action"),
        [
            ("complete_short", "complete_short"),
            ("complete_long", "error"),
            ("error", "complete_long"),
            ("error", "error"),
            ("complete_long", "cancel"),
            ("cancel", "complete_long"),
            ("cancel", "cancel"),
        ],
    )
    async def test_run_async(  # noqa: C901
        self,
        worker_process: WorkerProcess,
        stop_action: Literal["complete_short", "complete_long", "error", "cancel"],
        req_action: Literal["complete_short", "complete_long", "error", "cancel"],
    ):
        def make_task(action: str, state: dict):
            loops = {"error": 1, "cancel": 2, "complete_short": 3, "complete_long": 50}[
                action
            ]

            async def _run(self):
                state.update(called=True, iterations=0)
                try:
                    for _ in range(loops):
                        await asyncio.sleep(0.01)
                        state["iterations"] += 1
                    if action == "error":
                        state["errored"] = True
                        raise RuntimeError(state["error_message"])
                    if action == "cancel":
                        state["cancelled"] = True
                        raise asyncio.CancelledError(state["cancel_message"])
                    if action == "complete_short":
                        state["completed_short"] = True
                    if action == "complete_long":
                        state["completed_long"] = True
                except asyncio.CancelledError:
                    state["cancelled"] = True
                    raise

            return _run, loops

        def init_state(prefix):
            return {
                "called": False,
                "iterations": 0,
                "completed_short": False,
                "completed_long": False,
                "errored": False,
                "cancelled": False,
                "error_message": f"{prefix} processing error",
                "cancel_message": f"{prefix} processing cancelled",
            }

        stop_state, req_state = init_state("Stop"), init_state("Requests")
        stop_fn, stop_loops = make_task(stop_action, stop_state)
        req_fn, req_loops = make_task(req_action, req_state)

        expected_exc = RuntimeError if "error" in {stop_action, req_action} else None
        with (
            patch.object(
                type(worker_process), "run_async_stop_processing", new=stop_fn
            ),
            patch.object(
                type(worker_process), "run_async_requests_processing", new=req_fn
            ),
        ):
            if expected_exc:
                with pytest.raises(expected_exc):
                    await worker_process.run_async()
            else:
                await worker_process.run_async()

        assert stop_state["called"]
        assert req_state["called"]

        # build unified expected outcome table
        def is_long(a):
            return a == "complete_long"

        def is_short(a):
            return a in {"complete_short", "error", "cancel"}

        expectations = {
            "stop": {
                "errored": stop_action == "error",
                "cancelled": stop_action == "cancel"
                or (is_short(req_action) and is_long(stop_action))
                or (req_action == "error" and is_long(stop_action)),
            },
            "req": {
                "errored": req_action == "error",
                "cancelled": req_action == "cancel"
                or (is_short(stop_action) and is_long(req_action))
                or (stop_action == "error" and is_long(req_action)),
            },
        }

        # assert final state matches expectations
        for label, (state, action) in {
            "stop": (stop_state, stop_action),
            "req": (req_state, req_action),
        }.items():
            if expectations[label]["errored"]:
                assert state["errored"]
            if expectations[label]["cancelled"]:
                assert state["cancelled"]
            if action.startswith("complete_") and not expectations[label]["cancelled"]:
                key = (
                    "completed_short"
                    if action == "complete_short"
                    else "completed_long"
                )
                assert state[key]

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop_action",
        ["error_event", "shutdown_event", "cancel_event"],
    )
    async def test_run_async_stop_processing(
        self, worker_process: WorkerProcess, stop_action
    ):
        # ensure initial state
        assert not worker_process.error_event.is_set()
        assert not worker_process.shutdown_event.is_set()

        action = stop_action
        early_check_delay = 0.01
        trigger_delay = 0.05

        task = asyncio.create_task(worker_process.run_async_stop_processing())
        time_start = time.time()
        await asyncio.sleep(early_check_delay)
        assert not task.done(), "Task finished before any stop signal was triggered"

        async def trigger():
            await asyncio.sleep(trigger_delay - early_check_delay)
            if action == "error_event":
                worker_process.error_event.set()
            elif action == "shutdown_event":
                worker_process.shutdown_event.set()
            elif action == "cancel_event":
                task.cancel()

        trigger_task = asyncio.create_task(trigger())

        if action == "error_event":
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(task, timeout=1.0)
        elif action in {"shutdown_event", "cancel_event"}:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        else:
            raise ValueError(f"Unknown stop action: {action}")

        await asyncio.gather(trigger_task, return_exceptions=True)

        # validate correct ending states
        elapsed = time.time() - time_start
        assert elapsed >= trigger_delay - 0.01, (
            "Task completed too early: "
            f"elapsed={elapsed:.3f}s < trigger={trigger_delay:.3f}s"
        )
        if action == "error_event":
            assert worker_process.error_event.is_set()
            assert not worker_process.shutdown_event.is_set()
        elif action == "shutdown_event":
            assert worker_process.shutdown_event.is_set()
            assert not worker_process.error_event.is_set()
        elif action == "cancel_event":
            assert not worker_process.error_event.is_set()
            assert not worker_process.shutdown_event.is_set()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_timings_const", "async_limit"),
        [
            (lambda: LastCompletionRequestTimings(), 1),
            (lambda: PoissonRateRequestTimings(rate=10000), 2),
            (lambda: ConstantRateRequestTimings(rate=10000), 3),
            (lambda: NoDelayRequestTimings(), 4),
        ],
    )
    async def test_run_async_requests_processing(  # noqa: C901
        self,
        request_timings_const: Callable[[], ScheduledRequestTimings],
        async_limit: int,
    ):
        startup_barrier = Barrier(2)
        requests_queue = Queue()
        updates_queue = Queue()
        backend = MockBackend(delay=0.001)
        worker_process = WorkerProcess(
            local_rank=0,
            local_world_size=1,
            async_limit=async_limit,
            startup_barrier=startup_barrier,
            shutdown_event=Event(),
            error_event=Event(),
            requests_queue=requests_queue,
            updates_queue=updates_queue,
            backend=backend,
            request_timings=request_timings_const(),
            poll_intervals=0.01,
        )

        def _trip_barrier_later():
            time.sleep(0.02)
            with contextlib.suppress(RuntimeError):
                # barrier may be aborted (suppressed) during cancellation
                worker_process.startup_barrier.wait(timeout=1.0)

        threading.Thread(target=_trip_barrier_later, daemon=True).start()

        run_task = asyncio.create_task(worker_process.run_async_requests_processing())
        await asyncio.sleep(0.05)  # small delay to allow start up first

        # validate start up
        assert worker_process.backend.process_startup_called
        assert worker_process.backend.validate_called
        assert worker_process.pending_requests_queue is not None
        assert worker_process.pending_updates_queue is not None
        assert worker_process.startup_completed

        # ensure full processing of requests
        for index in range(20):
            requests_queue.put(
                MsgpackEncoding.encode(
                    (
                        f"req-{index}",
                        ScheduledRequestInfo[MeasuredRequestTimings](
                            request_id=f"req-{index}",
                            status="queued",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=time.time(),
                        ),
                    )
                )
            )

        updates = []
        num_failures = 0
        while True:
            try:
                update_message = updates_queue.get_nowait()
                updates.append(MsgpackEncoding.decode(update_message))
            except Empty:
                num_failures += 1
                if num_failures > 50:
                    break
                await asyncio.sleep(0.05)

        # validate updates are correct for each request
        assert len(updates) == 40
        per_request = defaultdict(dict)
        for update in updates:
            response, request, info = update
            if info.status == "in_progress":
                per_request[info.request_id]["start"] = (response, request, info)
                per_request[info.request_id]["targeted_start"] = (
                    info.scheduler_timings.targeted_start
                )
                per_request[info.request_id]["resolve_start"] = (
                    info.scheduler_timings.resolve_start
                )
            elif info.status == "completed":
                per_request[info.request_id]["complete"] = (response, request, info)
                per_request[info.request_id]["resolve_end"] = (
                    info.scheduler_timings.resolve_end
                )
        assert len(per_request) == 20
        assert all(
            "start" in parts and "complete" in parts for parts in per_request.values()
        )

        # validate request times match expected
        last_targeted_start = -1 * math.inf
        for index in range(20):
            targeted_start = per_request[f"req-{index}"]["targeted_start"]
            resolve_start = per_request[f"req-{index}"]["resolve_start"]
            resolve_end = per_request[f"req-{index}"]["resolve_end"]
            assert targeted_start >= last_targeted_start
            assert targeted_start < resolve_start
            assert resolve_start == pytest.approx(targeted_start)
            assert resolve_end == pytest.approx(resolve_start + backend.delay)

        # Validate concurrency limits are respected
        events = []
        for req_id in per_request:
            events.append((per_request[req_id]["resolve_start"], 1))
            events.append((per_request[req_id]["resolve_end"], -1))
        events.sort()
        max_concurrent = concurrent = 0
        for _, delta in events:
            concurrent += delta
            max_concurrent = max(max_concurrent, concurrent)
        assert max_concurrent <= async_limit

        # validate cancellation
        backend.delay = 10
        # max concurrent for backend + 2 queued for backend
        num_cancel_tasks = (async_limit + 2) * 2
        for index in range(20, 20 + num_cancel_tasks):
            requests_queue.put(
                MsgpackEncoding.encode(
                    (
                        f"req-{index}",
                        ScheduledRequestInfo[MeasuredRequestTimings](
                            request_id=f"req-{index}",
                            status="queued",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=time.time(),
                        ),
                    )
                )
            )
        await asyncio.sleep(0.5)
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)
        assert worker_process.backend.process_shutdown_called
        assert worker_process.pending_requests_queue is None
        assert worker_process.pending_updates_queue is None

        # validate canceled tasks
        updates = []
        num_failures = 0
        while True:
            try:
                update_message = updates_queue.get_nowait()
                updates.append(MsgpackEncoding.decode(update_message))
            except Empty:
                num_failures += 1
                if num_failures > 50:
                    break
                await asyncio.sleep(0.05)
        # Ensure we get all updates we expected (async_limit for pending + 2 for queued)
        assert len(updates) >= 2 * (async_limit + 2)
        # Ensure we didn't process all requests on the queue and shutdown early
        assert len(updates) < 2 * 2 * (async_limit + 2)

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_timings_const", "async_limit", "request_error_rate"),
        [
            (lambda: LastCompletionRequestTimings(), 1, 0.1),
            (lambda: PoissonRateRequestTimings(rate=10000), 2, 0.2),
            (lambda: ConstantRateRequestTimings(rate=10000), 3, 0.3),
            (lambda: NoDelayRequestTimings(), 4, 0.4),
        ],
    )
    def test_run_lifecycle(
        self,
        request_timings_const: Callable[[], ScheduledRequestTimings],
        async_limit: int,
        request_error_rate: float,
    ):
        backend = MockBackend(
            delay=0.01,
            request_error_rate=request_error_rate,
        )
        startup_barrier = Barrier(2)
        shutdown_event = Event()
        requests_queue = Queue()
        updates_queue = Queue()
        backend = MockBackend(delay=0.001)
        worker_process = WorkerProcess(
            local_rank=0,
            local_world_size=1,
            async_limit=async_limit,
            startup_barrier=startup_barrier,
            shutdown_event=shutdown_event,
            error_event=Event(),
            requests_queue=requests_queue,
            updates_queue=updates_queue,
            backend=backend,
            request_timings=request_timings_const(),
            poll_intervals=0.01,
        )

        def _background_thread():
            time.sleep(0.1)  # delay for startup
            startup_barrier.wait()

            for index in range(20):
                requests_queue.put(
                    MsgpackEncoding.encode(
                        (
                            f"req-{index}",
                            ScheduledRequestInfo[MeasuredRequestTimings](
                                request_id=f"req-{index}",
                                status="queued",
                                scheduler_node_id=0,
                                scheduler_process_id=0,
                                scheduler_start_time=time.time(),
                            ),
                        )
                    )
                )

            time.sleep(1.0)  # delay for processing
            shutdown_event.set()

        threading.Thread(target=_background_thread).start()
        worker_process.run()

        updates = []
        while True:
            try:
                update_message = updates_queue.get_nowait()
                updates.append(MsgpackEncoding.decode(update_message))
            except Empty:
                break

        # Validate updates
        assert len(updates) == 40
        per_request = defaultdict(dict)
        for update in updates:
            response, request, info = update
            if info.status == "in_progress":
                per_request[info.request_id]["start"] = (response, request, info)
                per_request[info.request_id]["targeted_start"] = (
                    info.scheduler_timings.targeted_start
                )
                per_request[info.request_id]["resolve_start"] = (
                    info.scheduler_timings.resolve_start
                )
            elif info.status == "completed":
                per_request[info.request_id]["complete"] = (response, request, info)
                per_request[info.request_id]["resolve_end"] = (
                    info.scheduler_timings.resolve_end
                )
        assert len(per_request) == 20
        assert all(
            "start" in parts and "complete" in parts for parts in per_request.values()
        )
