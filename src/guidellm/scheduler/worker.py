"""
Worker process management for multi-process request scheduling and execution.

This module provides the infrastructure for managing distributed worker processes
that handle request scheduling, processing, and coordination in the GuideLLM toolkit.
It implements a multiprocessing-based architecture with asyncio/await patterns for
efficient concurrent request handling.

Classes:
    WorkerProcess: Individual worker process for handling requests with a specific
        backend.
    WorkerProcessGroup: Orchestrates multiple worker processes with shared state
        management and inter-process communication.

Functions:
    worker_sync_iterable_to_async: Utility for converting synchronous iterables to
        async with proper cancellation and error handling.
"""

import asyncio
import contextlib
import math
import queue
import threading
import time
from asyncio import Task
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager, Queue
from multiprocessing.managers import BaseManager
from multiprocessing.synchronize import Barrier, Event
from threading import BrokenBarrierError
from threading import Event as ThreadingEvent
from typing import Any, Callable, Generic, Literal, Optional, Union

from guidellm.config import settings
from guidellm.scheduler.constraints import CallableConstraint
from guidellm.scheduler.objects import (
    BackendT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import ScheduledRequestTimings, SchedulingStrategy

__all__ = [
    "WorkerProcess",
    "WorkerProcessGroup",
    "worker_sync_iterable_to_async",
]


def _infinite_iter():
    while True:
        yield None


async def worker_sync_iterable_to_async(
    iter_func: Union[Callable, Literal["infinite"]],
    exit_events: Optional[dict[str, Event]] = None,
    exit_barrier: Optional[Barrier] = None,
    poll_interval: float = 0.1,
    *args,
    **kwargs,
) -> tuple[Union[Literal["completed", "canceled", "barrier"], str], Any]:
    """
    Convert a synchronous iterable to async execution with proper lifecycle management.

    This utility function allows synchronous iterables to be executed in an async
    context while respecting process synchronization primitives like barriers and
    events. It handles cancellation, error propagation, and graceful shutdown
    scenarios.

    :param iter_func: The iterable function to execute. If None, creates an infinite
        iterator and only polls for cancellation, events, and barriers.
    :param exit_events: Optional dictionary of Events to monitor for termination
        signals to exit the loop.
    :param exit_barrier: Optional Barrier to wait on before exiting the loop.
    :param poll_interval: Time in seconds between iteration cycles and barrier checks.
    :param args: Positional arguments to pass to iter_func.
    :param kwargs: Keyword arguments to pass to iter_func.
    :return: A tuple containing the stop reason and the last item yielded by the the
        iterator before termination.
    :raises RuntimeError: If error_event is set during iteration.
    :raises asyncio.CancelledError: If the async operation is cancelled.
    """
    if iter_func == "infinite":
        iter_func = _infinite_iter

    if exit_events is None:
        exit_events = {}

    canceled_event = ThreadingEvent()
    exit_events["canceled"] = canceled_event

    def _run_thread():
        nonlocal canceled_event
        stop_reason = "completed"
        last_item = None
        for item in iter_func(*args, **kwargs):
            last_item = item

            for event_name, event in exit_events.items():
                if event.is_set():
                    stop_reason = event_name
                    break

            if exit_barrier is not None:
                try:
                    exit_barrier.wait(timeout=poll_interval)
                    stop_reason = "barrier"
                    break
                except BrokenBarrierError:
                    pass
            else:
                time.sleep(poll_interval)

        return stop_reason, last_item

    try:
        return await asyncio.to_thread(_run_thread)
    except asyncio.CancelledError:
        canceled_event.set()
        raise


class WorkerProcess(Generic[BackendT, RequestT, ResponseT]):
    """
    Individual worker process for handling request processing with a specific backend.

    This class represents a single worker process designed to operate within a
    multi-process scheduler system. It manages the lifecycle of requests from
    queue consumption through backend processing and updates publication.
    The worker maintains proper synchronization with other processes and handles
    error conditions gracefully.

    :param local_rank: The process number/index for this worker within the group.
    :param local_world_size: Total number of worker processes in the group.
    :param async_limit: Maximum number of concurrent requests this worker can handle.
    :param startup_barrier: Multiprocessing barrier for coordinated startup.
    :param shutdown_event: Event to signal and monitor for graceful shutdown/stopping.
    :param error_event: Event to signal/monitor error conditions across processes.
    :param requests_queue: Queue for receiving requests to process.
    :param updates_queue: Queue for publishing processing updates, including
        request queued, request processing start, and request completion.
    :param backend: Backend instance for processing the requests through
        utilizing backend.resolve function. Additionally, backend.process_startup,
        backend.validate, and backend.process_shutdown methods are called
        for lifecycle management within the worker.
    :param request_timings: ScheduledRequestTimings instance for designating when to
        start processing the next request.
    :param poll_intervals: Time interval for polling operations.
    """

    def __init__(
        self,
        local_rank: int,
        local_world_size: int,
        async_limit: int,
        startup_barrier: Barrier,
        shutdown_event: Event,
        error_event: Event,
        requests_queue: Queue[tuple[RequestT, ScheduledRequestInfo]],
        updates_queue: Queue[
            tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo]
        ],
        backend: BackendT,
        request_timings: ScheduledRequestTimings,
        poll_intervals: float,
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
        self.pending_request: Optional[tuple[RequestT, ScheduledRequestInfo]] = None

    def run(self):
        """
        Main entry point for the worker process execution.

        Initializes the asyncio event loop and starts the worker's async operations.
        This method is designed to be passed into a ProcessPoolExecutor by reference
        in the main process, allowing it to run in a separate worker process.

        :raises RuntimeError: If the worker encounters an unrecoverable error during
            execution. The error event is set before raising to notify other processes.
        """
        try:
            asyncio.run(self.run_async())
        except Exception as exc:  # noqa: BLE001
            self.error_event.set()
            raise RuntimeError(
                f"Worker process {self.local_rank} encountered an error: {exc}"
            ) from exc

    async def run_async(self):
        completed_tasks, pending_tasks = await asyncio.wait(
            [
                asyncio.create_task(self.run_async_stop_processing()),
                asyncio.create_task(self.run_async_requests_processing()),
            ],
            return_when=asyncio.FIRST_EXCEPTION,
        )

        with contextlib.suppress(asyncio.CancelledError):
            for task in pending_tasks:
                task.cancel()
                await task

            for task in completed_tasks:
                if task.exception():
                    raise task.exception()

    async def run_async_stop_processing(self):
        exit_reason, _ = await worker_sync_iterable_to_async(
            iter_func="infinite",
            exit_events={
                "error_event": self.error_event,
                "shutdown_event": self.shutdown_event,
            },
            poll_interval=self.poll_intervals,
        )

        if exit_reason == "error_event":
            raise RuntimeError(
                f"Worker process {self.local_rank} received error signal."
            )
        elif exit_reason == "shutdown_event":
            raise asyncio.CancelledError(
                f"Worker process {self.local_rank} received shutdown signal."
            )
        else:
            raise RuntimeError(
                f"Worker process {self.local_rank} received unexpected exit reason: "
                f"{exit_reason}"
            )

    async def run_async_requests_processing(self):
        # Ensure backend is ready on this worker
        await self.backend.process_startup()
        await self.backend.validate()

        # Wait for all processes to be ready before starting
        barrier_exit_reason, _ = await worker_sync_iterable_to_async(
            iter_func="infinite",
            barrier=self.startup_barrier,
        )
        if barrier_exit_reason != "barrier":
            raise RuntimeError(
                f"Worker process {self.local_rank} failed to synchronize at startup: "
                f"{barrier_exit_reason}"
            )

        async_semaphore = asyncio.Semaphore(self.async_limit)
        pending_tasks = []

        def _task_done(task):
            pending_tasks.remove(task)
            async_semaphore.release()

        try:
            while True:
                await async_semaphore.acquire()
                (
                    request,
                    request_info,
                    request_history,
                ) = await self._next_ready_request()
                if isinstance(request, Iterable):
                    raise NotImplementedError(
                        "Multi-turn requests are not supported yet"
                    )

                request_task = asyncio.create_task(
                    self._process_request(request, request_info, request_history)
                )
                pending_tasks.append(request_task)
                request_task.add_done_callback(_task_done)
                await asyncio.sleep(0)  # Yield control to the event loop
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                await self._cancel_queued_requests()
                await self.backend.process_shutdown()

            raise

    async def _next_ready_request(
        self,
    ) -> tuple[
        Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]],
        ScheduledRequestInfo,
        list[tuple[RequestT, ResponseT]],
    ]:
        while True:
            try:
                timings_offset = self.request_timings.next_offset()
                request, request_info = self.requests_queue.get_nowait()
                request_info.scheduler_timings.dequeued = time.time()
                request_info.status = "pending"

                target_start = request_info.scheduler_start_time + timings_offset
                if target_start > time.time():
                    await asyncio.sleep(target_start - time.time())

                return request, request_info, []
            except asyncio.QueueEmpty:
                await asyncio.sleep(self.poll_intervals)
                continue

    async def _process_request(
        self,
        request: RequestT,
        request_info: ScheduledRequestInfo,
        request_history: list[tuple[RequestT, ResponseT]],
    ):
        last_response: Optional[ResponseT] = None
        request_info.status = "in_progress"
        request_info.scheduler_timings.resolve_start = time.time()
        self.updates_queue.put(
            (last_response, request, request_info.model_copy(deep=True))
        )
        cancelled = False

        try:
            async for response in self.backend.resolve(
                request, request_info, request_history
            ):
                last_response = response

            request_info.status = "completed"
        except asyncio.CancelledError:
            request_info.status = "cancelled"
            cancelled = True
        except Exception as exc:
            request_info.status = "error"
            request_info.error = str(exc)

        request_info.scheduler_timings.resolve_end = time.time()
        self.updates_queue.put((last_response, request, request_info))
        self.request_timings.update_completed(request_info)
        if cancelled:
            raise asyncio.CancelledError

    async def _cancel_queued_requests(self):
        async def _request_gen() -> AsyncIterator[
            tuple[RequestT, ScheduledRequestInfo]
        ]:
            if self.pending_request:
                yield self.pending_request
                self.pending_request = None

            try:
                while True:
                    yield self.requests_queue.get(timeout=self.poll_intervals)
            except queue.Empty:
                # Assume all requests were on the queue already, safe to stop
                pass

            yield None, None

        async for request, request_info in _request_gen():
            if request is None:
                break

            # Update in progress first; ensure the same updates for all requests
            request_info.status = "in_progress"
            request_info.scheduler_timings.resolve_start = time.time()
            self.updates_queue.put((None, request, request_info.model_copy(deep=True)))

            request_info.status = "cancelled"
            request_info.scheduler_timings.resolve_end = time.time()
            self.updates_queue.put((None, request, request_info.model_copy(deep=True)))


class WorkerProcessGroup(Generic[BackendT, RequestT, ResponseT]):
    """
    Orchestrates multiple worker processes with shared state management and
    coordination.

    This class manages a group of worker processes that collectively handle request
    processing for a distributed scheduling system. It provides centralized control
    over process lifecycle, request distribution, response collection, and state
    synchronization across all workers.

    The process group handles dynamic scaling, load balancing, constraint evaluation,
    and graceful shutdown coordination. It maintains shared queues for inter-process
    communication and tracks the overall system state including throughput metrics
    and constraint satisfaction.

    Example:
    ::
        from guidellm.scheduler.worker import WorkerProcessGroup

        group = WorkerProcessGroup(
            backend=my_backend,
            requests=request_iterable,
            strategy=scheduling_strategy,
            constraints={"max_requests": max_requests_constraint}
        )

        await group.create_processes()
        await group.start(time.time())

        async for response, request, info, state in group.request_updates():
            # Process each completed request
            handle_response(response, request, info, state)

        await group.shutdown()

    :param backend: Backend instance for processing requests.
    :param requests: Iterable of requests to process.
    :param strategy: Scheduling strategy configuration.
    :param constraints: Dictionary of named constraints for controlling execution.
    """

    def __init__(
        self,
        backend: BackendT,
        requests: Iterable[RequestT],
        strategy: SchedulingStrategy,
        constraints: dict[str, CallableConstraint],
    ):
        self.backend = backend
        self.requests = requests
        self.strategy = strategy
        self.constraints = constraints

        # multiprocessing attributes
        self.manager: BaseManager = None
        self.executor: ProcessPoolExecutor = None
        self.processes: list[Future] = None

        # synchronization primitives
        self.startup_barrier: Barrier = None
        self.shutdown_event: Event = None
        self.error_event: Event = None

        # queues for communication
        self.process_requests_queue: Queue[tuple[RequestT, ScheduledRequestInfo]] = None
        self.process_updates_queue: Queue[
            tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo]
        ] = None
        self.async_updates_queue: queue.Queue[
            tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo]
        ] = None

        # scheduler state and request management
        self.state_update_lock: threading.Lock = None
        self.scheduler_state: SchedulerState = None
        self.populate_requests_task: Task = None
        self.populate_updates_task: Task = None

    async def create_processes(self):
        """
        Initialize and start the worker process group.

        Sets up the multiprocessing infrastructure including process pool executor,
        synchronization primitives, communication queues, and individual worker
        processes. Determines optimal process count and concurrency limits based
        on strategy constraints, backend capabilities, and system configuration.

        :raises RuntimeError: If process initialization fails or if workers encounter
            errors during startup.
        """
        # Processes limits and params
        num_processes = min(
            self.strategy.processes_limit or math.inf,
            self.backend.processes_limit or math.inf,
            settings.max_worker_processes,
        )
        max_request_concurrency = min(
            self.strategy.requests_limit or math.inf,
            self.backend.requests_limit or math.inf,
            settings.max_concurrency,
        )
        per_process_max_concurrency = math.ceil(max_request_concurrency / num_processes)
        max_queued_requests = (  # Add one for each process to ensure readiness
            max_request_concurrency + num_processes
        )

        # Initialize multiprocessing components
        self.manager = Manager()
        self.executor = ProcessPoolExecutor()
        self.startup_barrier = self.manager.Barrier(num_processes + 1)
        self.shutdown_event = self.manager.Event()
        self.error_event = self.manager.Event()
        self.process_requests_queue = self.manager.Queue(maxsize=max_queued_requests)
        self.process_updates_queue = self.manager.Queue()
        self.async_updates_queue = asyncio.Queue()

        # Initialize worker processes
        self.processes = []
        for process_rank in range(num_processes):
            worker = WorkerProcess(
                local_rank=process_rank,
                local_world_size=num_processes,
                async_limit=(
                    per_process_max_concurrency
                    if process_rank < num_processes - 1
                    else max_request_concurrency
                    - (per_process_max_concurrency * (num_processes - 1))
                ),
                startup_barrier=self.startup_barrier,
                shutdown_event=self.shutdown_event,
                error_event=self.error_event,
                requests_queue=self.process_requests_queue,
                updates_queue=self.process_updates_queue,
                backend=self.backend,
                strategy=self.strategy,
                poll_intervals=settings.scheduler_poll_interval,
            )
            future = self.executor.submit(worker.run)
            self.processes.append(future)

        startup_exit_reason, _ = await worker_sync_iterable_to_async(
            iter_func="infinite",
            exit_events={
                "error_event": self.error_event,
                "shutdown_event": self.shutdown_event,
            },
            exit_barrier=self.startup_barrier,
            poll_interval=settings.scheduler_poll_interval,
        )
        if startup_exit_reason != "barrier":
            raise RuntimeError(
                "Worker process group startup failed with exit reason: "
                f"{startup_exit_reason}"
            )

    async def start(self, start_time: float):
        """
        Begin request processing at the specified start time.

        Initializes the scheduler state, creates background tasks for request
        population and response handling, and waits until the specified start
        time before beginning operations.

        :param start_time: Unix timestamp when processing should begin.
        :raises RuntimeError: If workers encounter errors during startup or if
            initialization fails.
        """
        self.state_update_lock = threading.Lock()
        self.scheduler_state = SchedulerState(
            node_id=0,  # Process group node identifier
            num_processes=len(self.processes),
            start_time=start_time,
        )
        self.populate_requests_task = asyncio.create_task(
            self._populate_requests(start_time)
        )
        self.populate_updates_task = asyncio.create_task(self._populate_updates())

        await asyncio.sleep(start_time - time.time())
        self._raise_if_error()

    async def request_updates(
        self,
    ) -> AsyncIterator[
        tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo, SchedulerState]
    ]:
        """
        Yield request processing updates as they become available.

        Returns an async iterator that yields tuples containing request processing
        updates, including the response (if available), the original request,
        scheduling metadata, and the current scheduler state.
        Updates occur on request queued, request processing start, and
        request completion, allowing for real-time monitoring and processing.
        The iterator continues until all requests have been processed or an error
        occurs.

        :return: Async iterator yielding (response, request, request_info, state)
            tuples for each request update; response is None until the request
            processing is complete.
        :raises RuntimeError: If workers encounter unrecoverable errors during
            processing.
        """
        last_state: SchedulerState = None
        last_check_time = time.time()
        shutdown = self.shutdown_event.is_set()

        while (
            not shutdown
            or last_state is None
            or last_state.processed_requests < last_state.created_requests
        ):
            try:
                (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) = self.async_updates_queue.get_nowait()
                last_state = scheduler_state

                yield response, request, request_info, scheduler_state
            except queue.Empty:
                await asyncio.sleep(settings.scheduler_poll_interval)

            if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                self._raise_if_error()
                shutdown = self.shutdown_event.is_set()
                last_check_time = time.time()

    async def shutdown(self) -> list[Exception]:
        """
        Gracefully shut down the worker process group and clean up resources.

        Performs a safe shutdown of all worker processes, background tasks,
        and multiprocessing resources to release system resources.
        Returns any errors encountered during the shutdown process,
        either during shutdown or previously occurred in worker processes.

        :return: A list of any exceptions raised while running or shutting down
            the worker processes. If no errors occurred, returns an empty list.
        """
        exceptions = []

        if self.shutdown_event is not None:
            self.shutdown_event.set()
            self.shutdown_event = None

        with contextlib.suppress(asyncio.CancelledError):
            try:
                for task in [
                    self.populate_requests_task,
                    self.populate_responses_task,
                ]:
                    if task is not None and not task.done():
                        task.cancel()
                        await task
            except Exception as err:
                exceptions.append(err)
            self.populate_requests_task = None
            self.populate_responses_task = None

            for process in self.processes:
                try:
                    asyncio.to_thread(process.result)
                except Exception as err:
                    exceptions.append(err)
            self.processes = None

        self.executor.shutdown(wait=True)
        self.executor = None
        self.manager.shutdown()
        self.manager = None

        self.startup_barrier = None
        self.error_event = None
        self.process_requests_queue = None
        self.process_updates_queue = None
        self.async_responses_queue = None

        return exceptions

    async def _raise_if_error(self):
        if self.error_event.is_set():
            raise RuntimeError(
                "error_event is set in WorkerProcessGroup, "
                "indicating an error occurred in one of the worker processes."
            )

    def _populate_requests(self, start_time: float):
        last_check_time = time.time()
        continue_requests = True

        while continue_requests:
            for request in self.requests:
                request_info = ScheduledRequestInfo(
                    request_id=getattr(
                        request, "id_", getattr(request, "id", id(request))
                    ),
                    status="queued",
                    scheduler_start_time=start_time,
                )
                _, continue_requests, _ = self._update_scheduler_state(request_info)
                self.process_requests_queue.put((request, request_info))

                if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                    if self.shutdown_event.is_set() or self.error_event.is_set():
                        continue_requests = False
                    last_check_time = time.time()

                if not continue_requests:
                    break

    def _populate_updates(self):
        last_check_time = time.time()

        while True:
            try:
                (response, request, request_info) = self.process_updates_queue.get(
                    timeout=settings.scheduler_poll_interval
                )
                scheduler_state, _, continue_processing = self._update_scheduler_state(
                    request_info
                )
                self.async_updates_queue.put(
                    (response, request, request_info, scheduler_state)
                )
                if not continue_processing:
                    self.shutdown_event.set()
                    if (
                        scheduler_state.processed_requests
                        >= scheduler_state.created_requests
                    ):
                        # Ensure we've processed all updates before exiting
                        break
            except queue.Empty:
                continue

            if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                if self.error_event.is_set():
                    break
                last_check_time = time.time()

    def _update_scheduler_state(
        self, request_info: ScheduledRequestInfo
    ) -> tuple[SchedulerState, bool, bool]:
        """
        :param request_info: The request information to update the state with.
        :return: A tuple containing the updated scheduler state,
            whether to continue adding requests,
            and whether to continue processing requests.
        """
        with self.state_update_lock:
            if request_info.status == "queued":
                self.scheduler_state.created_requests += 1
                self.scheduler_state.queued_requests += 1
            elif request_info.status == "pending":
                self.scheduler_state.queued_requests -= 1
                self.scheduler_state.pending_requests += 1
            elif request_info.status == "in_progress":
                self.scheduler_state.pending_requests -= 1
                self.scheduler_state.processing_requests += 1
            elif request_info.status == "completed":
                self.scheduler_state.processing_requests -= 1
                self.scheduler_state.processed_requests += 1
                self.scheduler_state.successful_requests += 1
            elif request_info.status == "errored":
                self.scheduler_state.processing_requests -= 1
                self.scheduler_state.processed_requests += 1
                self.scheduler_state.errored_requests += 1
            elif request_info.status == "cancelled":
                self.scheduler_state.processing_requests -= 1
                self.scheduler_state.processed_requests += 1
                self.scheduler_state.cancelled_requests += 1
            else:
                raise ValueError(
                    f"Unknown request status: {request_info.status}. "
                    "Supported statuses are: queued, pending, in_progress, "
                    "completed, errored, cancelled."
                )

            queuing_stopped = self.scheduler_state.end_queuing_time is not None
            processing_stopped = self.scheduler_state.end_processing_time is not None

            for key, constraint in self.constraints.items():
                action = constraint(self.scheduler_state, request_info)
                if not queuing_stopped and action.request_queuing == "stop":
                    # Ensure metadata for final state only includes the first
                    # constraints that signaled for stopping queuing
                    self.scheduler_state.end_queuing_constraints[key] = action.metadata
                    self.scheduler_state.end_queuing_time = time.time()
                if not processing_stopped and action.request_processing in (
                    "stop_local",
                    "stop_all",
                ):
                    # Ensure metadata for final state only includes the first
                    # constraints that signaled for stopping processing
                    self.scheduler_state.end_processing_constraints[key] = (
                        action.metadata
                    )
                    self.scheduler_state.end_processing_time = time.time()

            state_copy: SchedulerState = self.scheduler_state.model_copy(deep=True)

        return (
            state_copy,
            state_copy.end_queuing_time is None,
            state_copy.end_processing_time is None,
        )
