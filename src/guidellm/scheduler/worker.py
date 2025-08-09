"""
Worker process management for multi-process request scheduling and execution.

Provides infrastructure for managing individual worker processes that handle
request scheduling, processing, and coordination in multi-process environments.

Classes:
    WorkerProcess: Individual worker process for request processing and coordination.

Functions:
    worker_sync_iterable_to_async: Convert synchronous iterables to async execution
        with lifecycle management and process synchronization.
"""

import asyncio
import contextlib
import queue
import time
from collections.abc import AsyncIterator, Iterable
from multiprocessing import Queue
from multiprocessing.synchronize import Barrier, Event
from threading import BrokenBarrierError
from threading import Event as ThreadingEvent
from typing import Any, Callable, Generic, Literal, Optional, Union

from guidellm.scheduler.objects import (
    BackendT,
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
)
from guidellm.scheduler.strategy import ScheduledRequestTimings

__all__ = ["WorkerProcess", "worker_sync_iterable_to_async"]


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
    Convert synchronous iterable to async execution with lifecycle management.

    Enables synchronous iterables to execute within async contexts while respecting
    multiprocessing synchronization primitives like barriers and events.

    :param iter_func: Iterable function to execute, or "infinite" for polling.
    :param exit_events: Optional event mappings for monitoring termination signals.
    :param exit_barrier: Optional barrier for synchronization before exit.
    :param poll_interval: Time between iteration cycles and event checks.
    :param args: Positional arguments passed to iter_func.
    :param kwargs: Keyword arguments passed to iter_func.
    :return: Tuple of (exit_reason, last_item) from iterator termination.
    :raises RuntimeError: If error event is detected during iteration.
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


class WorkerProcess(Generic[BackendT, RequestT, RequestTimingsT, ResponseT]):
    """
    Individual worker process for request processing and coordination.

    Manages the complete lifecycle of requests from queue consumption through backend
    processing and updates publication, maintaining synchronization with other
    processes in the group.
    """

    def __init__(
        self,
        local_rank: int,
        local_world_size: int,
        async_limit: int,
        startup_barrier: Barrier,
        shutdown_event: Event,
        error_event: Event,
        requests_queue: Queue[tuple[RequestT, ScheduledRequestInfo[RequestTimingsT]]],
        updates_queue: Queue[
            tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ],
        backend: BackendT,
        request_timings: ScheduledRequestTimings,
        poll_intervals: float,
    ):
        """
        Initialize worker process instance.

        :param local_rank: Process rank within the worker group.
        :param local_world_size: Total number of worker processes in the group.
        :param async_limit: Maximum concurrent requests this worker can handle.
        :param startup_barrier: Multiprocessing barrier for coordinated startup.
        :param shutdown_event: Event for signaling graceful shutdown.
        :param error_event: Event for signaling error conditions across processes.
        :param requests_queue: Queue for receiving requests to process.
        :param updates_queue: Queue for publishing processing updates.
        :param backend: Backend instance for processing requests.
        :param request_timings: Timing strategy for request scheduling.
        :param poll_intervals: Time interval for polling operations.
        """
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
        self.pending_request: Optional[
            tuple[RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ] = None

    def run(self):
        """
        Main entry point for worker process execution.

        Initializes asyncio event loop and starts worker async operations.

        :raises RuntimeError: If worker encounters unrecoverable error during execution.
        """
        try:
            asyncio.run(self.run_async())
        except Exception as exc:  # noqa: BLE001
            self.error_event.set()
            raise RuntimeError(
                f"Worker process {self.local_rank} encountered an error: {exc}"
            ) from exc

    async def run_async(self):
        """
        Execute main asynchronous worker process logic.

        Orchestrates concurrent execution of request processing and shutdown monitoring
        tasks, handling cleanup and error propagation when tasks complete.

        :raises RuntimeError: If worker tasks encounter unrecoverable errors.
        """
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
        """
        Monitor for shutdown and error signals.

        Runs in parallel with request processing to monitor for shutdown or error
        events and trigger appropriate cleanup procedures.

        :raises RuntimeError: If error event is signaled or unexpected exit occurs.
        :raises asyncio.CancelledError: If shutdown event is signaled.
        """
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
        """
        Process incoming requests from the queue.

        Handles backend initialization, process synchronization, concurrent request
        processing with semaphore limiting, and graceful shutdown with task cleanup.

        :raises RuntimeError: If backend initialization or startup synchronization
            fails.
        :raises asyncio.CancelledError: If shutdown is requested during processing.
        :raises NotImplementedError: If multi-turn requests are encountered.
        """
        # Ensure backend is ready on this worker
        await self.backend.process_startup()
        await self.backend.validate()

        # Wait for all processes to be ready before starting
        barrier_exit_reason, _ = await worker_sync_iterable_to_async(
            iter_func="infinite",
            exit_barrier=self.startup_barrier,
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
        ScheduledRequestInfo[RequestTimingsT],
        list[tuple[RequestT, ResponseT]],
    ]:
        while True:
            try:
                request, request_info = self.requests_queue.get_nowait()
                request_info.scheduler_timings.dequeued = time.time()
                request_info.status = "pending"

                if request_info.scheduler_start_time > time.time():
                    # Ensure request_timings logic won't start until scheduler is ready
                    await asyncio.sleep(request_info.scheduler_start_time - time.time())

                # current way of passing through the start_time to request_timings
                timings_offset = self.request_timings.next_offset()
                target_start = request_info.scheduler_start_time + timings_offset
                if target_start > time.time():
                    await asyncio.sleep(target_start - time.time())

                return request, request_info, []
            except queue.Empty:
                await asyncio.sleep(self.poll_intervals)
                continue

    async def _process_request(
        self,
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
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
        except Exception as exc:  # noqa: BLE001
            request_info.status = "error"
            request_info.error = str(exc)

        request_info.scheduler_timings.resolve_end = time.time()
        self.updates_queue.put((last_response, request, request_info))
        self.request_timings.request_completed(request_info)
        if cancelled:
            raise asyncio.CancelledError

    async def _cancel_queued_requests(self):
        async def _request_gen() -> AsyncIterator[
            tuple[RequestT, ScheduledRequestInfo[RequestTimingsT]]
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
