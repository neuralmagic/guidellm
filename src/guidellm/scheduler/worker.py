"""
Worker process management for multi-process request scheduling and execution.

Provides infrastructure for managing individual worker processes that handle
request scheduling, processing, and coordination in multi-process environments.

Classes:
    WorkerProcess: Individual worker process for request processing and coordination.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterable
from multiprocessing import Queue
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from queue import Empty as QueueEmpty
from typing import Generic, Optional

import culsans
import msgpack

from guidellm.scheduler.objects import (
    BackendT,
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
)
from guidellm.scheduler.strategy import ScheduledRequestTimings
from guidellm.utils import synchronous_to_exitable_async

__all__ = ["WorkerProcess"]


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
        startup_barrier: ProcessingBarrier,
        shutdown_event: ProcessingEvent,
        error_event: ProcessingEvent,
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
        # worker info
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.async_limit = async_limit

        # process synchronization
        self.startup_barrier = startup_barrier
        self.shutdown_event = shutdown_event
        self.error_event = error_event
        self.requests_queue = requests_queue
        self.updates_queue = updates_queue

        # local synchronization
        self.pending_requests_queue: culsans.Queue[
            tuple[RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ] = None
        self.pending_updates_queue: culsans.Queue[
            tuple[RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ] = None

        # request processing
        self.backend = backend
        self.request_timings = request_timings
        self.poll_intervals = poll_intervals

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

        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

        for task in completed_tasks:
            if task.cancelled():
                continue
            exception = task.exception()
            if exception:
                raise exception

    async def run_async_stop_processing(self):
        """
        Monitor for shutdown and error signals.

        Runs in parallel with request processing to monitor for shutdown or error
        events and trigger appropriate cleanup procedures.

        :raises RuntimeError: If error event is signaled or unexpected exit occurs.
        :raises asyncio.CancelledError: If shutdown event is signaled.
        """
        exit_reason, _ = await synchronous_to_exitable_async(
            synchronous=None,
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

        # Setup local queues
        self.pending_requests_queue = culsans.Queue(maxsize=2)
        self.pending_updates_queue = culsans.Queue()

        # Start background tasks for queue management
        pull_requests_task = asyncio.create_task(
            synchronous_to_exitable_async(
                self._run_pull_requests,
                poll_interval=self.poll_intervals,
            )
        )
        push_updates_task = asyncio.create_task(
            synchronous_to_exitable_async(
                self._run_push_updates, poll_interval=self.poll_intervals
            )
        )

        # Wait for all processes to be ready before starting
        barrier_exit_reason, _ = await synchronous_to_exitable_async(
            synchronous=None,
            exit_barrier=self.startup_barrier,
            poll_interval=self.poll_intervals,
        )
        if barrier_exit_reason not in ["barrier", "canceled"]:
            raise RuntimeError(
                f"Worker process {self.local_rank} failed to synchronize at startup: "
                f"{barrier_exit_reason}"
            )

        async_semaphore = asyncio.Semaphore(self.async_limit)
        pending_tasks = set()

        def _task_done(task):
            pending_tasks.discard(task)
            async_semaphore.release()
            exception = task.exception()
            if exception and not isinstance(exception, asyncio.CancelledError):
                raise exception

        try:
            while True:
                await async_semaphore.acquire()
                request_task = asyncio.create_task(self._process_next_request())
                pending_tasks.add(request_task)
                request_task.add_done_callback(_task_done)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                if pending_tasks:
                    for task in list(pending_tasks):
                        task.cancel()
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                await self._cancel_queued_requests()
                pull_requests_task.cancel()
                push_updates_task.cancel()
                await asyncio.gather(
                    pull_requests_task, push_updates_task, return_exceptions=True
                )
                await self.backend.process_shutdown()

            raise

    def _run_pull_requests(self) -> Iterable:
        while True:
            try:
                request_bytes = self.requests_queue.get(timeout=self.poll_intervals)
                request_tuple = msgpack.unpackb(request_bytes, raw=False)
                self.pending_requests_queue.sync_put(request_tuple)
            except QueueEmpty:
                # No request available, continue polling
                pass
            except culsans.QueueShutDown:
                break

            yield None

    def _run_push_updates(self) -> Iterable:
        """Push updates from local queue to multiprocessing queue."""
        while True:
            try:
                update = self.pending_updates_queue.sync_get()
                update_bytes = msgpack.packb(update, use_bin_type=True)
                self.updates_queue.put(update_bytes)
            except culsans.QueueShutDown:
                break

            yield None

    async def _process_next_request(self):
        request: Optional[RequestT] = None
        request_info: Optional[ScheduledRequestInfo[RequestTimingsT]] = None
        response: Optional[ResponseT] = None
        canceled = False

        try:
            # get next request to send
            request, request_info = await self.pending_requests_queue.async_get()
            current_time = time.time()
            request_info.scheduler_timings.dequeued = current_time
            request_info.status = "pending"
            response: ResponseT | None = None

            # get time to send at and wait until that time
            timings_offset = self.request_timings.next_offset()
            target_start = request_info.scheduler_start_time + timings_offset
            if target_start > current_time:
                sleep_duration = target_start - current_time
                await asyncio.sleep(sleep_duration)
                request_info.scheduler_timings.scheduled_at = target_start
            else:
                request_info.scheduler_timings.scheduled_at = current_time

            # process the request
            request_info.status = "in_progress"
            request_info.scheduler_timings.resolve_start = time.time()
            await self.pending_updates_queue.async_put(
                (response, request, request_info.model_copy())
            )
            async for resp in self.backend.resolve(request, request_info, None):
                response = resp

            request_info.status = "completed"
        except asyncio.CancelledError:
            if request_info is not None:
                request_info.status = "cancelled"
            canceled = True
        except Exception as exc:  # noqa: BLE001
            if request_info is not None:
                request_info.status = "errored"
                request_info.error = str(exc)

        if request is not None and request_info is not None:
            # Ensure there is a request in case cancel happened while awaiting next
            request_info.scheduler_timings.resolve_end = time.time()
            await self.pending_updates_queue.async_put(
                (response, request, request_info)
            )
            self.request_timings.request_completed(request_info)

        if canceled:
            # Safe to propagate cancel now, we've fully handled request
            raise asyncio.CancelledError

    async def _cancel_queued_requests(self):
        """Cancel all queued requests that haven't been processed yet."""
        while True:
            try:
                await asyncio.sleep(0)  # Yield control to buffer any pending requests
                next_tuple = self.pending_requests_queue.get_nowait()
                request: RequestT = next_tuple[0]
                request_info: ScheduledRequestInfo[RequestTimingsT] = next_tuple[1]

                # Update in progress first; ensure the same updates for all requests
                request_info.status = "in_progress"
                request_info.scheduler_timings.resolve_start = time.time()
                await self.pending_updates_queue.async_put(
                    (None, request, request_info.model_copy())
                )

                request_info.status = "cancelled"
                request_info.scheduler_timings.resolve_end = time.time()
                await self.pending_updates_queue.async_put(
                    (None, request, request_info)
                )
            except culsans.QueueEmpty:
                break
