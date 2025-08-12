"""
Worker process management for multi-process request scheduling and execution.

Provides infrastructure for managing individual worker processes that handle
request scheduling, processing, and coordination in multi-process environments.

Classes:
    WorkerProcess: Individual worker process for request processing and coordination.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Generator
from multiprocessing import Queue
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from queue import Empty as QueueEmpty
from threading import Event as ThreadingEvent
from typing import Generic, Optional

import culsans

from guidellm.scheduler.objects import (
    BackendT,
    MeasuredRequestTimingsT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
)
from guidellm.scheduler.strategy import ScheduledRequestTimings
from guidellm.utils import MsgpackEncoding, synchronous_to_exitable_async

__all__ = ["WorkerProcess"]


class WorkerProcess(Generic[BackendT, RequestT, MeasuredRequestTimingsT, ResponseT]):
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
        requests_queue: Queue[
            tuple[RequestT, ScheduledRequestInfo[MeasuredRequestTimingsT]]
        ],
        updates_queue: Queue[
            tuple[
                ResponseT | None,
                RequestT,
                ScheduledRequestInfo[MeasuredRequestTimingsT],
            ]
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
        # Worker info
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.async_limit = async_limit

        # Process synchronization
        self.startup_barrier = startup_barrier
        self.shutdown_event = shutdown_event
        self.error_event = error_event
        self.requests_queue = requests_queue
        self.updates_queue = updates_queue

        # Local synchronization (initialized during start up)
        self.pending_requests_queue: culsans.Queue[
            tuple[RequestT, ScheduledRequestInfo[MeasuredRequestTimingsT]]
        ] = None
        self.pending_updates_queue: culsans.Queue[
            tuple[RequestT, ScheduledRequestInfo[MeasuredRequestTimingsT]]
        ] = None
        self.requests_canceled: ThreadingEvent = None
        self.pull_task: asyncio.Task = None
        self.push_task: asyncio.Task = None

        # Request processing
        self.backend = backend
        self.request_timings = request_timings
        self.poll_intervals = poll_intervals
        self.startup_completed: bool = False

    def run(self):
        """
        Main entry point for worker process execution.

        Initializes asyncio event loop and starts worker async operations.

        :raises RuntimeError: If worker encounters unrecoverable error during execution.
        """
        try:
            asyncio.run(self.run_async())
        except Exception as exc:
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
        # Start both shutdown monitoring and request processing concurrently
        tasks = [
            asyncio.create_task(self.run_async_stop_processing()),
            asyncio.create_task(self.run_async_requests_processing()),
        ]

        try:
            # Wait for the first task to complete (shut down or error)
            completed, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            # Check for exceptions in completed tasks
            for task in completed:
                if not task.cancelled() and (exception := task.exception()):
                    raise exception
        except asyncio.CancelledError:
            # Ensure all tasks are canceled before re-raising
            for task in tasks:
                if not task.done():
                    task.cancel()
            if any(not task.done() for task in tasks):
                await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
        try:
            await self._initialize_requests_processing()
            await self._start_ready_requests_processing()
            await self._loop_requests_processing()
        except asyncio.CancelledError:
            await self._shutdown_requests_processing()

            raise

    async def _initialize_requests_processing(self):
        # Ensure backend is ready on this worker
        await self.backend.process_startup()
        await self.backend.validate()

        # Setup local queues
        self.pending_requests_queue = culsans.Queue(maxsize=2)
        self.pending_updates_queue = culsans.Queue()
        self.requests_canceled = ThreadingEvent()

        # Start background tasks for queue management
        self.pull_task = asyncio.create_task(
            synchronous_to_exitable_async(
                self._pull_requests_generator(),
                poll_interval=self.poll_intervals,
            )
        )
        self.push_task = asyncio.create_task(
            synchronous_to_exitable_async(
                self._push_updates_generator(), poll_interval=self.poll_intervals
            )
        )

    async def _start_ready_requests_processing(self):
        # Wait for all processes to be ready
        barrier_exit_reason, _ = await synchronous_to_exitable_async(
            synchronous=None,
            exit_barrier=self.startup_barrier,
            poll_interval=self.poll_intervals,
        )

        if barrier_exit_reason not in ["barrier", "canceled"]:
            raise RuntimeError(
                f"Worker process {self.local_rank} failed to synchronize at "
                f"startup: {barrier_exit_reason}"
            )

        self.startup_completed = True

    async def _loop_requests_processing(self):
        async_semaphore = asyncio.Semaphore(self.async_limit)
        pending_tasks = set()

        def _task_done(task):
            pending_tasks.discard(task)
            async_semaphore.release()

            if not task.cancelled() and (exception := task.exception()):
                raise exception

        try:
            # Main loop; loop until canceled
            while True:
                await async_semaphore.acquire()
                request_task = asyncio.create_task(self._process_next_request())
                pending_tasks.add(request_task)
                request_task.add_done_callback(_task_done)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            # Shut down requests queuing
            self.requests_canceled.set()

            # Cancel pending requests
            if pending_tasks:
                for task in list(pending_tasks):
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            raise

    async def _shutdown_requests_processing(self):
        if self.requests_canceled is not None:
            # Queues have been constructed, cancel pending and ensure updates
            self.requests_canceled.set()
            await self._cancel_pending_requests()
            await self.pending_updates_queue.async_join()
            await self.pending_requests_queue.aclose()
            await self.pending_updates_queue.aclose()

        # Cancel background tasks
        tasks = []
        if self.push_task is not None and not self.push_task.done():
            self.push_task.cancel()
            tasks.append(self.push_task)
        if self.pull_task is not None and not self.pull_task.done():
            self.pull_task.cancel()
            tasks.append(self.pull_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Shut down backend
        await self.backend.process_shutdown()

        # Reset state
        self.pending_requests_queue = None
        self.pending_updates_queue = None
        self.pull_task = None
        self.push_task = None
        self.requests_canceled = None

    async def _process_next_request(self):
        request: Optional[RequestT] = None
        request_info: Optional[ScheduledRequestInfo[MeasuredRequestTimingsT]] = None
        response: Optional[ResponseT] = None

        try:
            # get next request to send
            request, request_info = await self.pending_requests_queue.async_get()
            current_time = time.time()
            request_info.scheduler_timings.dequeued = current_time
            request_info.status = "pending"
            response = None

            # Calculate when to start processing request
            timings_offset = self.request_timings.next_offset()
            target_start = request_info.scheduler_start_time + timings_offset
            request_info.scheduler_timings.targeted_start = target_start

            if target_start > current_time:
                await asyncio.sleep(target_start - current_time)
                request_info.scheduler_timings.scheduled_at = target_start
            else:
                request_info.scheduler_timings.scheduled_at = current_time

            # Send start processing update
            request_info.status = "in_progress"
            request_info.scheduler_timings.resolve_start = time.time()
            await self.pending_updates_queue.async_put(
                (response, request, request_info.model_copy())
            )

            # Process the request
            async for resp in self.backend.resolve(request, request_info, None):
                response = resp

            # Send completion update
            request_info.status = "completed"
            request_info.scheduler_timings.resolve_end = time.time()
            await self.pending_updates_queue.async_put(
                (response, request, request_info)
            )

            # Notify instance states
            self.request_timings.request_completed(request_info)
            self.pending_requests_queue.task_done()
        except asyncio.CancelledError:
            # Handle cancellation
            if request is not None and request_info is not None:
                await self._handle_request_cancellation(response, request, request_info)
            raise
        except Exception as exc:  # noqa: BLE001
            if request is not None and request_info is not None:
                await self._handle_request_error(response, request, request_info, exc)

    async def _handle_request_cancellation(
        self,
        response: Optional[ResponseT],
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
    ):
        request_info.status = "cancelled"
        request_info.scheduler_timings.resolve_end = time.time()
        await self.pending_updates_queue.async_put((response, request, request_info))

        # Notify instance states
        self.request_timings.request_completed(request_info)
        self.pending_requests_queue.task_done()

    async def _handle_request_error(
        self,
        response: Optional[ResponseT],
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        exc: Exception,
    ):
        request_info.status = "errored"
        request_info.error = str(exc)
        request_info.scheduler_timings.resolve_end = time.time()
        await self.pending_updates_queue.async_put((response, request, request_info))

        # Notify instance states
        self.request_timings.request_completed(request_info)
        self.pending_requests_queue.task_done()

    async def _cancel_pending_requests(self):
        while True:
            try:
                request, request_info = await asyncio.wait_for(
                    self.pending_requests_queue.async_get(), timeout=self.poll_intervals
                )

                # Send in_progress update first, every request has same update sequence
                request_info.status = "in_progress"
                request_info.scheduler_timings.resolve_start = time.time()
                await self.pending_updates_queue.async_put(
                    (None, request, request_info.model_copy())
                )

                await self._handle_request_cancellation(None, request, request_info)

            except (culsans.QueueEmpty, asyncio.TimeoutError):
                break

    def _pull_requests_generator(self) -> Generator:
        while True:
            if self.requests_canceled.is_set():
                break

            try:
                message = self.requests_queue.get(timeout=self.poll_intervals)
                request_tuple = MsgpackEncoding.decode(message)
                self.pending_requests_queue.sync_put(request_tuple)
            except QueueEmpty:
                pass  # No update available, continue polling
            except culsans.QueueShutDown:
                break
            except Exception as exc:  # noqa: BLE001
                print(exc)

            yield None

    def _push_updates_generator(self) -> Generator:
        while True:
            try:
                update_tuple = self.pending_updates_queue.sync_get(
                    timeout=self.poll_intervals
                )
                message = MsgpackEncoding.encode(update_tuple)
                self.updates_queue.put(message)
                self.pending_updates_queue.task_done()
            except culsans.QueueEmpty:
                pass  # No update available, continue polling
            except culsans.QueueShutDown:
                break
            except Exception as exc:  # noqa: BLE001
                print(exc)

            yield None
