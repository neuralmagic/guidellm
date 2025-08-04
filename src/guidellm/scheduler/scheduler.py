"""
Scheduler for coordinating distributed load testing and benchmarking workloads.

This module provides a thread-safe singleton scheduler for orchestrating
benchmarking operations across worker processes and distributed environments.

Classes:
    Scheduler: Generic singleton scheduler for distributed request processing.
"""

import threading
from collections.abc import AsyncIterator, Iterable
from typing import Any, Generic, Optional, Union

from guidellm.scheduler.constraints import (
    CallableConstraint,
    ConstraintsInitializerFactory,
)
from guidellm.scheduler.environment import Environment
from guidellm.scheduler.objects import (
    BackendT,
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker_group import WorkerProcessGroup
from guidellm.utils.singleton import ThreadSafeSingletonMixin

__all__ = ["Scheduler"]


class Scheduler(
    Generic[BackendT, RequestT, RequestTimingsT, ResponseT], ThreadSafeSingletonMixin
):
    """
    Generic singleton scheduler for coordinating distributed load testing workloads.

    Orchestrates benchmarking operations by managing request distribution across
    worker processes, coordinating timing with distributed environments, and
    aggregating results. Supports generic backend types for adaptability to
    various testing scenarios including LLM inference and API testing.

    Example:
    ::
        from guidellm.scheduler import Scheduler
        from guidellm.backend import (
            OpenAIBackend,
            GenerationRequest,
            GenerationResponse,
            GenerationRequestTimings
        )

        scheduler = Scheduler[
            OpenAIBackend,
            GenerationRequest,
            GenerationRequestTimings,
            GenerationResponse
        ]()
        async for response, request, info, state in scheduler.run(
            requests=request_list,
            backend=backend,
            strategy=strategy,
            env=environment,
            max_requests=1000
        ):
            print(f"Response: {response}")
    """

    def __init__(self):
        """Initialize the scheduler singleton instance."""
        if not self.initialized:
            self.run_lock = threading.Lock()
        super().__init__()

    async def run(
        self,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        env: Environment,
        **constraints: dict[str, Union[Any, dict[str, Any], CallableConstraint]],
    ) -> AsyncIterator[
        tuple[
            Optional[ResponseT],
            RequestT,
            ScheduledRequestInfo[RequestTimingsT],
            SchedulerState,
        ]
    ]:
        """
        Execute request processing with the provided configuration.

        Coordinates execution across worker processes with the specified backend
        and scheduling strategy. Manages timing, synchronization, and resource
        cleanup while yielding real-time updates.

        :param requests: Requests to process. Supports single requests
            (Iterable[RequestT]) or multi-turn sequences (Iterable[Iterable]) where
            each item is either a RequestT or tuple of (RequestT, delay_seconds).
        :param backend: Backend instance for processing requests.
        :param strategy: Scheduling strategy for request timing.
        :param env: Environment for distributed execution coordination.
        :param constraints: Execution control constraints (max_requests, duration,
            etc.). Values can be primitives or callable functions.
        :yields: Tuples of (response, request, scheduling_info, scheduler_state).
            Response may be None for failed requests.
        :raises Exception: Worker process, environment, or constraint evaluation errors
            are propagated after cleanup.
        """
        with self.run_lock:
            worker_group: Optional[
                WorkerProcessGroup[BackendT, RequestT, RequestTimingsT, ResponseT]
            ] = None

            # Any issues during the run will raise an error (local or remote),
            # be caught and passed to the environment,
            # and will ensure clean up before raising the error.
            try:
                # Setup local run parameters, sync with the environment
                constraints = ConstraintsInitializerFactory.resolve_constraints(
                    constraints
                )
                (
                    local_requests,
                    local_strategy,
                    local_constraints,
                ) = await env.sync_run_params(requests, strategy, constraints)

                # Setup the worker group, sync start with the environment
                worker_group = WorkerProcessGroup[
                    BackendT, RequestT, RequestTimingsT, ResponseT
                ](
                    backend=backend,
                    requests=local_requests,
                    strategy=local_strategy,
                    constraints=local_constraints,
                )
                await worker_group.create_processes()
                local_start_time = await env.sync_run_start()
                await worker_group.start(local_start_time)

                # Yield any updates and sync with the environment for non-local updates
                async for (
                    response,
                    request,
                    request_info,
                    state,
                ) in worker_group.request_updates():
                    await env.update_run_iteration(
                        response, request, request_info, state
                    )
                    yield response, request, request_info, state
            except Exception as err:  # noqa: BLE001
                env.sync_run_error(err)
            finally:
                # Ensure all worker processes are cleaned up for error or completion
                if worker_group is not None:
                    err = await worker_group.shutdown()
                    if err is not None:
                        env.sync_run_error(err)

            # Ensure any errors are raised and all responses
            # are yielded for aggregation on the primary node
            async for (
                response,
                request,
                request_info,
                state,
            ) in env.sync_run_end():
                yield response, request, request_info, state
