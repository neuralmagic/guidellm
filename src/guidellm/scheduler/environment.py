"""
Scheduler environment abstractions for distributed and non-distributed execution.

Provides environment abstractions for coordinating scheduler execution across
single or multiple nodes, handling synchronization, error propagation, and lifecycle
management.

Classes:
    Environment: Abstract base for scheduler coordination across nodes.
    NonDistributedEnvironment: Single-node implementation with minimal overhead.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import (
    Generic,
)

from guidellm.config import settings
from guidellm.scheduler.constraints import Constraint
from guidellm.scheduler.objects import (
    MeasuredRequestTimingsT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy

__all__ = ["Environment", "NonDistributedEnvironment"]


class Environment(ABC, Generic[RequestT, ResponseT]):
    """
    Abstract base for scheduler execution environments.

    Defines the interface for coordinating scheduler execution across single or
    multiple nodes, handling parameter synchronization, timing, state updates,
    error propagation, and cleanup.
    """

    @abstractmethod
    async def sync_run_params(
        self,
        requests: Iterable[RequestT],
        strategy: SchedulingStrategy,
        constraints: dict[str, Constraint],
    ) -> tuple[
        Iterable[RequestT],
        SchedulingStrategy,
        dict[str, Constraint],
    ]:
        """
        Synchronize run parameters across nodes and resolve local execution scope.

        Coordinates parameter distribution across active nodes. For distributed
        environments, handles validation, node assignment, and workload partitioning.
        For non-distributed environments, typically returns parameters unchanged.

        :param requests: Complete set of requests to process across all nodes.
        :param strategy: Scheduling strategy to apply during execution.
        :param constraints: Runtime constraints to enforce during execution.
        :return: Tuple of (local_requests, strategy, constraints) for this node.
        :raises Exception: If parameter synchronization fails or nodes inconsistent.
        """
        ...

    @abstractmethod
    async def sync_run_start(self) -> float:
        """
        Coordinate global start time across nodes for synchronized execution.

        Ensures all nodes begin processing simultaneously for accurate benchmarking.

        :return: Unix timestamp when all nodes should begin processing.
        :raises Exception: If startup synchronization fails across nodes.
        """
        ...

    @abstractmethod
    async def update_run_iteration(
        self,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
    ):
        """
        Update environment state with completed request iteration.

        Called after each request is processed to update execution progress.
        Enables state synchronization across nodes in distributed environments.

        :param response: Response generated for the request, if successful.
        :param request: The processed request.
        :param request_info: Metadata about request processing including timings.
        :raises Exception: If state update fails or indicates critical errors.
        """
        ...

    @abstractmethod
    async def sync_run_error(self, err: Exception):
        """
        Handle and propagate errors across all nodes.

        Coordinates error handling when failures occur, ensuring all nodes are
        notified and can perform appropriate cleanup or shutdown.

        :param err: The exception that occurred during execution.
        """
        ...

    @abstractmethod
    async def sync_run_end(
        self,
    ) -> AsyncIterator[
        tuple[
            ResponseT,
            RequestT,
            ScheduledRequestInfo[MeasuredRequestTimingsT],
            SchedulerState,
        ]
    ]:
        """
        Finalize execution and aggregate results from all nodes.

        Handles cleanup, result synchronization, and error propagation at run end.
        Collects results from worker nodes in distributed environments.

        :return: Iterator of (response, request, request_info, state) tuples from
            remote nodes in distributed environments, empty for non-distributed.
        :raises Exception: Any errors that occurred during the run.
        """
        ...


class NonDistributedEnvironment(Environment):
    """
    Single-node scheduler execution environment.

    Simplified environment for running schedulers on a single node without
    distributed coordination. Implements the Environment interface with minimal
    synchronization overhead for local testing, development, and single-machine
    benchmarking.

    :ivar run_err: Exception that occurred during execution, if any.
    """

    def __init__(self):
        """Initialize with no stored errors."""
        self.run_err: Exception = None

    async def sync_run_params(
        self,
        requests: Iterable[RequestT],
        strategy: SchedulingStrategy,
        constraints: dict[str, Constraint],
    ) -> tuple[Iterable[RequestT], SchedulingStrategy, dict[str, Constraint]]:
        """
        Return parameters unchanged for single-node execution.

        :param requests: Iterable of requests to process.
        :param strategy: Scheduling strategy to apply during execution.
        :param constraints: Runtime constraints to enforce during execution.
        :return: Tuple containing the original (requests, strategy, constraints).
        """
        return requests, strategy, constraints

    async def sync_run_start(self) -> float:
        """
        Return current time plus configuration delay.

        :return: Unix timestamp for when the run should start.
        """
        return time.time() + settings.scheduler_start_delay_non_distributed

    async def update_run_iteration(
        self,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
    ):
        """
        No-op for single-node execution.

        :param response: Response generated for the request, if successful.
        :param request: The request that was processed.
        :param request_info: Metadata about request processing including timings.
        """

    async def sync_run_error(self, err: Exception):
        """
        Store error for later propagation during run finalization.

        :param err: The exception that occurred during execution.
        """
        self.run_err = err

    async def sync_run_end(
        self,
    ) -> AsyncIterator[
        tuple[
            ResponseT,
            RequestT,
            ScheduledRequestInfo[MeasuredRequestTimingsT],
            SchedulerState,
        ]
    ]:
        """
        Finalize single-node execution and propagate any stored errors.

        :return: Empty iterator since there are no remote nodes.
        :raises Exception: Any error stored during execution via sync_run_error.
        """
        if self.run_err:
            raise self.run_err
        # Return empty async iterator for non-distributed environment
        return
        yield
