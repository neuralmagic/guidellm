"""
Core data structures and interfaces for the GuideLLM scheduler system.

Provides type-safe abstractions for distributed request processing, timing
measurements, and backend interfaces for benchmarking operations.

Classes:
    RequestSchedulerTimings: Scheduler-level request timing measurements.
    RequestTimings: Base backend request timing measurements.
    ScheduledRequestInfo: Complete request lifecycle information.
    BackendInterface: Abstract backend processing interface.
    SchedulerState: Scheduler operation state tracking.
    SchedulerUpdateAction: Scheduler behavior control directives.

Type Variables:
    RequestT: Generic request object type.
    ResponseT: Generic response object type.
    RequestTimingsT: Generic request timing object type.
    BackendT: Generic backend interface type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
)

from pydantic import Field, computed_field
from typing_extensions import TypeAliasType, TypedDict

from guidellm.objects import StandardBaseModel

__all__ = [
    "BackendInterface",
    "BackendT",
    "MeasuredRequestTimings",
    "MeasuredRequestTimingsT",
    "MultiTurnRequestT",
    "RequestSchedulerTimings",
    "RequestT",
    "ResponseT",
    "ScheduledRequestInfo",
    "SchedulerState",
    "SchedulerUpdateAction",
    "SchedulerUpdateActionProgress",
]

RequestT = TypeVar("RequestT")
MultiTurnRequestT = TypeAliasType(
    "MultiTurnRequestT",
    Iterable[Union[RequestT, tuple[RequestT, float]]],
    type_params=(RequestT,),
)
ResponseT = TypeVar("ResponseT")


class RequestSchedulerTimings(StandardBaseModel):
    """Scheduler-level timing measurements for request lifecycle tracking."""

    targeted_start: float | None = Field(
        default=None,
        description="When the request was initially targeted for execution",
    )
    queued: float | None = Field(
        default=None,
        description="When the request was placed into the processing queue",
    )
    dequeued: float | None = Field(
        default=None,
        description="When the request was removed from the queue for processing",
    )
    scheduled_at: float | None = Field(
        default=None, description="When the request was scheduled for processing"
    )
    resolve_start: float | None = Field(
        default=None, description="When backend resolution of the request began"
    )
    resolve_end: float | None = Field(
        default=None, description="When backend resolution of the request completed"
    )
    finalized: float | None = Field(
        default=None,
        description="When the request was processed/acknowledged by the scheduler",
    )


class MeasuredRequestTimings(StandardBaseModel):
    """Base timing measurements for backend request processing."""

    request_start: float | None = Field(
        default=None, description="When the backend began processing the request"
    )
    request_end: float | None = Field(
        default=None, description="When the backend completed processing the request"
    )


MeasuredRequestTimingsT = TypeVar(
    "MeasuredRequestTimingsT", bound=MeasuredRequestTimings
)


class ScheduledRequestInfo(StandardBaseModel, Generic[MeasuredRequestTimingsT]):
    """Complete request information including status, timings, and metadata."""

    request_id: str = Field(description="Unique identifier for the request")
    status: Literal[
        "queued", "pending", "in_progress", "completed", "errored", "cancelled"
    ] = Field(description="Current processing status of the request")
    scheduler_node_id: int = Field(
        description="ID/rank of the scheduler node handling the request"
    )
    scheduler_process_id: int = Field(
        description="ID/rank of the node's scheduler process handling the request"
    )
    scheduler_start_time: float = Field(
        description="Unix timestamp for the local time when scheduler processing began"
    )

    error: str | None = Field(
        default=None, description="Error message if the request.status is 'errored'"
    )
    scheduler_timings: RequestSchedulerTimings = Field(
        default_factory=RequestSchedulerTimings,
        description="Scheduler-level timing measurements for request lifecycle",
    )
    request_timings: MeasuredRequestTimingsT | None = Field(
        default=None,
        description="Backend-specific timing measurements for request processing",
    )

    @computed_field
    @property
    def started_at(self) -> float | None:
        """
        Get the effective request processing start time.

        :return: Unix timestamp when processing began, or None if not started.
        """
        request_start = (
            self.request_timings.request_start if self.request_timings else None
        )

        return request_start or self.scheduler_timings.resolve_start

    @computed_field
    @property
    def completed_at(self) -> float | None:
        """
        Get the effective request processing completion time.

        :return: Unix timestamp when processing completed, or None if not completed.
        """
        request_end = self.request_timings.request_end if self.request_timings else None

        return request_end or self.scheduler_timings.resolve_end


class BackendInterface(ABC, Generic[RequestT, MeasuredRequestTimingsT, ResponseT]):
    """
    Abstract interface for request processing backends. Note: before process_startup
    is invoked, the implementation must ensure all properties are pickleable.
    """

    @property
    @abstractmethod
    def processes_limit(self) -> int | None:
        """Maximum worker processes supported, or None if unlimited."""

    @property
    @abstractmethod
    def requests_limit(self) -> int | None:
        """Maximum concurrent requests supported, or None if unlimited."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """
        :return: Backend metadata including model any initializaiton and
            configuration information.
        """
        ...

    @abstractmethod
    async def process_startup(self) -> None:
        """
        Perform backend initialization and startup procedures.

        :raises: Implementation-specific exceptions for startup failures.
        """

    @abstractmethod
    async def validate(self) -> None:
        """
        Validate backend configuration and operational status.

        :raises: Implementation-specific exceptions for validation failures.
        """

    @abstractmethod
    async def process_shutdown(self) -> None:
        """
        Perform backend cleanup and shutdown procedures.

        :raises: Implementation-specific exceptions for shutdown failures.
        """

    @abstractmethod
    async def resolve(
        self,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        history: list[tuple[RequestT, ResponseT]] | None = None,
    ) -> AsyncIterator[tuple[ResponseT, ScheduledRequestInfo[MeasuredRequestTimingsT]]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process.
        :param request_info: Scheduling metadata and timing information.
        :param history: Optional conversation history for multi-turn requests.
        :yield: Tuples of (response, updated_request_info) for each response chunk.
        :raises: Implementation-specific exceptions for processing failures.
        """


BackendT = TypeVar("BackendT", bound=BackendInterface)


class SchedulerUpdateActionProgress(TypedDict, total=False):
    """Progress information for a scheduler update action."""

    remaining_fraction: float | None = None
    remaining_requests: float | None = None
    remaining_duration: float | None = None


class SchedulerUpdateAction(StandardBaseModel):
    """Scheduler behavior control directives and actions."""

    request_queuing: Literal["continue", "stop"] = Field(
        default="continue", description="Action to take for request queuing operations"
    )
    request_processing: Literal["continue", "stop_local", "stop_all"] = Field(
        default="continue",
        description="Action to take for request processing operations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context and data for the scheduler action",
    )
    progress: SchedulerUpdateActionProgress = Field(
        default_factory=SchedulerUpdateActionProgress,
        description="Progress information for the scheduler action",
    )


class SchedulerState(StandardBaseModel):
    """Scheduler operation state tracking and statistics."""

    node_id: int = Field(description="Unique identifier for this scheduler node")
    num_processes: int = Field(
        description="Number of worker processes in this scheduler"
    )
    start_time: float = Field(description="Unix timestamp when the scheduler started")
    end_time: float | None = Field(
        default=None, description="Unix timestamp when the scheduler stopped"
    )
    end_queuing_time: float | None = Field(
        default=None, description="When request queuing stopped, if applicable"
    )
    end_queuing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered queuing termination",
    )
    end_processing_time: float | None = Field(
        default=None, description="When request processing stopped, if applicable"
    )
    end_processing_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="Constraints that triggered processing termination",
    )
    scheduler_constraints: dict[str, SchedulerUpdateAction] = Field(
        default_factory=dict,
        description="The latest state from all constraints applied during the scheduler run",
    )

    remaining_fraction: float | None = Field(
        default=None,
        description="Estimated fraction for the remaining progress of the scheduler run, if known",
    )
    remaining_requests: int | None = Field(
        default=None,
        description="Estimated number of requests remaining to be processed, if known",
    )
    remaining_duration: float | None = Field(
        default=None,
        description="Estimated time remaining in seconds for the scheduler run, if known",
    )

    created_requests: int = Field(
        default=0, description="Total number of requests created"
    )
    queued_requests: int = Field(
        default=0, description="Total number of requests queued for processing"
    )
    pending_requests: int = Field(
        default=0, description="Number of requests currently pending processing"
    )
    processing_requests: int = Field(
        default=0, description="Number of requests currently being processed"
    )
    processed_requests: int = Field(
        default=0, description="Total number of requests that completed processing"
    )
    successful_requests: int = Field(
        default=0, description="Number of requests that completed successfully"
    )
    errored_requests: int = Field(
        default=0, description="Number of requests that failed with errors"
    )
    cancelled_requests: int = Field(
        default=0, description="Number of requests that were cancelled"
    )
