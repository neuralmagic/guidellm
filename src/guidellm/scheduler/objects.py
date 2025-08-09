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

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
)

from pydantic import Field

from guidellm.objects import StandardBaseModel

__all__ = [
    "BackendInterface",
    "BackendT",
    "RequestSchedulerTimings",
    "RequestT",
    "RequestTimings",
    "RequestTimingsT",
    "ResponseT",
    "ScheduledRequestInfo",
    "SchedulerState",
    "SchedulerUpdateAction",
]

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class RequestSchedulerTimings(StandardBaseModel):
    """Scheduler-level timing measurements for request lifecycle tracking."""

    targeted_start: Optional[float] = Field(
        default=None,
        description="When the request was initially targeted for execution",
    )
    queued: Optional[float] = Field(
        default=None,
        description="When the request was placed into the processing queue",
    )
    dequeued: Optional[float] = Field(
        default=None,
        description="When the request was removed from the queue for processing",
    )
    resolve_start: Optional[float] = Field(
        default=None, description="When backend resolution of the request began"
    )
    resolve_end: Optional[float] = Field(
        default=None, description="When backend resolution of the request completed"
    )
    finalized: Optional[float] = Field(
        default=None,
        description="When the request was processed/acknowledged by the scheduler",
    )


class RequestTimings(StandardBaseModel):
    """Base timing measurements for backend request processing."""

    request_start: Optional[float] = Field(
        default=None, description="When the backend began processing the request"
    )
    request_end: Optional[float] = Field(
        default=None, description="When the backend completed processing the request"
    )


RequestTimingsT = TypeVar("RequestTimingsT", bound=RequestTimings)


class ScheduledRequestInfo(StandardBaseModel, Generic[RequestTimingsT]):
    """Complete request information including status, timings, and metadata."""

    request_id: str = Field(description="Unique identifier for the request")
    status: Literal[
        "queued", "pending", "in_progress", "completed", "errored", "cancelled"
    ] = Field(description="Current processing status of the request")
    error: Optional[str] = Field(
        default=None, description="Error message if the request.status is 'errored'"
    )
    scheduler_node_id: int = Field(
        description="ID/rank of the scheduler node handling the request"
    )
    scheduler_process_id: int = Field(
        description="ID/rank of the node's scheduler process handling the request"
    )
    scheduler_start_time: float = Field(
        description="Unix timestamp for the local time when scheduler processing began"
    )

    scheduler_timings: RequestSchedulerTimings = Field(
        default_factory=RequestSchedulerTimings,
        description="Scheduler-level timing measurements for request lifecycle",
    )
    request_timings: Optional[RequestTimingsT] = Field(
        default=None,
        description="Backend-specific timing measurements for request processing",
    )

    @property
    def started_at(self) -> Optional[float]:
        """
        Get the effective request processing start time.

        :return: Unix timestamp when processing began, or None if not started.
        """
        request_start = (
            self.request_timings.request_start if self.request_timings else None
        )

        return request_start or self.scheduler_timings.resolve_start

    @property
    def completed_at(self) -> Optional[float]:
        """
        Get the effective request processing completion time.

        :return: Unix timestamp when processing completed, or None if not completed.
        """
        request_end = self.request_timings.request_end if self.request_timings else None

        return request_end or self.scheduler_timings.resolve_end


class BackendInterface(ABC, Generic[RequestT, RequestTimingsT, ResponseT]):
    """
    Abstract interface for request processing backends. Note: before process_startup
    is invoked, the implementation must ensure all properties are pickleable.
    """

    @property
    @abstractmethod
    def processes_limit(self) -> Optional[int]:
        """Maximum worker processes supported, or None if unlimited."""

    @property
    @abstractmethod
    def requests_limit(self) -> Optional[int]:
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
        request_info: ScheduledRequestInfo[RequestTimingsT],
        history: Optional[list[tuple[RequestT, ResponseT]]] = None,
    ) -> AsyncIterator[tuple[ResponseT, ScheduledRequestInfo[RequestTimingsT]]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process.
        :param request_info: Scheduling metadata and timing information.
        :param history: Optional conversation history for multi-turn requests.
        :yield: Tuples of (response, updated_request_info) for each response chunk.
        :raises: Implementation-specific exceptions for processing failures.
        """


BackendT = TypeVar("BackendT", bound="BackendInterface")


class SchedulerState(StandardBaseModel):
    """Scheduler operation state tracking and statistics."""

    node_id: int = Field(description="Unique identifier for this scheduler node")
    num_processes: int = Field(
        description="Number of worker processes in this scheduler"
    )
    start_time: float = Field(description="Unix timestamp when the scheduler started")
    end_time: Optional[float] = Field(
        default=None, description="Unix timestamp when the scheduler stopped"
    )
    end_queuing_time: Optional[float] = Field(
        default=None, description="When request queuing stopped, if applicable"
    )
    end_queuing_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Constraints that triggered queuing termination",
    )
    end_processing_time: Optional[float] = Field(
        default=None, description="When request processing stopped, if applicable"
    )
    end_processing_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Constraints that triggered processing termination",
    )
    scheduler_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="The latest state from all constraints applied during the scheduler run",
    )

    remaining_fraction: Optional[float] = Field(
        default=None,
        description="Estimated fraction for the remaining progress of the scheduler run, if known",
    )
    remaining_requests: Optional[int] = Field(
        default=None,
        description="Estimated number of requests remaining to be processed, if known",
    )
    remaining_duration: Optional[float] = Field(
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
