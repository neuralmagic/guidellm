from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
)

from pydantic import Field

from guidellm.backend import Backend
from guidellm.objects import StandardBaseModel

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
BackendT = TypeVar("BackendT", bound="Backend")


class RequestSchedulerTimings(StandardBaseModel):
    targeted_start: Optional[float] = None
    queued: Optional[float] = None
    dequeued: Optional[float] = None
    resolve_start: Optional[float] = None
    resolve_end: Optional[float] = None
    finalized: Optional[float] = None


class RequestTimings(StandardBaseModel):
    request_start: Optional[float] = None
    request_end: Optional[float] = None


RequestTimingsT = TypeVar("RequestTimingsT", bound=RequestTimings)


class ScheduledRequestInfo(StandardBaseModel, Generic[RequestTimingsT]):
    request_id: str
    status: Literal[
        "queued", "pending", "in_progress", "completed", "errored", "cancelled"
    ]
    error: Optional[str] = None
    scheduler_node_id: int
    scheduler_process_id: int
    scheduler_start_time: float

    scheduler_timings: RequestSchedulerTimings = Field(
        default_factory=RequestSchedulerTimings
    )
    request_timings: Optional[RequestTimingsT] = None

    @property
    def started_at(self) -> Optional[float]:
        request_start = (
            self.request_timings.request_start if self.request_timings else None
        )

        return request_start or self.scheduler_timings.resolve_start

    @property
    def completed_at(self) -> Optional[float]:
        request_end = self.request_timings.request_end if self.request_timings else None

        return request_end or self.scheduler_timings.resolve_end


class SchedulerState(StandardBaseModel):
    node_id: int
    num_processes: int
    start_time: float
    end_queuing_time: Optional[float] = None
    end_queuing_constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    end_processing_time: Optional[float] = None
    end_processing_constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)

    created_requests: int = 0
    queued_requests: int = 0
    pending_requests: int = 0
    processing_requests: int = 0
    processed_requests: int = 0
    successful_requests: int = 0
    errored_requests: int = 0
    cancelled_requests: int = 0


class SchedulerUpdateAction(StandardBaseModel):
    request_queuing: Literal["continue", "stop"] = "continue"
    request_processing: Literal["continue", "stop_local", "stop_all"] = "continue"
    metadata: dict[str, Any] = Field(default_factory=dict)
