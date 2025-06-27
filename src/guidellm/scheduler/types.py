from dataclasses import dataclass
from multiprocessing import Queue
from typing import Generic, Literal, Optional, TypeVar

from guidellm.request.session import RequestSession
from guidellm.scheduler.result import SchedulerRequestInfo

__all__ = [
    "MPQueues",
    "RequestT",
    "ResponseT",
    "WorkerProcessRequestTime",
    "WorkerProcessResult",
]


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


# TODO: Move dataclasses somewhere else


@dataclass
class WorkerProcessRequestTime:
    start_time: float
    timeout_time: float
    queued_time: float


@dataclass
class WorkerProcessResult(Generic[RequestT, ResponseT]):
    type_: Literal["request_scheduled", "request_start", "request_complete"]
    request: RequestT
    response: Optional[ResponseT]
    info: SchedulerRequestInfo


@dataclass
class MPQueues(Generic[RequestT, ResponseT]):
    requests: Queue[RequestSession[RequestT, ResponseT]]
    times: Queue[WorkerProcessRequestTime]
    responses: Queue[WorkerProcessResult[RequestT, ResponseT]]
