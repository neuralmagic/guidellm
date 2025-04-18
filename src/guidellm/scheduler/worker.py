import asyncio
import math
import multiprocessing
import multiprocessing.queues
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Union,
)

from loguru import logger
from pydantic import Field

from guidellm.backend import (
    Backend,
    BackendType,
    RequestArgs,
    ResponseSummary,
    StreamingTextResponse,
)
from guidellm.objects import StandardBaseModel
from guidellm.request import GenerationRequest
from guidellm.scheduler.result import SchedulerRequestInfo
from guidellm.scheduler.types import RequestT, ResponseT

__all__ = [
    "WorkerProcessRequest",
    "WorkerProcessResult",
    "ResolveStatus",
    "WorkerDescription",
    "RequestsWorker",
    "GenerativeRequestsWorkerDescription",
    "GenerativeRequestsWorker",
]


@dataclass
class WorkerProcessRequest(Generic[RequestT]):
    request: RequestT
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
class ResolveStatus:
    requested: bool
    completed: bool
    errored: bool
    canceled: bool

    request_start: float
    request_end: float


class WorkerDescription(StandardBaseModel):
    type_: Literal["worker"] = "worker"


class RequestsWorker(ABC, Generic[RequestT, ResponseT]):
    """
    An abstract base class for a worker that processes requests.
    This class defines the interface for a worker that can resolve requests
    asynchronously or synchronously within the Scheduler class.
    Subclasses must implement the `resolve` method,
    which takes a request directly given from the load generator,
    along with the desired start_time for the request and a timeout_time.
    The `resolve` method should return the response from the backend.
    """

    @property
    @abstractmethod
    def description(self) -> WorkerDescription:
        """
        An abstract property that must be implemented by subclasses.
        This property should return a Serializable class representing the information
        about the worker instance.
        """
        ...

    @abstractmethod
    async def prepare_multiprocessing(self):
        """
        An abstract method that must be implemented by subclasses.
        This is useful for workers that have instance state that can not
        be shared across processes and should be cleared out and re-initialized
        for each new process.
        """
        ...

    @abstractmethod
    async def resolve(
        self,
        request: RequestT,
        timeout_time: float,
    ) -> tuple[ResolveStatus, ResponseT]:
        """
        An abstract method that must be implemented by subclasses.
        This method should handle the resolution of a request through asyncio,
        including any necessary backend processing and response handling.

        :param request: The request to be resolved generated by the load generator.
        :param timeout_time: The timeout time for the request, if there is no timeout
            given, then this will be math.inf.
        :return: The response from the worker.
        """
        ...

    async def get_request(
        self, requests_queue: multiprocessing.Queue
    ) -> Optional[WorkerProcessRequest[RequestT]]:
        return await asyncio.to_thread(requests_queue.get)  # type: ignore[attr-defined]

    async def send_result(
        self,
        results_queue: multiprocessing.Queue,
        result: WorkerProcessResult[RequestT, ResponseT],
    ):
        await asyncio.to_thread(results_queue.put, result)  # type: ignore[attr-defined]

    async def resolve_scheduler_request(
        self,
        request: Any,
        queued_time: float,
        dequeued_time: float,
        start_time: float,
        timeout_time: float,
        results_queue: multiprocessing.Queue,
        process_id: int,
    ):
        info = SchedulerRequestInfo(
            targeted_start_time=start_time,
            queued_time=queued_time,
            dequeued_time=dequeued_time,
            scheduled_time=time.time(),
            process_id=process_id,
        )
        result: WorkerProcessResult[RequestT, ResponseT] = WorkerProcessResult(
            type_="request_scheduled",
            request=request,
            response=None,
            info=info,
        )
        asyncio.create_task(self.send_result(results_queue, result))

        if (wait_time := start_time - time.time()) > 0:
            await asyncio.sleep(wait_time)

        info.worker_start = time.time()
        result = WorkerProcessResult(
            type_="request_start",
            request=request,
            response=None,
            info=info,
        )
        asyncio.create_task(self.send_result(results_queue, result))

        status, response = await self.resolve(request, timeout_time)
        info.worker_end = time.time()
        info.requested = status.requested
        info.completed = status.completed
        info.errored = status.errored
        info.canceled = status.canceled
        info.request_start = status.request_start
        info.request_end = status.request_end
        result = WorkerProcessResult(
            type_="request_complete",
            request=request,
            response=response,
            info=info,
        )
        asyncio.create_task(self.send_result(results_queue, result))

    def process_loop_synchronous(
        self,
        requests_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        process_id: int,
    ):
        async def _process_runner():
            while (
                process_request := await self.get_request(requests_queue)
            ) is not None:
                dequeued_time = time.time()

                await self.resolve_scheduler_request(
                    request=process_request.request,
                    queued_time=process_request.queued_time,
                    dequeued_time=dequeued_time,
                    start_time=process_request.start_time,
                    timeout_time=process_request.timeout_time,
                    results_queue=results_queue,
                    process_id=process_id,
                )

        try:
            asyncio.run(_process_runner())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Error in worker process {process_id}: {exc}",
                exc_info=True,
                stack_info=True,
            )

    def process_loop_asynchronous(
        self,
        requests_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        max_concurrency: int,
        process_id: int,
    ):
        async def _process_runner():
            pending = asyncio.Semaphore(max_concurrency)

            if pending.locked():
                raise ValueError(
                    "Async worker called with max_concurrency < 1"
                )

            while (
                process_request := await self.get_request(requests_queue)
            ) is not None:
                dequeued_time = time.time()

                await pending.acquire()

                def _task_done(_: asyncio.Task):
                    nonlocal pending
                    pending.release()

                task = asyncio.create_task(
                    self.resolve_scheduler_request(
                        request=process_request.request,
                        queued_time=process_request.queued_time,
                        dequeued_time=dequeued_time,
                        start_time=process_request.start_time,
                        timeout_time=process_request.timeout_time,
                        results_queue=results_queue,
                        process_id=process_id,
                    )
                )
                task.add_done_callback(_task_done)
                await asyncio.sleep(0)  # enable start task immediately

        try:
            asyncio.run(_process_runner())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Error in worker process {process_id}: {exc}",
                exc_info=True,
                stack_info=True,
            )


class GenerativeRequestsWorkerDescription(WorkerDescription):
    type_: Literal["generative_requests_worker"] = "generative_requests_worker"  # type: ignore[assignment]
    backend_type: BackendType
    backend_target: str
    backend_model: str
    backend_info: dict[str, Any] = Field(
        default_factory=dict,
    )


class GenerativeRequestsWorker(RequestsWorker[GenerationRequest, ResponseSummary]):
    """
    A class that handles the execution of requests using a backend.
    This class is responsible for sending requests to the backend,
    handling responses, and managing errors.

    :param backend: The backend to use for handling requests.
        This should be an instance of Backend such as an OpenAIHTTPBackend.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

    @property
    def description(self) -> GenerativeRequestsWorkerDescription:
        """
        Get the description of the worker.
        :return: The description of the worker.
        """
        return GenerativeRequestsWorkerDescription(
            backend_type=self.backend.type_,
            backend_target=self.backend.target,
            backend_model=self.backend.model or "None",
            backend_info=self.backend.info,
        )

    async def prepare_multiprocessing(self):
        """
        Prepare the worker for multiprocessing.
        This is useful for workers that have instance state that can not
        be shared across processes and should be cleared out and re-initialized
        for each new process.
        """
        await self.backend.prepare_multiprocessing()

    def process_loop_synchronous(
        self,
        requests_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        process_id: int,
    ):
        asyncio.run(self.backend.validate())
        super().process_loop_synchronous(
            requests_queue=requests_queue,
            results_queue=results_queue,
            process_id=process_id,
        )

    def process_loop_asynchronous(
        self,
        requests_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        max_concurrency: int,
        process_id: int,
    ):
        asyncio.run(self.backend.validate())
        super().process_loop_asynchronous(
            requests_queue=requests_queue,
            results_queue=results_queue,
            max_concurrency=max_concurrency,
            process_id=process_id,
        )

    async def resolve(
        self,
        request: GenerationRequest,
        timeout_time: float,
    ) -> tuple[ResolveStatus, ResponseSummary]:
        """
        Resolve a request by sending it to the backend and handling the response.
        This method sends the request to the backend, waits for a response,
        and handles any errors that may occur during the process.

        :param request: The request to resolve.
        :param timeout_time: The time to wait for a response before timing out.
            If timeout_time is math.inf, the request will not timeout.
        :return: A ResponseSummary object containing the response from the backend.
            If an error occurs, the ResponseSummary will contain the error message.
        """
        resolve_start_time = time.time()
        response = None
        error: Optional[str] = None
        status = ResolveStatus(
            requested=False,
            completed=False,
            errored=False,
            canceled=False,
            request_start=-1,
            request_end=-1,
        )

        try:
            if timeout_time < time.time():
                raise asyncio.TimeoutError(
                    "The timeout time has already passed."
                )  # exit early

            status.requested = True
            request_func, request_kwargs = self._create_request_func_kwargs(request)

            async def _runner():
                # wrap function so we can enforce timeout and
                # still return the latest state from the backend
                async for resp in request_func(**request_kwargs):  # type: ignore[operator]
                    nonlocal response
                    response = resp

            await asyncio.wait_for(
                _runner(),
                timeout=timeout_time - time.time() if timeout_time < math.inf else None,
            )

            if not response:
                raise ValueError(
                    f"No response received for request: {request} "
                    f"and backend: {self.backend}"
                )
            if not isinstance(response, ResponseSummary):
                raise ValueError(
                    f"Received no ResponseSummary for request: {request} "
                    f"and backend: {self.backend}, received: {response}"
                )

            status.completed = True
        except asyncio.TimeoutError:
            error = "TimeoutError: The request timed out before completing."
            status.errored = True
            status.canceled = True
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            status.errored = True

        return self._handle_response(
            status=status,
            request=request,
            response=response,
            error=error,
            resolve_start_time=resolve_start_time,
        )

    def _create_request_func_kwargs(
        self,
        request: GenerationRequest,
    ) -> tuple[
        AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None],
        dict[str, Any],
    ]:
        request_func: AsyncGenerator[
            Union[StreamingTextResponse, ResponseSummary], None
        ]
        request_kwargs: dict[str, Any]

        if request.request_type == "text_completions":
            request_func = self.backend.text_completions  # type: ignore[assignment]
            request_kwargs = {
                "prompt": request.content,
                "request_id": request.request_id,
                "prompt_token_count": request.stats.get("prompt_tokens", None),
                "output_token_count": request.constraints.get("output_tokens", None),
                **request.params,
            }
        elif request.request_type == "chat_completions":
            request_func = self.backend.chat_completions  # type: ignore[assignment]
            request_kwargs = {
                "content": request.content,
                "request_id": request.request_id,
                "prompt_token_count": request.stats.get("prompt_tokens", None),
                "output_token_count": request.constraints.get("output_tokens", None),
                **request.params,
            }
        else:
            raise ValueError(
                f"Invalid request type: {request.request_type} for {request}"
            )

        return request_func, request_kwargs

    def _handle_response(
        self,
        status: ResolveStatus,
        request: GenerationRequest,
        response: Any,
        error: Optional[str],
        resolve_start_time: float,
    ) -> tuple[ResolveStatus, ResponseSummary]:
        if response is None or not isinstance(
            response, (ResponseSummary, StreamingTextResponse)
        ):
            # nothing received or invalid response, fill in defaults for error
            if response:
                error = str(
                    ValueError(
                        f"Invalid response: {type(response)} for request: {request}; "
                    )
                ) + (error or "")

            response = ResponseSummary(
                value="",
                request_args=RequestArgs(
                    target=self.backend.target,
                    headers={},
                    payload={},
                ),
                start_time=resolve_start_time,
                end_time=status.request_end,
                first_iter_time=None,
                last_iter_time=None,
                request_id=request.request_id,
                error=error or "Unknown error",
            )
        elif isinstance(response, StreamingTextResponse):
            response = ResponseSummary(
                value=response.value,
                request_args=RequestArgs(
                    target=self.backend.target,
                    headers={},
                    payload={},
                ),
                start_time=response.start_time,
                end_time=time.time(),
                first_iter_time=response.first_iter_time,
                last_iter_time=response.time if response.iter_count > 0 else None,
                request_prompt_tokens=request.stats.get("prompt_tokens", None),
                request_output_tokens=request.constraints.get("output_tokens", None),
                response_prompt_tokens=None,
                response_output_tokens=response.iter_count,
                request_id=request.request_id,
                error=error or "Unknown error",
            )

        response.error = error
        status.request_start = response.start_time
        status.request_end = response.end_time

        return status, response
