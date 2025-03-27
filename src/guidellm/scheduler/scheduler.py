import asyncio
import math
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Optional, Union, get_args

from loguru import logger

from guidellm.backend import Backend, ResponseSummary, StreamingTextResponse
from guidellm.config import settings
from guidellm.core import (
    TextGenerationBenchmark,
    TextGenerationError,
    TextGenerationRequest,
    TextGenerationResult,
)
from guidellm.request import RequestGenerator
from guidellm.scheduler.load_generator import LoadGenerationMode, LoadGenerator

__all__ = ["Scheduler", "SchedulerResult"]


@dataclass
class SchedulerResult:
    """
    Represents the result of a single task execution within the Scheduler.

    :param completed: Indicates if the task is completed.
    :type completed: bool
    :param count_total: Total number of tasks to be executed.
    :type count_total: int
    :param count_completed: Number of tasks that have been completed so far.
    :type count_completed: int
    :param report: Benchmark data for the task execution.
    :type benchmark: TextGenerationBenchmark
    :param current_result: The result of the current request, if any.
    :type current_result: Optional[Union[TextGenerationResult, Exception]]
    """

    completed: bool
    count_total: int
    count_completed: int
    benchmark: TextGenerationBenchmark
    current_result: Optional[Union[TextGenerationResult, TextGenerationError]] = None


class Scheduler:
    """
    Schedules and manages the execution of tasks for text generation requests.

    :param generator: The request generator that produces text generation requests.
    :type generator: RequestGenerator
    :param backend: The backend that processes the requests.
    :type backend: Backend
    :param mode: The mode of load generation (e.g., synchronous, asynchronous).
    :type mode: LoadGenerationMode
    :param rate: The rate at which requests are generated, if applicable.
    :type rate: Optional[float]
    :param max_number: Maximum number of requests to be processed.
    :type max_number: Optional[int]
    :param max_duration: Maximum duration in seconds for which requests
        should be processed.
    :type max_duration: Optional[float]

    :raises ValueError: If neither max_number nor max_duration is specified or
        if they are not positive.
    """

    def __init__(
        self,
        generator: RequestGenerator,
        backend: Backend,
        mode: LoadGenerationMode = "synchronous",
        rate: Optional[float] = None,
        max_number: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        logger.info(
            "Scheduler initialized with params: generator={}, backend={}, mode={}, "
            "rate={}, max_number={}, max_duration={}",
            generator,
            backend,
            mode,
            rate,
            max_number,
            max_duration,
        )

        if mode not in get_args(LoadGenerationMode):
            err = ValueError(
                f"{mode} is not a valid Load Generation Mode. "
                f"Valid options are {get_args(LoadGenerationMode)}"
            )
            logger.error(err)
            raise err

        if not max_number and not max_duration:
            err = ValueError("Either max_number or max_duration must be specified")
            logger.error(err)
            raise err

        if max_number and max_number <= 0:
            err = ValueError(f"max_number must be > 0, given: {max_number}")
            logger.error(err)
            raise err

        if max_duration and max_duration <= 0:
            err = ValueError(f"max_duration must be > 0, given: {max_duration}")
            logger.error(err)
            raise err

        if mode in ["constant", "poisson"] and not rate:
            err = ValueError(f"Rate must be > 0 for mode: {mode}. Given: {rate}")
            logger.error(err)
            raise err

        self._generator = generator
        self._backend = backend
        self._mode = mode
        self._rate = rate
        self._max_number = max_number
        self._max_duration = max_duration

        self._load_generator = LoadGenerator(mode, rate)

    @property
    def generator(self) -> RequestGenerator:
        """
        The request generator that produces text generation requests.

        :return: The request generator instance.
        :rtype: RequestGenerator
        """
        return self._generator

    @property
    def backend(self) -> Backend:
        """
        The backend that processes the requests.

        :return: The backend instance.
        :rtype: Backend
        """
        return self._backend

    @property
    def mode(self) -> LoadGenerationMode:
        """
        The mode of load generation (e.g., synchronous, asynchronous).

        :return: The load generation mode.
        :rtype: LoadGenerationMode
        """
        return self._mode

    @property
    def rate(self) -> Optional[float]:
        """
        The rate at which requests are generated, if applicable.

        :return: The rate of request generation.
        :rtype: Optional[float]
        """
        return self._rate

    @property
    def max_number(self) -> Optional[int]:
        """
        Maximum number of requests to be processed.

        :return: The maximum number of requests.
        :rtype: Optional[int]
        """
        return self._max_number

    @property
    def max_duration(self) -> Optional[float]:
        """
        Maximum duration in seconds for which requests should be processed.

        :return: The maximum duration in seconds.
        :rtype: Optional[float]
        """
        return self._max_duration

    @property
    def load_generator(self) -> LoadGenerator:
        """
        The load generator responsible for generating load based on mode and rate.

        :return: The load generator instance.
        :rtype: LoadGenerator
        """
        return self._load_generator

    @property
    def benchmark_mode(self) -> Literal["asynchronous", "synchronous", "throughput"]:
        """
        The report mode for the scheduler.

        :return: The report mode.
        :rtype: Literal["asynchronous", "synchronous", "throughput"]
        """
        if self._mode == "synchronous":
            return "synchronous"

        if self._mode == "throughput":
            return "throughput"

        return "asynchronous"

    async def run(self) -> AsyncGenerator[SchedulerResult, None]:
        """
        Run the scheduler to process requests based on the configured mode, rate,
        maximum number, and maximum duration.

        :yield: The result of each task executed by the scheduler.
        :rtype: Generator[SchedulerResult, None, None]
        """
        logger.info("Starting Scheduler run")

        benchmark = TextGenerationBenchmark(mode=self.benchmark_mode, rate=self.rate)
        start_time = time.time()
        end_time = start_time + self.max_duration if self.max_duration else math.inf
        max_number = float(self.max_number) if self.max_number else math.inf
        runner = self._run_sync if self._mode == "synchronous" else self._run_async
        count_total = (
            self.max_number
            if self.max_number
            else round(self.max_duration)
            if self.max_duration
            else 0
        )

        # yield initial result for progress tracking
        yield SchedulerResult(
            completed=False,
            count_total=count_total,
            count_completed=0,
            benchmark=benchmark,
        )

        run_count = 0
        async for res in runner(benchmark, end_time, max_number):
            run_count += 1
            count_completed = (
                min(run_count, self.max_number)
                if self.max_number
                else round(time.time() - start_time)
                if self.max_duration
                else 0
            )

            yield SchedulerResult(
                completed=False,
                count_total=count_total,
                count_completed=count_completed,
                benchmark=benchmark,
                current_result=res,
            )

        logger.info("Scheduler run completed")

        yield SchedulerResult(
            completed=True,
            count_total=count_total,
            count_completed=(
                benchmark.request_count + benchmark.error_count
                if self.max_number
                else round(time.time() - start_time)
                if self.max_duration
                else 0
            ),
            benchmark=benchmark,
        )

    async def _run_sync(
        self, benchmark: TextGenerationBenchmark, end_time: float, max_number: float
    ) -> AsyncGenerator[Union[TextGenerationResult, TextGenerationError], None]:
        for index, (request, submit_at) in enumerate(
            zip(self.generator, self.load_generator.times())
        ):
            if index >= max_number or time.time() >= end_time:
                break

            logger.debug(
                "Running synchronous request={} at submit_at={}",
                request,
                submit_at,
            )
            benchmark.request_started()
            result = await self._scheduled_request(request, submit_at, end_time)
            if result is not None:
                benchmark.request_completed(result)
                logger.debug("Request completed with output: {}", result)
                yield result

    async def _run_async(
        self, benchmark: TextGenerationBenchmark, end_time: float, max_number: float
    ) -> AsyncGenerator[Union[TextGenerationResult, TextGenerationError], None]:
        tasks = []
        pending = asyncio.Semaphore(settings.max_concurrency)

        for index, (request, submit_at) in enumerate(
            zip(self.generator, self.load_generator.times())
        ):
            # wait for number of pending tasks to be >= max_concurrency
            await pending.acquire()

            if index >= max_number or time.time() >= end_time or submit_at >= end_time:
                break

            logger.debug(
                "Running asynchronous request={} at submit_at={}",
                request,
                submit_at,
            )

            def _completed(_task: asyncio.Task) -> None:
                # NOTE: this is only ok because we don't use threads/processes
                nonlocal pending
                pending.release()
                _res = _task.result()

                if _res:
                    benchmark.request_completed(_res)
                    logger.debug("Request completed: {}", _res)

            benchmark.request_started()
            task = asyncio.create_task(
                self._scheduled_request(request, submit_at, end_time)
            )
            task.add_done_callback(_completed)
            tasks.append(task)

            # release control to the event loop for other tasks
            await asyncio.sleep(0)

        for compl_task in asyncio.as_completed(tasks):
            task_res = await compl_task
            if task_res is not None:
                yield task_res

    async def _scheduled_request(
        self, request: TextGenerationRequest, submit_at: float, end_time: float
    ) -> Optional[Union[TextGenerationResult, TextGenerationError]]:
        try:
            if submit_at > end_time:
                raise asyncio.TimeoutError(
                    f"Request submission time {submit_at} "
                    f"is greater than end time {end_time}"
                )

            if submit_at > time.time():
                await asyncio.sleep(submit_at - time.time())

            timeout = (
                end_time - time.time() if end_time and end_time < math.inf else None
            )

            return await asyncio.wait_for(
                self._resolve_text_request(request), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, asyncio.TimeoutError):
                logger.warning("Request {} failed: {}", request, exc)

            return TextGenerationError(request=request, message=str(exc))

    async def _resolve_text_request(
        self, request: TextGenerationRequest
    ) -> TextGenerationResult:
        final_resp = None
        first_token_time = None
        last_token_time = None

        if request.type_ == "text":
            async for resp in self._backend.text_completions(  # type: ignore[attr-defined]
                prompt=request.prompt,
                id_=request.id,
                prompt_token_count=request.prompt_token_count,
                output_token_count=request.output_token_count,
            ):
                if isinstance(resp, StreamingTextResponse) and resp.type_ == "iter":
                    first_token_time = first_token_time or resp.time
                    last_token_time = resp.time

                final_resp = resp
        elif request.type_ == "chat":
            async for resp in self._backend.chat_completions(  # type: ignore[attr-defined]
                content=request.prompt,
                id_=request.id,
                prompt_token_count=request.prompt_token_count,
                output_token_count=request.output_token_count,
            ):
                if isinstance(resp, StreamingTextResponse) and resp.type_ == "iter":
                    first_token_time = first_token_time or resp.time
                    last_token_time = resp.time

                final_resp = resp

        if not final_resp or not isinstance(final_resp, ResponseSummary):
            raise ValueError(
                f"Invalid final response for request: {request} "
                f"and backend: {self._backend}, recieved: {final_resp}"
            )

        return TextGenerationResult(
            request=request,
            prompt_token_count=final_resp.prompt_tokens,
            output=final_resp.value,
            output_token_count=resp.output_tokens,
            start_time=resp.start_time,
            end_time=resp.end_time,
            first_token_time=first_token_time,
            last_token_time=last_token_time,
        )
