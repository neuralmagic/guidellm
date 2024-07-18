import asyncio
import functools
import time
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple

from loguru import logger

from guidellm.backend import Backend
from guidellm.core import (
    TextGenerationBenchmark,
    TextGenerationError,
    TextGenerationResult,
)
from guidellm.request import RequestGenerator

from .load_generator import LoadGenerationMode, LoadGenerator

__all__ = ["Scheduler"]


class Scheduler:
    """
    The scheduler class is responsible for handling tasks and running
    """

    def __init__(
        self,
        request_generator: RequestGenerator,
        backend: Backend,
        load_gen_mode: LoadGenerationMode = LoadGenerationMode.SYNCHRONOUS,
        load_gen_rate: Optional[float] = None,
        max_requests: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        if max_requests is None and max_duration is None:
            raise ValueError("Either num_requests or duration must be specified")

        if (max_requests is not None and max_requests <= 0) or (
            max_duration is not None and max_duration <= 0
        ):
            raise ValueError("max_requests anx max_duration must be > 0")

        if load_gen_mode != LoadGenerationMode.SYNCHRONOUS and load_gen_rate is None:
            raise ValueError(
                "Rate must be specified for non-synchronous load generation modes"
            )

        self._request_generator = request_generator
        self._backend = backend
        self._load_gen_mode = load_gen_mode
        self._load_gen_rate = load_gen_rate
        self._max_requests = max_requests
        self._max_duration = max_duration

    def run(self) -> TextGenerationBenchmark:
        if self._load_gen_mode == LoadGenerationMode.SYNCHRONOUS:
            report = self._run_sync()
        else:
            report = asyncio.run(self._run_async())

        return report

    @property
    def load_generator(self) -> LoadGenerator:
        if not self._load_gen_rate:
            raise ValueError("Invalid empty value for self._load_gen_rate")

        return LoadGenerator(self._load_gen_mode, self._load_gen_rate)

    def _cancel_running_tasks(
        self,
        tasks: Iterable[Tuple[asyncio.Task, Dict]],
        benchmark: TextGenerationBenchmark,
    ) -> None:
        """
        Cancel all the running tasks for the scheduler and augment the
        benchmark with error reports.

        :param tasks: The `tasks` iterable batch. Where the batch includes
            the asyncio.Task and the signature context of that task.
        """

        for task, context in tasks:
            if not task.done():
                logger.debug(f"Cancelling running task {task}")
                task.cancel()
                benchmark.errors.append(
                    # TODO: Extract the data from the Coroutine parameters
                    TextGenerationError(**context, error_class=asyncio.CancelledError())
                )

    def _run_sync(self) -> TextGenerationBenchmark:
        benchmark = TextGenerationBenchmark(mode=self._load_gen_mode.value, rate=None)
        start_time = time.time()
        requests_counter = 0

        for callback in self._sync_tasks():
            if (
                self._max_requests is not None
                and requests_counter >= self._max_requests
            ) or (
                self._max_duration is not None
                and time.time() - start_time >= self._max_duration
            ):
                break

            benchmark.request_started()
            res = callback()
            benchmark.request_completed(res)

            requests_counter += 1

        return benchmark

    async def _run_async(self) -> TextGenerationBenchmark:
        """
        Running in async mode determines next steps:
        * Iterate through all the tasks with load attached
        * Check the execution time does not go over the max duration
        * Check the number of requests is not greater than max requests

        If the max duration is not specified for the scheduler - check only
        max requests and just break the loop without cancelling tasks.
        """

        benchmark: TextGenerationBenchmark = TextGenerationBenchmark(
            mode=self._load_gen_mode.value, rate=self._load_gen_rate
        )
        requests_counter: int = 0
        tasks: List[Tuple[asyncio.Task, Dict]] = []
        start_time: float = time.time()

        for _task, task_start_time in zip(
            self._async_tasks(benchmark), self.load_generator.times()
        ):
            task, task_context = _task
            tasks.append((task, task_context))
            requests_counter += 1

            if (
                self._max_duration is not None
                and time.time() - start_time >= self._max_duration
            ):
                self._cancel_running_tasks(tasks=tasks, benchmark=benchmark)
                break
            elif (
                self._max_requests is not None
                and requests_counter >= self._max_requests
            ):
                break

            if (pending_time := task_start_time - time.time()) > 0:
                await asyncio.sleep(pending_time)

        if self._max_duration is None:
            await asyncio.gather(*(t for t, _ in tasks))
        else:
            try:
                # Set the timeout if the max duration is specified
                await asyncio.wait_for(
                    asyncio.gather(*(t for t, _ in tasks), return_exceptions=True),
                    self._max_duration,
                )
            except TimeoutError:
                self._cancel_running_tasks(tasks=tasks, benchmark=benchmark)

        return benchmark

    def _sync_tasks(self) -> Generator[Callable[..., TextGenerationResult], None, None]:
        """
        Iterate through `Backend.submit()` sync callbacks.
        """

        for request in self._request_generator:
            yield functools.partial(self._backend.submit, request=request)

    def _async_tasks(
        self, benchmark: TextGenerationBenchmark
    ) -> Generator[Tuple[asyncio.Task, Dict], None, None]:
        """
        Iterate through `Backend.submit()` async tasks.
        """

        for request in self._request_generator:
            submit_payload = {"request": request}
            task: asyncio.Task = asyncio.create_task(
                self._run_task_async(benchmark=benchmark, **submit_payload),
                name=f"Backend.submit({request.prompt})",
            )

            yield task, submit_payload

    async def _run_task_async(
        self, benchmark: TextGenerationBenchmark, **backend_submit_payload
    ):
        benchmark.request_started()
        try:
            res = await self._event_loop.run_in_executor(
                None, functools.partial(self._backend.submit, **backend_submit_payload)
            )
        except Exception as error:
            benchmark.errors.append(
                TextGenerationError(
                    **backend_submit_payload, error_class=asyncio.CancelledError()
                )
            )
        else:
            benchmark.request_completed(res)

    @property
    def _event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()
        else:
            return loop
