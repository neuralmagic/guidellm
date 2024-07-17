import asyncio
import time
from typing import Generator, List, Optional, Tuple

from loguru import logger

from guidellm.backend import Backend
from guidellm.core import TextGenerationBenchmark, TextGenerationError
from guidellm.request import RequestGenerator
from guidellm.scheduler.load_generator import LoadGenerationMode, LoadGenerator
from guidellm.scheduler.task import Task

__all__ = ["Scheduler"]


class Scheduler:
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

        # Tasks that scheduler is going to manage.
        # NOTE: Tasks are populated in sync/async manner and limited by
        #       the max number of requests and max duration on the execution.
        self._tasks: List[Tuple[asyncio.Task, Task]] = []

    def __len__(self) -> int:
        """
        The length of the scheduler
        is the number of total tasks in the processing at the moment.
        """

        return len(self._tasks)

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()
        else:
            return loop

    def run(self) -> TextGenerationBenchmark:
        if self._load_gen_mode == LoadGenerationMode.SYNCHRONOUS:
            report = self._run_sync()
        else:
            report = asyncio.run(self._run_async())

        return report

    def _run_sync(self) -> TextGenerationBenchmark:
        benchmark = TextGenerationBenchmark(mode=self._load_gen_mode.value, rate=None)
        start_time = time.time()
        requests_counter = 0

        for task in self._task_iterator():
            benchmark.request_started()
            res = task.run_sync()
            benchmark.request_completed(res)
            requests_counter += 1

            if (
                self._max_requests is not None
                and requests_counter >= self._max_requests
            ) or (
                self._max_duration is not None
                and time.time() - start_time >= self._max_duration
            ):
                break

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
        if not self._load_gen_rate:
            raise ValueError("Invalid empty value for self._load_gen_rate")
        load_gen = LoadGenerator(self._load_gen_mode, self._load_gen_rate)

        start_time: float = time.time()
        requests_counter: int = 0

        for task, task_start_time in zip(self._task_iterator(), load_gen.times()):
            if (
                self._max_duration is not None
                and time.time() - start_time >= self._max_duration
            ):
                self.cancel_running_tasks(benchmark)
                break
            elif (
                self._max_requests is not None
                and requests_counter >= self._max_requests
            ):
                break

            pending_time = task_start_time - time.time()

            if pending_time > 0:
                await asyncio.sleep(pending_time)

            self._tasks.append(
                (asyncio.create_task(self._run_task_async(task, benchmark)), task)
            )

            requests_counter += 1

        if self._max_duration is None:
            await asyncio.gather(
                *(asyncio_task for asyncio_task, _ in self._tasks),
                return_exceptions=False,
            )
        else:
            try:
                # Set the timeout if the max duration is specified
                await asyncio.wait_for(
                    asyncio.gather(
                        *(asyncio_task for asyncio_task, _ in self._tasks),
                        return_exceptions=True,
                    ),
                    self._max_duration,
                )
            except TimeoutError:
                self.cancel_running_tasks(benchmark)

        return benchmark

    def cancel_running_tasks(self, benchmark: TextGenerationBenchmark) -> None:
        """
        Cancel all the running tasks for the scheduler
        """

        for asyncio_task, guidellm_task in self._tasks:
            if not asyncio_task.done():
                logger.debug(f"Cancelling running task {asyncio_task}")
                asyncio_task.cancel()
                benchmark.errors.append(
                    TextGenerationError(
                        **guidellm_task._params, error_class=asyncio.CancelledError()
                    )
                )

    async def _run_task_async(self, task: Task, benchmark: TextGenerationBenchmark):
        benchmark.request_started()
        res = await task.run_async(self.event_loop)
        benchmark.request_completed(res)

    def _task_iterator(self) -> Generator[Task, None, None]:
        for request in self._request_generator:
            yield Task(
                func=self._backend.submit,
                params={"request": request},
                err_container=TextGenerationError,
            )
