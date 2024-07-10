import asyncio
import time
from typing import Iterable, Optional

from loguru import logger

from domain.backend import Backend
from domain.core import TextGenerationBenchmark, TextGenerationError
from domain.load_generator import LoadGenerationMode, LoadGenerator
from domain.request import RequestGenerator

from .task import Task

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

        if not isinstance(load_gen_mode, LoadGenerationMode):
            load_gen_mode = LoadGenerationMode(load_gen_mode)

        self._request_generator = request_generator
        self._backend = backend
        self._load_gen_mode = load_gen_mode
        self._load_gen_rate = load_gen_rate
        self._max_requests = max_requests
        self._max_duration = max_duration

    def run(self) -> TextGenerationBenchmark:
        if self._load_gen_mode == LoadGenerationMode.SYNCHRONOUS:
            result = self._run_sync()
        else:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._run_async())

        return result

    def _run_sync(self) -> TextGenerationBenchmark:
        result_set = TextGenerationBenchmark(mode=self._load_gen_mode, rate=None)
        start_time = time.time()
        counter = 0

        for task in self._task_iterator():
            result_set.request_started()
            res = task.run_sync()
            result_set.request_completed(res)
            counter += 1

            if (self._max_requests is not None and counter >= self._max_requests) or (
                self._max_duration is not None
                and time.time() - start_time >= self._max_duration
            ):
                break

        return result_set

    async def _run_async(self) -> TextGenerationBenchmark:
        if self._load_gen_rate is None:
            raise ValueError(
                "Rate must be specified for non-synchronous load generation modes"
            )

        result_set = TextGenerationBenchmark(
            mode=self._load_gen_mode, rate=self._load_gen_rate
        )
        load_gen = LoadGenerator(self._load_gen_mode, self._load_gen_rate)

        tasks = []
        start_time = time.time()
        counter = 0
        try:
            for task, task_start_time in zip(self._task_iterator(), load_gen.times()):
                pending_time = task_start_time - time.time()

                if pending_time > 0:
                    await asyncio.sleep(pending_time)

                tasks.append(
                    asyncio.create_task(self._run_task_async(task, result_set))
                )
                counter += 1

                if (
                    self._max_requests is not None and counter >= self._max_requests
                ) or (
                    self._max_duration is not None
                    and time.time() - start_time >= self._max_duration
                ):
                    break

            if self._max_duration is not None:
                pending_duration = self._max_duration - (time.time() - start_time)
                if pending_duration > 0:
                    await asyncio.sleep(pending_duration)
                raise asyncio.CancelledError()

            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

        return result_set

    async def _run_task_async(self, task: Task, result_set: TextGenerationBenchmark):
        result_set.request_started()
        res = await task.run_async()
        result_set.request_completed(res)

    def _task_iterator(self) -> Iterable[Task]:
        for request in self._request_generator:
            yield Task(
                func=self._backend.submit,
                params={"request": request},
                err_container=TextGenerationError,
            )
