from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Union

from loguru import logger

from guidellm.backend import Backend
from guidellm.core import TextGenerationBenchmarkReport
from guidellm.executor.profile_generator import ProfileGenerationMode, ProfileGenerator
from guidellm.request import RequestGenerator
from guidellm.scheduler import Scheduler, SchedulerResult

__all__ = ["Executor", "ExecutorResult"]


@dataclass
class ExecutorResult:
    """
    Data class representing the result of executing tasks in the Executor.

    :param completed: Indicates whether all tasks have completed.
    :type completed: bool
    :param count_total: Total number of profiles.
    :type count_total: int
    :param count_completed: Number of completed profiles.
    :type count_completed: int
    :param report: A benchmark report for text generation.
    :type report: TextGenerationBenchmarkReport
    :param scheduler_result: Optional scheduler result for the last task.
    :type scheduler_result: Optional[SchedulerResult]
    """

    completed: bool
    count_total: int
    count_completed: int
    report: TextGenerationBenchmarkReport
    scheduler_result: Optional[SchedulerResult] = None


class Executor:
    """
    The Executor class manages the execution of tasks based on a given profile
    generation mode and rate. It orchestrates the interaction between the backend,
    request generator, and profile generator, and runs benchmarks accordingly.

    :param backend: The backend to run tasks against.
    :type backend: Backend
    :param request_generator: The generator that creates requests for execution.
    :type request_generator: RequestGenerator
    :param mode: The mode for profile generation (e.g., sweep, synchronous).
    :type mode: ProfileGenerationMode
    :param rate: The list of rates for load generation, or None.
    :type rate: Optional[List[float]]
    :param max_number: Maximum number of requests to generate for the scheduler
        (a single benchmark run), or None.
    :type max_number: Optional[int]
    :param max_duration: Maximum duration for generating requests for the scheduler,
        (a single benchmark run), or None.
    :type max_duration: Optional[float]
    """

    def __init__(
        self,
        backend: Backend,
        request_generator: RequestGenerator,
        mode: ProfileGenerationMode = "sweep",
        rate: Optional[Union[float, List[float]]] = None,
        max_number: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self._backend = backend
        self._generator = request_generator
        self._max_number = max_number
        self._max_duration = max_duration
        self._profile_generator = ProfileGenerator(mode=mode, rate=rate)
        logger.info("Executor initialized with mode: {}, rate: {}", mode, rate)

    @property
    def backend(self) -> Backend:
        """
        Returns the backend being used by the Executor.

        :return: Backend
        :rtype: Backend
        """
        return self._backend

    @property
    def request_generator(self) -> RequestGenerator:
        """
        Returns the request generator used by the Executor.

        :return: RequestGenerator
        :rtype: RequestGenerator
        """
        return self._generator

    @property
    def profile_generator(self) -> ProfileGenerator:
        """
        Returns the profile generator for generating profiles during execution.

        :return: ProfileGenerator
        :rtype: ProfileGenerator
        """
        return self._profile_generator

    @property
    def max_number(self) -> Optional[int]:
        """
        Returns the maximum number of requests to generate.

        :return: Maximum number of requests or None.
        :rtype: Optional[int]
        """
        return self._max_number

    @property
    def max_duration(self) -> Optional[float]:
        """
        Returns the maximum duration for generating requests.

        :return: Maximum duration in seconds or None.
        :rtype: Optional[float]
        """
        return self._max_duration

    async def run(self) -> AsyncGenerator[ExecutorResult, None]:
        """
        Runs the Executor, generating and scheduling tasks based on the profile
        generation mode. Yields results incrementally.

        :rtype: AsyncGenerator[ExecutorResult, None]
        """
        report = TextGenerationBenchmarkReport()
        report.args = {
            "mode": self.profile_generator.mode,
            "rate": self.profile_generator.rates,
            "max_number": self.max_number,
            "max_duration": self.max_duration,
        }
        logger.info("Starting Executor run")

        yield ExecutorResult(
            completed=False,
            count_total=len(self.profile_generator),
            count_completed=0,
            report=report,
        )

        while profile := self.profile_generator.next(report):
            logger.debug("Generated profile: {}", profile)
            scheduler = Scheduler(
                generator=self.request_generator,
                worker=self.backend,
                mode=profile.load_gen_mode,
                rate=profile.load_gen_rate,
                max_number=self.max_number,
                max_duration=self.max_duration,
            )

            logger.info(
                "Scheduling tasks with mode: {}, rate: {}",
                profile.load_gen_mode,
                profile.load_gen_rate,
            )

            async for scheduler_result in scheduler.run():
                if scheduler_result.completed:
                    report.add_benchmark(scheduler_result.benchmark)
                    logger.debug(
                        "Benchmark added for scheduler result: {}",
                        scheduler_result.benchmark,
                    )

                yield ExecutorResult(
                    completed=False,
                    count_total=len(self.profile_generator),
                    count_completed=len(report.benchmarks),
                    report=report,
                    scheduler_result=scheduler_result,
                )

        logger.info("Executor run completed")
        yield ExecutorResult(
            completed=True,
            count_total=len(self.profile_generator),
            count_completed=len(report.benchmarks),
            report=report,
        )
