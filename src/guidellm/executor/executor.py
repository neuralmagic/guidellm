from typing import Any, Dict, Optional

from guidellm.backend import Backend
from guidellm.core import TextGenerationBenchmarkReport
from guidellm.request import RequestGenerator
from guidellm.scheduler import Scheduler

from .profile_generator import ProfileGenerationMode, ProfileGenerator

__all__ = ["Executor"]


class Executor:
    """
    The main purpose of the `class Executor` is to dispatch running tasks according
    to the Profile Generation mode
    """

    def __init__(
        self,
        backend: Backend,
        request_generator: RequestGenerator,
        profile_mode: ProfileGenerationMode = ProfileGenerationMode.SINGLE,
        profile_args: Optional[Dict[str, Any]] = None,
        max_requests: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self.request_generator = request_generator
        self.backend = backend
        self.profile_generator: ProfileGenerator = ProfileGenerator.create(
            profile_mode, **(profile_args or {})
        )
        self.max_requests: Optional[int] = max_requests
        self.max_duration: Optional[float] = max_duration
        self._scheduler: Optional[Scheduler] = None

    @property
    def scheduler(self) -> Scheduler:
        if self._scheduler is None:
            raise ValueError("The scheduler is not defined. Did you run the execution?")
        else:
            return self._scheduler

    @scheduler.setter
    def scheduler(self, value: Any):
        if not isinstance(value, Scheduler):
            raise TypeError(
                "Only Scheduler instances could be set as a self._scheduler"
            )
        else:
            self._scheduler = value

    def run(self) -> TextGenerationBenchmarkReport:
        report = TextGenerationBenchmarkReport()

        while True:
            if not (profile := self.profile_generator.next(report)):
                break

            self.scheduler = Scheduler(
                request_generator=self.request_generator,
                backend=self.backend,
                load_gen_mode=profile.load_gen_mode,
                load_gen_rate=profile.load_gen_rate,
                max_requests=self.max_requests,
                max_duration=self.max_duration,
            )

            benchmark = self.scheduler.run()
            report.add_benchmark(benchmark)

        return report
