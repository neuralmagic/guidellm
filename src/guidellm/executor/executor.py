from typing import Any, Dict, Optional

from guidellm.backend import Backend
from guidellm.core import TextGenerationBenchmark, TextGenerationBenchmarkReport
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
        rate_type: str = "sweep",
        profile_args: Optional[Dict[str, Any]] = None,
        max_requests: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self.request_generator = request_generator
        self.backend = backend
        profile_mode = "sweep"
        if rate_type in {"synchronous", "constant", "poisson"}:
            profile_mode = ProfileGenerationMode.FIXED_RATE
        elif rate_type == "sweep":
            profile_mode = ProfileGenerationMode.SWEEP
        else:
            raise ValueError("invalid rate type provided")
        self.profile_generator: ProfileGenerator = ProfileGenerator.create(
            profile_mode, **(profile_args or {})
        )
        self.max_requests: Optional[int] = max_requests
        self.max_duration: Optional[float] = max_duration
        self._scheduler: Optional[Scheduler] = None

    @property
    def scheduler(self) -> Scheduler:
        if self._scheduler is None:
            raise ValueError("The scheduler is not set. Did you run the execution?")
        else:
            return self._scheduler

    def run(self) -> TextGenerationBenchmarkReport:
        report = TextGenerationBenchmarkReport()

        while True:
            if not (profile := self.profile_generator.next(report)):
                break

            scheduler = Scheduler(
                request_generator=self.request_generator,
                backend=self.backend,
                load_gen_mode=profile.load_gen_mode,
                load_gen_rate=profile.load_gen_rate,
                max_requests=self.max_requests,
                max_duration=self.max_duration,
            )

            benchmark: TextGenerationBenchmark = scheduler.run()
            report.add_benchmark(benchmark)

        return report
