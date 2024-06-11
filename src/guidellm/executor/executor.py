from typing import Dict, Union, Optional, Any
from guidellm.core import BenchmarkReport
from guidellm.request import RequestGenerator
from guidellm.backend import Backend
from guidellm.scheduler.scheduler import Scheduler
from guidellm.executor.profile_generator import ProfileGenerator, ProfileGenerationModes


__all__ = ["Executor"]


class Executor:
    def __init__(
        self,
        request_generator: RequestGenerator,
        backend: Backend,
        profile_mode: Union[str, ProfileGenerationModes] = "single",
        profile_args: Optional[Dict[str, Any]] = None,
        max_requests: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self.request_generator = request_generator
        self.backend = backend
        self.profile = ProfileGenerator.create_generator(
            profile_mode, **(profile_args or {})
        )
        self.max_requests = max_requests
        self.max_duration = max_duration

    def run(self) -> BenchmarkReport:
        report = BenchmarkReport()

        while True:
            profile = self.profile.next_profile(report)

            if profile is None:
                break

            scheduler = Scheduler(
                request_generator=self.request_generator,
                backend=self.backend,
                load_gen_mode=profile.load_gen_mode,
                load_gen_rate=profile.load_gen_rate,
                max_requests=self.max_requests,
                max_duration=self.max_duration,
            )

            benchmark = scheduler.run()
            report.add_benchmark(benchmark)

        return report
