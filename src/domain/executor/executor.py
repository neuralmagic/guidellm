from typing import Any, Dict, Optional

from domain.backend import Backend
from domain.core import TextGenerationBenchmarkReport
from domain.executor.profile_generator import ProfileGenerationMode, ProfileGenerator
from domain.request import RequestGenerator
from domain.scheduler.scheduler import Scheduler


class Executor:
    def __init__(
        self,
        backend: Backend,
        request_generator: RequestGenerator,
        profile_mode: ProfileGenerationMode = ProfileGenerationMode.SINGLE,
        profile_args: Optional[Dict[str, Any]] = None,
        max_requests: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self.backend: Backend = backend
        self.request_generator: RequestGenerator = request_generator
        self.profile_generator: ProfileGenerator = ProfileGenerator.create(
            profile_mode, **(profile_args or {})
        )
        self.max_requests: Optional[int] = max_requests
        self.max_duration: Optional[float] = max_duration

    def run(self) -> TextGenerationBenchmarkReport:
        report = TextGenerationBenchmarkReport()

        for profile in self.profile_generator:
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
