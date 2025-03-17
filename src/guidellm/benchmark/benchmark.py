from typing import Any, Dict, List, Optional, TypeVar

from pydantic import Field, computed_field

from guidellm.benchmark.profile import Profile
from guidellm.objects import (
    DistributionSummary,
    Serializable,
    StatusDistributionSummary,
)
from guidellm.scheduler import SchedulingStrategy

__all__ = [
    "BENCH",
    "Benchmark",
    "GenerativeBenchmark",
]


class BenchmarkSettings(Serializable):
    profile: Profile
    profile_index: int
    strategy: SchedulingStrategy
    max_number: Optional[int]
    max_duration: Optional[float]
    warmup_number: Optional[int]
    warmup_duration: Optional[float]
    cooldown_number: Optional[int]
    cooldown_duration: Optional[float]


class BenchmarkRunStats(Serializable):
    run_start_time: float
    run_end_time: float

    completed: int
    errored: int
    total: int

    queued_time_avg: float
    scheduled_time_avg: float
    worker_time_avg: float
    worker_delay_avg: float
    request_delay_avg: float
    process_idle_time_avg: float


class Benchmark(Serializable):
    settings: BenchmarkSettings
    run_stats: BenchmarkRunStats
    worker_description: Serializable
    requests_loader_description: Serializable
    extras: Dict[str, Any] = Field(
        default_factory=dict,
    )

    requests_per_second: StatusDistributionSummary
    requests_concurrency: StatusDistributionSummary


BENCH = TypeVar("BENCH", bound=Benchmark)


class GenerativeTextResponseStats(Serializable):
    request_id: str
    prompt: str
    output: str
    prompt_tokens: int
    output_tokens: int
    start_time: float
    end_time: float
    first_token_time: float
    last_token_time: float

    @computed_field
    @property
    def request_latency(self) -> float:
        return self.end_time - self.start_time

    @computed_field
    @property
    def time_to_first_token_ms(self) -> float:
        return 1000 * (self.first_token_time - self.start_time)

    @computed_field
    @property
    def inter_token_latency_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0

        return (
            1000
            * (self.last_token_time - self.first_token_time)
            / (self.output_tokens - 1)
        )

    @computed_field
    @property
    def output_tokens_per_second(self) -> float:
        if (itl_ms := self.inter_token_latency_ms) == 0.0:
            return 0.0

        return 1000.0 / itl_ms


class GenerativeTextErrorStats(GenerativeTextResponseStats):
    error: str
    request_id: str
    prompt: str
    output: Optional[str]
    prompt_tokens: int
    output_tokens: Optional[int]
    start_time: float
    end_time: None = None  # no end since it failed
    first_token_time: Optional[float]
    last_token_time: Optional[float]

    @computed_field
    @property
    def request_latency(self) -> None:
        return None

    @computed_field
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None

        return 1000 * (self.first_token_time - self.start_time)

    @computed_field
    @property
    def inter_token_latency_ms(self) -> Optional[float]:
        if (
            self.output_tokens is None
            or self.output_tokens <= 1
            or self.first_token_time is None
            or self.last_token_time is None
        ):
            return None

        return (
            1000
            * (self.last_token_time - self.first_token_time)
            / (self.output_tokens - 1)
        )

    @computed_field
    @property
    def output_tokens_per_second(self) -> Optional[float]:
        if (itl_ms := self.inter_token_latency_ms) is None:
            return None

        return 1000.0 / itl_ms


class GenerativeBenchmark(Benchmark):
    completed_requests: List[GenerativeTextResponseStats] = Field(
        description="The list of completed requests.",
    )
    completed_sampled_size: Optional[int] = None
    errored_requests: List[GenerativeTextErrorStats] = Field(
        description="The list of errored requests.",
    )
    errored_sampled_size: Optional[int] = None

    start_time: float = Field(
        description="The start time of the first request for the benchmark.",
    )
    end_time: float = Field(
        description="The end time of the last request for the benchmark.",
    )

    requests_latency: DistributionSummary = Field(
        description="The distribution of latencies for the completed requests.",
    )
    prompts_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the prompts for completed, "
            "errored, and all requests."
        )
    )
    outputs_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the outputs for completed, "
            "errored, and all requests."
        )
    )
    times_to_first_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies to receiving the first token in "
            "milliseconds for completed, errored, and all requests."
        ),
    )
    inter_token_latencies_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies between tokens in milliseconds for "
            "completed, errored, and all requests."
        ),
    )
    outputs_tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of output tokens per second for completed, "
            "errored, and all requests."
        ),
    )

    @computed_field
    @property
    def duration(self) -> float:
        """
        :return: The duration of the benchmark in seconds from the start of the
            first request to the end of the last request.
        """
        return self.end_time - self.start_time

    @staticmethod
    def from_stats(
        completed: List[GenerativeTextResponseStats],
        errored: List[GenerativeTextErrorStats],
        settings: BenchmarkSettings,
        run_stats: BenchmarkRunStats,
        worker_description: Serializable,
        requests_loader_description: Serializable,
        extras: Dict[str, Any],
    ) -> "GenerativeBenchmark":
        pass  # TODO
