import uuid
from typing import Any, Generic, Literal, Optional, TypedDict, TypeVar, Union

from pydantic import Field, computed_field

from guidellm.backend import GenerationRequestTimings
from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    Profile,
    SweepProfile,
    SynchronousProfile,
    ThroughputProfile,
)
from guidellm.objects import (
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
)
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    ScheduledRequestInfo,
    SchedulerState,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)

__all__ = [
    "Benchmark",
    "BenchmarkMetrics",
    "BenchmarkSchedulerStats",
    "BenchmarkT",
    "GenerativeBenchmark",
    "GenerativeMetrics",
    "GenerativeRequestStats",
    "StatusBreakdown",
]


class BenchmarkSchedulerStats(StandardBaseModel):
    """
    A serializable model representing the run process statistics for the
    entire benchmark run across all requests including warmup and cooldown.
    """

    start_time: float = Field(
        description="The start time of the benchmark run.",
    )
    end_time: float = Field(
        description="The end time of the benchmark run.",
    )
    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description=(
            "The number of requests made for the benchmark run broken down by "
            "status including successful, incomplete, errored, and the sum of all three"
        )
    )
    queued_time_avg: float = Field()
    worker_resolve_start_delay_avg: float = Field()
    worker_resolve_time_avg: float = Field()
    worker_resolve_end_delay_avg: float = Field()
    finalized_delay_avg: float = Field()
    worker_targeted_start_delay_avg: float = Field()
    request_start_delay_avg: float = Field()
    request_time_avg: float = Field()
    request_targeted_delay_avg: float = Field()


class SchedulerDict(TypedDict, total=False):
    strategy: Union[
        AsyncConstantStrategy,
        AsyncPoissonStrategy,
        ConcurrentStrategy,
        SynchronousStrategy,
        ThroughputStrategy,
        SchedulingStrategy,
    ]
    constraints: dict[str, dict[str, Any]]
    state: SchedulerState


class BenchmarkerDict(TypedDict, total=False):
    profile: Union[
        AsyncProfile,
        ConcurrentProfile,
        SynchronousProfile,
        ThroughputProfile,
        SweepProfile,
        Profile,
    ]
    requests: dict[str, Any]
    backend: dict[str, Any]
    environment: dict[str, Any]
    aggregators: dict[str, dict[str, Any]]


class BenchmarkMetrics(StandardBaseModel):
    """
    A serializable model representing the metrics for a benchmark run.
    """

    requests_per_second: StatusDistributionSummary = Field(
        description="The distribution of requests per second for the benchmark.",
    )
    request_concurrency: StatusDistributionSummary = Field(
        description="The distribution of requests concurrency for the benchmark.",
    )
    request_latency: StatusDistributionSummary = Field(
        description="The distribution of latencies for the completed requests.",
    )


BenchmarkMetricsT = TypeVar("BenchmarkMetricsT", bound=BenchmarkMetrics)


class BenchmarkRequestStats(StandardBaseModel):
    scheduler_info: ScheduledRequestInfo[GenerationRequestTimings] = Field(
        description=(
            "The info about the request from the scheduler about how it was run."
        ),
    )


BenchmarkRequestStatsT = TypeVar("BenchmarkRequestStatsT", bound=BenchmarkRequestStats)


class Benchmark(StandardBaseModel, Generic[BenchmarkMetricsT, BenchmarkRequestStatsT]):
    """
    The base serializable model representing a benchmark run and its results.
    Specific benchmarker implementations should extend this model to include
    additional information or metadata as needed.

    Note, requests_per_second and request_concurrency are kept at this level
    and are expected to be populated by the subclass implementation to ensure
    the logic for Profiles can include more complicated logic for determining
    what rates and concurrency values to use for subsequent strategies.
    """

    # Benchmark run information
    type_: Literal["benchmark"] = "benchmark"
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the benchmark.",
    )
    run_id: str = Field(
        description=(
            "The unique identifier for the encompasing benchmarker run that this "
            "benchmark was a part of."
        )
    )
    run_index: int = Field(
        description=("The index of this benchmark in the benchmarker run.")
    )
    scheduler: SchedulerDict = Field()
    benchmarker: BenchmarkerDict = Field()
    env_args: dict[str, Any] = Field()
    extras: dict[str, Any] = Field()

    # Benchmark stats and metrics
    run_stats: BenchmarkSchedulerStats = Field()
    start_time: float = Field(
        description="The start time of the first request for the benchmark.",
        default=-1.0,
    )
    end_time: float = Field(
        description="The end time of the last request for the benchmark.",
        default=-1.0,
    )

    @computed_field  # type: ignore[misc]
    @property
    def duration(self) -> float:
        """
        :return: The duration of the benchmark in seconds from the start of the
            first request to the end of the last request.
        """
        return self.end_time - self.start_time

    metrics: BenchmarkMetricsT = Field(
        description=(
            "The metrics for the benchmark run represented as a distribution of "
            "various per-request statistics."
        ),
    )

    # Benchmark response stats (ordered at the end for readability)
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description=(
            "The number of requests made for the benchmark broken down by status "
            "including successful, incomplete, errored, and the sum of all three"
        )
    )
    requests: StatusBreakdown[
        list[BenchmarkRequestStatsT],
        list[BenchmarkRequestStatsT],
        list[BenchmarkRequestStatsT],
        None,
    ] = Field(
        description=(
            "The breakdown of requests for the benchmark run including successful, "
            "incomplete, and errored requests."
        ),
    )


BenchmarkT = TypeVar("BenchmarkT", bound=Benchmark)


class GenerativeRequestStats(BenchmarkRequestStats):
    """
    A serializable model representing the request values, response values, and
    statistics for a generative text response.
    """

    type_: Literal["generative_request_stats"] = "generative_request_stats"
    request_id: str = Field(
        description="The unique identifier for the request.",
    )
    request_type: Literal["text_completions", "chat_completions"] = Field(
        description="The type of request made to the generative backend."
    )
    prompt: str = Field(
        description="The text prompt used for the generative request.",
    )
    request_args: dict[str, Any] = Field(
        description="The parameters used for the generative request.",
    )
    output: Optional[str] = Field(
        description="The generated text output from the generative request.",
    )
    iterations: int = Field(
        description="The number of iterations the request went through.",
    )
    prompt_tokens: Optional[int] = Field(
        description="The number of tokens in the prompt text.",
    )
    output_tokens: Optional[int] = Field(
        description="The number of tokens in the generated output text.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> Optional[float]:
        """
        :return: The duration of the request in seconds from the start to the end.
        """
        if (
            not self.scheduler_info.request_timings.request_end
            or not self.scheduler_info.request_timings.request_start
        ):
            return None

        return (
            self.scheduler_info.request_timings.request_end
            - self.scheduler_info.request_timings.request_start
        )

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        """
        :return: The time in milliseconds from the start of the request to the first
            token received.
        """
        if (
            not self.scheduler_info.request_timings.first_iteration
            or not self.scheduler_info.request_timings.request_start
        ):
            return None

        return 1000 * (
            self.scheduler_info.request_timings.first_iteration
            - self.scheduler_info.request_timings.request_start
        )

    @computed_field  # type: ignore[misc]
    @property
    def time_per_output_token_ms(self) -> Optional[float]:
        """
        :return: The average time in milliseconds per output token generated.
            This includes the time to generate the first token and all other tokens.
        """
        if (
            not self.scheduler_info.request_timings.request_start
            or not self.scheduler_info.request_timings.last_iteration
            or not self.output_tokens
        ):
            return None

        return (
            1000
            * (
                self.scheduler_info.request_timings.last_iteration
                - self.scheduler_info.request_timings.request_start
            )
            / self.output_tokens
        )

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency_ms(self) -> Optional[float]:
        """
        :return: The average time in milliseconds between generating tokens in the
            output text. Note, does not include the time to generate the first token.
        """
        if (
            not self.scheduler_info.request_timings.first_iteration
            or not self.scheduler_info.request_timings.last_iteration
            or not self.output_tokens
            or self.output_tokens <= 1
        ):
            return None

        return (
            1000
            * (
                self.scheduler_info.request_timings.last_iteration
                - self.scheduler_info.request_timings.first_iteration
            )
            / (self.output_tokens - 1)
        )

    @computed_field  # type: ignore[misc]
    @property
    def tokens_per_second(self) -> Optional[float]:
        """
        :return: The average number of tokens generated per second in the prompt and
            output text.
        """
        if (
            not (latency := self.request_latency)
            or not self.prompt_tokens
            or not self.output_tokens
        ):
            return None

        return (self.prompt_tokens + self.output_tokens) / latency

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> Optional[float]:
        """
        :return: The average number of output tokens generated per second.
        """
        if not (latency := self.request_latency) or not self.output_tokens:
            return None

        return self.output_tokens / latency


class GenerativeMetrics(BenchmarkMetrics):
    """
    A serializable model representing the metrics for a generative benchmark run.
    """

    prompt_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the prompts for completed, "
            "errored, and all requests."
        )
    )
    output_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the outputs for completed, "
            "errored, and all requests."
        )
    )
    time_to_first_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies to receiving the first token in "
            "milliseconds for completed, errored, and all requests."
        ),
    )
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies per output token in milliseconds for "
            "completed, errored, and all requests. "
            "This includes the time to generate the first token and all other tokens."
        ),
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies between tokens in milliseconds for "
            "completed, errored, and all requests."
        ),
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of output tokens per second for completed, "
            "errored, and all requests."
        ),
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of tokens per second, including prompt and output tokens "
            "for completed, errored, and all requests."
        ),
    )


class GenerativeBenchmark(Benchmark[GenerativeMetrics, GenerativeRequestStats]):
    """
    A serializable model representing a benchmark run and its results for generative
    requests and responses. Includes the completed and errored requests, the start
    and end times for the benchmark, and the statistics for the requests and responses.
    """

    type_: Literal["generative_benchmark"] = "generative_benchmark"  # type: ignore[assignment]
