"""
Benchmark data models and metrics for performance measurement and analysis.

Provides comprehensive data structures for capturing, storing, and analyzing
benchmark results from scheduler executions. Includes timing measurements,
token statistics, and performance metrics for generative AI workloads.

Classes:
    BenchmarkSchedulerStats: Scheduler timing and performance statistics.
    BenchmarkMetrics: Core benchmark metrics and distributions.
    BenchmarkRequestStats: Individual request processing statistics.
    Benchmark: Base benchmark result container with generic metrics.
    GenerativeRequestStats: Request statistics for generative AI workloads.
    GenerativeMetrics: Comprehensive metrics for generative benchmarks.
    GenerativeBenchmark: Complete generative benchmark results and analysis.
    GenerativeBenchmarksReport: Container for multiple benchmark results.

Type Variables:
    BenchmarkMetricsT: Generic benchmark metrics type.
    BenchmarkRequestStatsT: Generic request statistics type.
    BenchmarkT: Generic benchmark container type.
"""

import json
import uuid
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import yaml
from pydantic import Field, computed_field

from guidellm.backend import GenerationRequestTimings
from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
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
    SynchronousStrategy,
    ThroughputStrategy,
)

__all__ = [
    "Benchmark",
    "BenchmarkMetrics",
    "BenchmarkSchedulerStats",
    "BenchmarkT",
    "GenerativeBenchmark",
    "GenerativeBenchmarksReport",
    "GenerativeMetrics",
    "GenerativeRequestStats",
]


class BenchmarkSchedulerStats(StandardBaseModel):
    """Scheduler timing and performance statistics."""

    start_time: float = Field(
        description="Unix timestamp when the benchmark run started"
    )
    end_time: float = Field(description="Unix timestamp when the benchmark run ended")
    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )
    queued_time_avg: float = Field(
        description="Avg time requests spent in the queue (seconds)"
    )
    worker_resolve_start_delay_avg: float = Field(
        description="Avg delay before worker begins resolving req after dequeue (sec)"
    )
    worker_resolve_time_avg: float = Field(
        description="Avg time for worker to resolve requests (seconds)"
    )
    worker_resolve_end_delay_avg: float = Field(
        description="Avg delay after request end till worker resolves (seconds)"
    )
    finalized_delay_avg: float = Field(
        description="Avg delay after resolve til finalized with in scheduler (sec)"
    )
    worker_targeted_start_delay_avg: float = Field(
        description="Avg delay from targeted start to actual worker start (seconds)"
    )
    request_start_delay_avg: float = Field(
        description="Avg delay after resolve til request start (seconds)"
    )
    request_time_avg: float = Field(description="Avg request processing time (seconds)")
    request_targeted_delay_avg: float = Field(
        description="Avg delay from targeted start to actual request start"
    )


class SchedulerDict(StandardBaseModel):
    """Scheduler configuration and execution state dictionary."""

    strategy: Annotated[
        Union[
            AsyncConstantStrategy,
            AsyncPoissonStrategy,
            ConcurrentStrategy,
            SynchronousStrategy,
            ThroughputStrategy,
        ],
        Field(discriminator="type_"),
    ] = Field(description="Scheduling strategy configuration")
    constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Strategy execution constraints"
    )
    state: SchedulerState = Field(description="Scheduler execution state")


class BenchmarkerDict(StandardBaseModel):
    """Benchmarker configuration and component settings dictionary."""

    profile: Annotated[
        Union[
            AsyncProfile,
            ConcurrentProfile,
            SynchronousProfile,
            ThroughputProfile,
            SweepProfile,
        ],
        Field(discriminator="type_"),
    ] = Field(description="Benchmarking profile configuration")
    requests: dict[str, Any] = Field(
        default_factory=dict, description="Request loader configuration"
    )
    backend: dict[str, Any] = Field(
        default_factory=dict, description="Backend configuration"
    )
    environment: dict[str, Any] = Field(
        default_factory=dict, description="Environment configuration"
    )
    aggregators: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Aggregator configurations"
    )


class BenchmarkMetrics(StandardBaseModel):
    """Core benchmark metrics and statistical distributions."""

    requests_per_second: StatusDistributionSummary = Field(
        description="Distribution of requests per second across benchmark execution"
    )
    request_concurrency: StatusDistributionSummary = Field(
        description="Distribution of concurrent request counts during execution"
    )
    request_latency: StatusDistributionSummary = Field(
        description="Distribution of request latencies for completed requests"
    )


BenchmarkMetricsT = TypeVar("BenchmarkMetricsT", bound=BenchmarkMetrics)


class BenchmarkRequestStats(StandardBaseModel):
    """Individual request processing statistics and scheduling metadata."""

    scheduler_info: ScheduledRequestInfo[GenerationRequestTimings] = Field(
        description="Scheduler metadata and timing information for the request"
    )


BenchmarkRequestStatsT = TypeVar("BenchmarkRequestStatsT", bound=BenchmarkRequestStats)


class Benchmark(StandardBaseModel, Generic[BenchmarkMetricsT, BenchmarkRequestStatsT]):
    """Base benchmark result container with execution metadata."""

    type_: Literal["benchmark"] = "benchmark"
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this benchmark execution",
    )
    run_id: str = Field(
        description="Identifier for the benchmarker run containing this benchmark"
    )
    run_index: int = Field(
        description="Sequential index of this benchmark within the benchmarker run"
    )
    scheduler: SchedulerDict = Field(
        description="Scheduler configuration and execution state"
    )
    benchmarker: BenchmarkerDict = Field(
        description="Benchmarker configuration and component settings"
    )
    env_args: dict[str, Any] = Field(
        description="Environment arguments and runtime configuration"
    )
    extras: dict[str, Any] = Field(
        description="Additional metadata and custom benchmark parameters"
    )
    run_stats: BenchmarkSchedulerStats = Field(
        description="Scheduler timing and performance statistics"
    )
    start_time: float = Field(
        default=-1.0, description="Unix timestamp when the first request was initiated"
    )
    end_time: float = Field(
        default=-1.0, description="Unix timestamp when the last request completed"
    )

    @computed_field  # type: ignore[misc]
    @property
    def duration(self) -> float:
        """
        Benchmark execution duration in seconds.

        :return: Time elapsed from first request start to last request completion.
        """
        return self.end_time - self.start_time

    metrics: BenchmarkMetricsT = Field(
        description="Performance metrics and statistical distributions"
    )
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )
    requests: StatusBreakdown[
        list[BenchmarkRequestStatsT],
        list[BenchmarkRequestStatsT],
        list[BenchmarkRequestStatsT],
        None,
    ] = Field(
        description="Request details grouped by status: successful, incomplete, errored"
    )


BenchmarkT = TypeVar("BenchmarkT", bound=Benchmark)


class GenerativeRequestStats(BenchmarkRequestStats):
    """Request statistics for generative AI text generation workloads."""

    type_: Literal["generative_request_stats"] = "generative_request_stats"
    request_id: str = Field(description="Unique identifier for the request")
    request_type: Literal["text_completions", "chat_completions"] = Field(
        description="Type of generative request: text or chat completion"
    )
    prompt: str = Field(description="Input text prompt for generation")
    request_args: dict[str, Any] = Field(
        description="Generation parameters and configuration options"
    )
    output: Optional[str] = Field(
        description="Generated text output, if request completed successfully"
    )
    iterations: int = Field(
        description="Number of processing iterations for the request"
    )
    prompt_tokens: Optional[int] = Field(
        description="Number of tokens in the input prompt"
    )
    output_tokens: Optional[int] = Field(
        description="Number of tokens in the generated output"
    )

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> Optional[int]:
        """
        Total token count including prompt and output tokens.

        :return: Sum of prompt and output tokens, or None if either is unavailable.
        """
        if self.prompt_tokens is None or self.output_tokens is None:
            return None

        return self.prompt_tokens + self.output_tokens

    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> Optional[float]:
        """
        End-to-end request processing latency in seconds.

        :return: Duration from request start to completion, or None if unavailable.
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
        Time to first token generation in milliseconds.

        :return: Latency from request start to first token, or None if unavailable.
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
        Average time per output token in milliseconds.

        Includes time for first token and all subsequent tokens.

        :return: Average milliseconds per output token, or None if unavailable.
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
        Average inter-token latency in milliseconds.

        Measures time between token generations, excluding first token.

        :return: Average milliseconds between tokens, or None if unavailable.
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
        Overall token throughput including prompt and output tokens.

        :return: Total tokens per second, or None if unavailable.
        """
        if not (latency := self.request_latency) or not (tokens := self.total_tokens):
            return None

        return tokens / latency

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> Optional[float]:
        """
        Output token generation throughput.

        :return: Output tokens per second, or None if unavailable.
        """
        if not (latency := self.request_latency) or not self.output_tokens:
            return None

        return self.output_tokens / latency


class GenerativeMetrics(BenchmarkMetrics):
    """Comprehensive metrics for generative AI benchmarks."""

    prompt_token_count: StatusDistributionSummary = Field(
        description="Distribution of prompt token counts by request status"
    )
    output_token_count: StatusDistributionSummary = Field(
        description="Distribution of output token counts by request status"
    )
    total_token_count: StatusDistributionSummary = Field(
        description="Distribution of total token counts by request status"
    )
    time_to_first_token_ms: StatusDistributionSummary = Field(
        description="Distribution of first token latencies in milliseconds"
    )
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description="Distribution of average time per output token in milliseconds"
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description="Distribution of inter-token latencies in milliseconds"
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of output token generation rates"
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of total token throughput including prompt and output"
    )


class GenerativeBenchmark(Benchmark[GenerativeMetrics, GenerativeRequestStats]):
    """Complete generative AI benchmark results with specialized metrics."""

    type_: Literal["generative_benchmark"] = "generative_benchmark"  # type: ignore[assignment]


class GenerativeBenchmarksReport(StandardBaseModel):
    """Container for multiple benchmark results with load/save functionality."""

    DEFAULT_FILE: ClassVar[str] = "benchmarks.json"

    @staticmethod
    def load_file(
        path: Union[str, Path], type_: Literal["json", "yaml"] | None = None
    ) -> "GenerativeBenchmarksReport":
        """
        Load a report from a file.

        :param path: The path to load the report from.
        :param type_: File type override, auto-detected from extension if None.
        :return: The loaded report.
        :raises ValueError: If file type is unsupported.
        """
        path = Path(path) if not isinstance(path, Path) else path

        if path.is_dir():
            path = path / GenerativeBenchmarksReport.DEFAULT_FILE

        path.parent.mkdir(parents=True, exist_ok=True)
        path_suffix = path.suffix.lower()[1:]

        with path.open("r") as file:
            if (type_ or path_suffix) == "json":
                model_dict = json.loads(file.read())
            elif (type_ or path_suffix) in ["yaml", "yml"]:
                model_dict = yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported file type: {type_} for {path}.")

        return GenerativeBenchmarksReport.model_validate(model_dict)

    benchmarks: list[GenerativeBenchmark] = Field(
        description="The list of completed benchmarks contained within the report.",
        default_factory=list,
    )

    def save_file(
        self, path: Union[str, Path], type_: Literal["json", "yaml"] | None = None
    ) -> Path:
        """
        Save the report to a file.

        :param path: The path to save the report to.
        :param type_: File type override, auto-detected from extension if None.
        :return: The path to the saved report.
        :raises ValueError: If file type is unsupported.
        """
        path = Path(path) if not isinstance(path, Path) else path

        if path.is_dir():
            path = path / GenerativeBenchmarksReport.DEFAULT_FILE

        path.parent.mkdir(parents=True, exist_ok=True)
        path_suffix = path.suffix.lower()[1:]
        model_dict = self.model_dump()

        if (type_ or path_suffix) == "json":
            save_str = json.dumps(model_dict)
        elif (type_ or path_suffix) in ["yaml", "yml"]:
            save_str = yaml.dump(model_dict)
        else:
            raise ValueError(f"Unsupported file type: {type_} for {path}.")

        with path.open("w") as file:
            file.write(save_str)

        return path
