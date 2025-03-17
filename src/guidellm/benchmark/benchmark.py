import random
import uuid
from typing import Any, Dict, List, Literal, Optional, TypeVar

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


class BenchmarkArgs(Serializable):
    """
    A serializable model representing the arguments used to specify a benchmark run
    and how data was collected for it.
    """

    profile: Profile = Field(
        description=(
            "The profile used for the entire benchmark run that the strategy for "
            "this benchmark was pulled from."
        )
    )
    strategy_index: int = Field(
        description=(
            "The index of the strategy in the profile that was used for this benchmark."
        )
    )
    strategy: SchedulingStrategy = Field(
        description="The scheduling strategy used to run this benchmark. "
    )
    max_number: Optional[int] = Field(
        description="The maximum number of requests to run for this benchmark, if any."
    )
    max_duration: Optional[float] = Field(
        description="The maximum duration in seconds to run this benchmark, if any."
    )
    warmup_number: Optional[int] = Field(
        description=(
            "The number of requests to run for the warmup phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )
    warmup_duration: Optional[float] = Field(
        description=(
            "The duration in seconds to run for the warmup phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )
    cooldown_number: Optional[int] = Field(
        description=(
            "The number of requests to run for the cooldown phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )
    cooldown_duration: Optional[float] = Field(
        description=(
            "The duration in seconds to run for the cooldown phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )


class BenchmarkRunStats(Serializable):
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

    total: int = Field(
        description=(
            "The total number of requests in the benchmark run, "
            "including warmup and cooldown."
        ),
    )
    total_completed: int = Field(
        description=(
            "The total number of completed requests in the benchmark run, "
            "including warmup and cooldown."
        ),
    )
    total_errored: int = Field(
        description=(
            "The total number of errored requests in the benchmark run, "
            "including warmup and cooldown."
        )
    )

    queued_time_avg: float = Field(
        description=(
            "The average time spent in the queue for requests in the benchmark run."
        )
    )
    scheduled_time_avg: float = Field(
        description=(
            "The average time spent in the scheduled state for requests in the "
            "benchmark run."
        )
    )
    worker_time_avg: float = Field(
        description=(
            "The average time spent running each request in the benchmark run."
        )
    )
    worker_delay_avg: float = Field(
        description=(
            "The average delay between when a request was targeted to start at "
            "and when it was started by the worker in the benchmark run."
        )
    )
    resolve_delay_avg: float = Field(
        description=(
            "The average delay between when a request was targeted to start at "
            "and when it was resolved/requested by the worker in the benchmark run."
        )
    )
    process_idle_time_avg: float = Field(
        description=(
            "The average time spent in the idle state for each process in the "
            "benchmark run where it wasn't actively running a request."
        )
    )


class Benchmark(Serializable):
    """
    The base serializable model representing a benchmark run and its results.
    Specific benchmarker implementations should extend this model to include
    additional information or metadata as needed.

    Note, requests_per_second and requests_concurrency are kept at this level
    and are expected to be populated by the subclass implementation to ensure
    the logic for Profiles can include more complicated logic for determining
    what rates and concurrency values to use for subsequent strategies.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the benchmark.",
    )
    run_id: str = Field(
        description=(
            "The unique identifier for the encompasing benchmark run that this "
            "benchmark was a part of."
        )
    )
    args: BenchmarkArgs = Field(
        description=(
            "The arguments used to specify how to run the benchmark and collect data."
        )
    )
    run_stats: BenchmarkRunStats = Field(
        description=(
            "The process statistics for the entire benchmark run across all requests."
        )
    )
    worker: Optional[Serializable] = Field(
        description=(
            "The description and specifics for the worker used to resolve requests "
            "for this benchmark."
        )
    )
    requests_loader: Optional[Serializable] = Field(
        description=(
            "The description and specifics for the request loader used to create "
            "requests for this benchmark."
        )
    )
    extras: Dict[str, Any] = Field(
        description=(
            "Any additional information or metadata that was passed for this benchmark."
        )
    )

    requests_per_second: StatusDistributionSummary = Field(
        description="The distribution of requests per second for the benchmark.",
    )
    requests_concurrency: StatusDistributionSummary = Field(
        description="The distribution of requests concurrency for the benchmark.",
    )


BENCH = TypeVar("BENCH", bound=Benchmark)


class GenerativeTextResponseStats(Serializable):
    """
    A serializable model representing the request values, response values, and
    statistics for a generative text response.
    """

    request_id: str = Field(
        description="The unique identifier for the request.",
    )
    request_type: Literal["text_completions", "chat_completions"] = Field(
        description="The type of request made to the generative backend."
    )
    prompt: str = Field(
        description="The text prompt used for the generative request.",
    )
    output: str = Field(
        description="The generated text output from the generative request.",
    )
    prompt_tokens: int = Field(
        description="The number of tokens in the prompt text.",
    )
    output_tokens: int = Field(
        description="The number of tokens in the generated output text.",
    )
    start_time: float = Field(
        description="The time the request started.",
    )
    end_time: float = Field(
        description="The time the request ended.",
    )
    first_token_time: float = Field(
        description="The time the first token was received.",
    )
    last_token_time: float = Field(
        description="The time the last token was received.",
    )

    @computed_field
    @property
    def request_latency(self) -> float:
        """
        :return: The duration of the request in seconds from the start to the end.
        """
        return self.end_time - self.start_time

    @computed_field
    @property
    def time_to_first_token_ms(self) -> float:
        """
        :return: The time in milliseconds from the start of the request to the first
            token received.
        """
        return 1000 * (self.first_token_time - self.start_time)

    @computed_field
    @property
    def inter_token_latency_ms(self) -> float:
        """
        :return: The average time in milliseconds between generating tokens in the
            output text. Note, does not include the time to generate the first token.
        """
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
        """
        :return: The average number of tokens generated per second in the output text.
            Note, does not include the time to generate the first token.
        """
        if (itl_ms := self.inter_token_latency_ms) == 0.0:
            return 0.0

        return 1000.0 / itl_ms


class GenerativeTextErrorStats(GenerativeTextResponseStats):
    """
    A serializable model representing the request values, response values, and
    statistics for a generative text response that errored.
    Extends and overrides the GenerativeTextResponseStats model to include the
    error message and optional properties given the error occurred.
    """

    error: str = Field(
        description=(
            "The error message for the error that occurred while making the request."
        )
    )
    output: Optional[str] = Field(
        default=None,
        description=(
            "The generated text output from the generative request, if any, "
            "before the error occurred."
        ),
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description=(
            "The number of tokens in the generated output text, if any, "
            "before the error occurred."
        ),
    )
    first_token_time: Optional[float] = Field(
        default=None,
        description=(
            "The time the first token was received, if any, before the error occurred."
        ),
    )
    last_token_time: Optional[float] = Field(
        default=None,
        description=(
            "The time the last token was received, if any, before the error occurred."
        ),
    )

    @computed_field
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        """
        :return: The time in milliseconds from the start of the request to the first
            token received. None if the first token was not received.
        """
        if self.first_token_time is None:
            return None

        return super().time_to_first_token_ms

    @computed_field
    @property
    def inter_token_latency_ms(self) -> Optional[float]:
        """
        :return: The average time in milliseconds between generating tokens in the
            output text. Note, does not include the time to generate the first token.
            None if there were no output_tokens or the first token was not received.
        """
        if (
            self.output_tokens is None
            or self.first_token_time is None
            or self.last_token_time is None
        ):
            return None

        return super().inter_token_latency_ms

    @computed_field
    @property
    def output_tokens_per_second(self) -> Optional[float]:
        """
        :return: The average number of tokens generated per second in the output text.
            Note, does not include the time to generate the first token. None if there
            were no output_tokens or the first token was not received.
        """
        if self.inter_token_latency_ms is None:
            return None

        return super().output_tokens_per_second


class GenerativeBenchmark(Benchmark):
    """
    A serializable model representing a benchmark run and its results for generative
    requests and responses. Includes the completed and errored requests, the start
    and end times for the benchmark, and the statistics for the requests and responses.
    """

    completed_total: int = Field(
        description=(
            "The total number of completed requests in the benchmark, "
            "excluding warmup and cooldown."
        )
    )
    completed_sampled_size: Optional[int] = Field(
        default=None,
        description=(
            "The number of completed requests that were randomly sampled for "
            "the benchmark. None if no sampling was applied."
        ),
    )
    completed_requests: List[GenerativeTextResponseStats] = Field(
        description="The list of completed requests.",
    )
    errored_total: int = Field(
        description=(
            "The total number of errored requests in the benchmark, "
            "excluding warmup and cooldown."
        )
    )
    errored_sampled_size: Optional[int] = Field(
        default=None,
        description=(
            "The number of errored requests that were randomly sampled for "
            "the benchmark. None if no sampling was applied."
        ),
    )
    errored_requests: List[GenerativeTextErrorStats] = Field(
        description="The list of errored requests.",
    )
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

    def create_sampled(
        self, sample_size: int, error_sample_size: Optional[int] = None
    ) -> "GenerativeBenchmark":
        """
        Create a new benchmark instance with a random sample of the completed and
        errored requests based on the given sample sizes. If the sample sizes are
        larger than the total number of requests, the sample sizes are capped at
        the total number of requests.

        :param sample_size: The number of completed requests to sample.
        :param error_sample_size: The number of errored requests to sample.
            If None, defaults to the sample_size.
        :return: A new benchmark instance with the sampled requests.
        :raises ValueError: If the sample sizes are negative or if the
            GenerativeBenchmark has already been sampled and the requested sample
            sizes are larger than the previously sampled sizes.
        """
        if error_sample_size is None:
            error_sample_size = sample_size

        if sample_size < 0:
            raise ValueError(f"Sample size must be non-negative, given {sample_size}")
        if error_sample_size < 0:
            raise ValueError(
                f"Error sample size must be non-negative, given {error_sample_size}"
            )

        if (
            self.completed_sampled_size is not None
            and sample_size > self.completed_sampled_size
        ):
            raise ValueError(
                "The benchmark's completed response have already been sampled with "
                f"size {self.completed_sampled_size} and cannot be resampled with "
                f"a larger size, given: {sample_size}"
            )
        if (
            self.errored_sampled_size is not None
            and error_sample_size > self.errored_sampled_size
        ):
            raise ValueError(
                "The benchmark's errored response have already been sampled with "
                f"size {self.errored_sampled_size} and cannot be resampled with "
                f"a larger size, given: {error_sample_size}"
            )

        sample_size = min(sample_size, len(self.completed_requests))
        error_sample_size = min(error_sample_size, len(self.errored_requests))

        sampled_instance = self.model_copy()
        sampled_instance.completed_sampled_size = sample_size
        sampled_instance.completed_requests = random.sample(
            self.completed_requests, sample_size
        )
        sampled_instance.errored_sampled_size = error_sample_size
        sampled_instance.errored_requests = random.sample(
            self.errored_requests, error_sample_size
        )

        return sampled_instance

    @staticmethod
    def from_stats(
        run_id: str,
        completed: List[GenerativeTextResponseStats],
        errored: List[GenerativeTextErrorStats],
        args: BenchmarkArgs,
        run_stats: BenchmarkRunStats,
        worker: Optional[Serializable],
        requests_loader: Optional[Serializable],
        extras: Optional[Dict[str, Any]],
    ) -> "GenerativeBenchmark":
        """
        Create a GenerativeBenchmark instance from the given statistics and metadata.
        Given the completed and errored requests, the benchmark will fill in the
        remaining statistics for the various metrics required for a benchmark.
        This is the preferred method for creating a GenerativeBenchmark instance
        to ensure all statistics are properly calculated and populated.

        :param run_id: The unique identifier for the benchmark run.
        :param completed: The list of completed requests.
        :param errored: The list of errored requests.
        :param args: The arguments used to specify how to run the benchmark
            and collect data.
        :param run_stats: The process statistics for the entire benchmark run across
            all requests.
        :param worker: The description and specifics for the worker used to resolve
            requests.
        :param requests_loader: The description and specifics for the request loader
            used to create requests.
        :param extras: Any additional information or metadata that was passed for
            this benchmark.
        :return: A GenerativeBenchmark instance with the given statistics and metadata
            populated and calculated
        """
        start_time = min(req.start_time for req in completed) if completed else 0.0

        return GenerativeBenchmark(
            run_id=run_id,
            args=args,
            run_stats=run_stats,
            worker=worker,
            requests_loader=requests_loader,
            extras=extras or {},
            completed_total=len(completed),
            completed_requests=completed,
            errored_total=len(errored),
            errored_requests=errored,
            start_time=start_time,
            end_time=max(req.end_time for req in completed) if completed else 0.0,
            requests_per_second=StatusDistributionSummary.from_timestamped_values_per_frequency(
                completed_values=(
                    [(start_time, 0.0)]  # start time to cover full time range
                    + [(req.end_time, 1.0) for req in completed]
                ),
                errored_values=(
                    [(start_time, 0.0)]  # start time to cover full time range
                    + [(req.end_time, 1.0) for req in errored if req.end_time]
                ),
                frequency=1.0,  # 1 second
            ),
            requests_concurrency=StatusDistributionSummary.from_timestamped_interval_values(
                completed_values=(
                    [(req.start_time, req.end_time, 1) for req in completed]
                ),
                errored_values=([(req.start_time, req.end_time, 1) for req in errored]),
            ),
            requests_latency=StatusDistributionSummary.from_values(
                completed_values=[req.request_latency for req in completed],
                errored_values=[req.request_latency for req in errored],
            ),
            prompts_token_count=StatusDistributionSummary.from_values(
                completed_values=[req.prompt_tokens for req in completed],
                errored_values=[req.prompt_tokens for req in errored],
            ),
            outputs_token_count=StatusDistributionSummary.from_values(
                completed_values=[req.output_tokens for req in completed],
                errored_values=[req.output_tokens for req in errored],
            ),
            times_to_first_token_ms=StatusDistributionSummary.from_values(
                completed_values=[req.time_to_first_token_ms for req in completed],
                errored_values=[req.time_to_first_token_ms for req in errored],
            ),
            inter_token_latencies_ms=StatusDistributionSummary.from_values(
                completed_values=[
                    req.inter_token_latency_ms
                    for req in completed
                    for _ in range(req.output_tokens - 1)
                    if req.output_tokens > 1 and req.inter_token_latency_ms
                ],
                errored_values=[
                    req.inter_token_latency_ms
                    for req in errored
                    for _ in range(req.output_tokens - 1)
                    if req.output_tokens > 1 and req.inter_token_latency_ms
                ],
            ),
            outputs_tokens_per_second=StatusDistributionSummary.from_values(
                completed_values=[
                    req.output_tokens_per_second
                    for req in completed
                    for _ in range(req.output_tokens - 1)
                    if req.output_tokens > 1 and req.output_tokens_per_second
                ],
                errored_values=[
                    req.output_tokens_per_second
                    for req in errored
                    for _ in range(req.output_tokens - 1)
                    if req.output_tokens > 1 and req.output_tokens_per_second
                ],
            ),
        )
