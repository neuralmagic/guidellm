import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field

from guidellm.backend import ResponseSummary
from guidellm.benchmark.benchmark import (
    BENCH,
    BenchmarkArgs,
    BenchmarkRunStats,
    GenerativeBenchmark,
    GenerativeTextErrorStats,
    GenerativeTextResponseStats,
)
from guidellm.benchmark.profile import Profile
from guidellm.config import settings
from guidellm.objects import RunningStats, Serializable
from guidellm.request import GenerationRequest
from guidellm.scheduler import (
    REQ,
    RES,
    SchedulerResult,
    SchedulingStrategy,
)
from guidellm.utils import check_load_processor

__all__ = [
    "AGG",
    "BenchmarkAggregator",
    "GenerativeBenchmarkAggregator",
]


class BenchmarkAggregator(ABC, BaseModel, Generic[BENCH, REQ, RES]):
    """
    A pydantic base class representing the base class for aggregating benchmark results.
    The purpose is to receive and process results from a Benchmarker as it iterates
    through a Scheduler for an individual benchmark run.
    As results are added, lightweight statistics are updated and stored for immediate
    progress and informational updates to the caller.
    Once the benchmark run is complete, the `compile` method is called to finalize
    the benchmark and return a Benchmark object with all the results and statistics
    fully calculated.
    """

    run_id: str = Field(
        description=(
            "The unique identifier for the encompasing benchmark run that this "
            "benchmark was a part of."
        )
    )
    profile: Profile = Field(
        description=(
            "The profile used for the entire benchamrk run that the strategy for "
            "the active benchmark was pulled from."
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
    worker_description: Optional[Serializable] = Field(
        description=(
            "The description and specifics for the worker used to resolve requests "
            "for this benchmark."
        )
    )
    request_loader_description: Optional[Serializable] = Field(
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

    results: List[SchedulerResult[GenerationRequest, ResponseSummary]] = Field(
        default_factory=list,
        description=(
            "The list of results from the benchmark, both completed and errored, "
            "that were not within the warmup or cooldown periods."
        ),
    )

    start_time: float = Field(
        description=(
            "The timestamp when the benchmark run started. Defaults to the current "
            "time.time() on creation."
        ),
        default_factory=time.time,
    )
    created_requests: int = Field(
        description="The number of requests created for this benchmark run.",
        default=0,
    )
    queued_requests: int = Field(
        description="The number of requests pending in queue for this benchmark run.",
        default=0,
    )
    scheduled_requests: int = Field(
        description=(
            "The number of requests scheduled (actively running but waiting for the "
            "desired start time) for this benchmark run."
        ),
        default=0,
    )
    processing_requests: int = Field(
        description=(
            "The number of requests actively being processed by the worker for this "
            "benchmark run."
        ),
        default=0,
    )
    completed_requests: int = Field(
        description=(
            "The number of requests completed for this benchmark run. This includes "
            "requests within the warmup and cooldown period, if any, along with the "
            "final results."
        ),
        default=0,
    )
    successful_requests: int = Field(
        description=(
            "The number of requests that completed successfully without error. "
            "This is a subset of the completed requests for any that did not error. "
            "This includes requests within the warmup and cooldown period, if any, "
            "along with the final results."
        ),
        default=0,
    )
    errored_requests: int = Field(
        description=(
            "The number of requests that errored during processing. This is a subset "
            "of the completed requests for any that errored. This includes requests "
            "within the warmup and cooldown period, if any, "
            "along with the final results."
        ),
        default=0,
    )
    in_warmup: bool = Field(
        description=(
            "A flag to indicate if the benchmark is currently in the warmup phase."
        ),
        default=False,
        exclude=True,
    )
    in_cooldown: bool = Field(
        description=(
            "A flag to indicate if the benchmark is currently in the cooldown phase."
        ),
        default=False,
        exclude=True,
    )

    queued_time: RunningStats = Field(
        description=(
            "The running statistics for the time spent in queue for all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was created to when it was scheduled to be processed."
        ),
        default_factory=RunningStats,
    )
    scheduled_time: RunningStats = Field(
        description=(
            "The running statistics for the time spent scheduled for all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was scheduled to be processed to when it was actually started."
        ),
        default_factory=RunningStats,
    )
    worker_time: RunningStats = Field(
        description=(
            "The running statistics for the time spent processing for all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was started to when it was completed."
        ),
        default_factory=RunningStats,
    )
    targeted_worker_start_delay: RunningStats = Field(
        description=(
            "The running statistics for the delay between the targeted start time and "
            "the actual start time for all requests that completed within the benchmark "
            "run. This is the time from when the request was scheduled to be processed "
            "to when it was actually started."
        ),
        default_factory=RunningStats,
    )

    def add_result(self, result: SchedulerResult[REQ, RES]):
        """
        Add a result to the aggregator. This will update the internal statistics
        and add the result to the list of results if it is not within the warmup or
        cooldown period.

        :param result: The result to add to the aggregator.
        """
        self.add_base_result(result)

    @abstractmethod
    def compile(self) -> BENCH:
        """
        Compile the benchmark results and statistics into a Benchmark object.
        This is required to be implemented by subclasses to finalize the benchmark
        and return the compiled object.
        """
        ...

    def add_base_result(
        self, result: SchedulerResult[REQ, RES], is_error: bool = False
    ):
        """
        Helper function to update the base statistics for the aggregator and add the
        result to the list of results if it is not within the warmup or cooldown period.

        :param result: The result to add to the aggregator.
        :param is_error: A flag to indicate if the result was an error or not.
        """
        self.created_requests = result.run_info.created_requests
        self.queued_requests = result.run_info.queued_requests
        self.scheduled_requests = result.run_info.scheduled_requests
        self.processing_requests = result.run_info.processing_requests
        self.completed_requests = result.run_info.completed_requests

        if result.type_ == "request_complete":
            self._update_stats_from_result(result, is_error)
            self._add_to_results_within_active_period(result)

    def _update_stats_from_result(
        self, result: SchedulerResult[REQ, RES], is_error: bool
    ):
        if is_error:
            self.errored_requests += 1
        else:
            self.successful_requests += 1

        self.queued_time.update(
            result.request_info.scheduled_time - result.request_info.queued_time
        )
        self.scheduled_time.update(
            result.request_info.worker_start - result.request_info.scheduled_time
        )
        self.worker_time.update(
            result.request_info.worker_end - result.request_info.worker_start
        )
        self.targeted_worker_start_delay.update(
            result.request_info.worker_start - result.request_info.targeted_start_time
        )

    def _add_to_results_within_active_period(self, result: SchedulerResult[REQ, RES]):
        start_time = result.request_info.worker_start
        end_time = result.request_info.worker_end
        completed_number = self.errored_requests + self.successful_requests

        if (self.warmup_number and completed_number <= self.warmup_number) or (
            self.warmup_duration and start_time <= self.warmup_duration
        ):
            # within warmup period
            self.in_warmup = True
            return

        if (
            self.cooldown_number
            and completed_number > self.max_number - self.cooldown_number
        ) or (
            self.cooldown_duration
            and end_time >= self.max_duration - self.cooldown_duration
        ):
            # within cooldown period
            self.in_cooldown = True
            return

        self.in_warmup = False
        self.in_cooldown = False
        self.results.append(result)


AGG = TypeVar("AGG", bound=BenchmarkAggregator[BENCH, REQ, RES])


class GenerativeBenchmarkAggregator(
    BenchmarkAggregator[GenerativeBenchmark, GenerationRequest, ResponseSummary]
):
    processor: Optional[Union[str, Path, Any]] = Field(
        description=(
            "The tokenizer to use for calculating token counts when none are "
            "avaiable that match the preferred source."
        )
    )

    targeted_request_delay: RunningStats = Field(
        description=(
            "The running statistics for the delay between the targeted start time and "
            "the actual start time for all requests that completed within the "
            "benchmark run. This is the time from when the request was scheduled to "
            "be processed to when it was actually started."
        ),
        default_factory=RunningStats,
    )
    request_latency: RunningStats = Field(
        description=(
            "The running statistics for the time spent processing all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was created to when it was completed."
        ),
        default_factory=RunningStats,
    )
    time_to_first_token: RunningStats = Field(
        description=(
            "The running statistics for the time from the start of the request to the "
            "first token being generated for all requests that completed within the "
            "benchmark run."
        ),
        default_factory=RunningStats,
    )
    inter_token_latency: RunningStats = Field(
        description=(
            "The running statistics for the time between each token being generated "
            "for all requests that completed within the benchmark run."
        ),
        default_factory=RunningStats,
    )
    prompt_tokens: RunningStats = Field(
        description=(
            "The running statistics for the token count for the prompt for all "
            "requests that completed, if available in the response."
        ),
        default_factory=RunningStats,
    )
    output_tokens: RunningStats = Field(
        description=(
            "The running statistics for the token count for the output for all "
            "requests that completed, if available in the response."
        ),
        default_factory=RunningStats,
    )
    total_tokens: RunningStats = Field(
        description=(
            "The running statistics for the total token count for all requests that "
            "completed, if available in the response."
        ),
        default_factory=RunningStats,
    )

    def add_result(self, result: SchedulerResult[GenerationRequest, ResponseSummary]):
        """
        Add a result to the aggregator. This will update the internal statistics
        and add the result to the list of results if it is not within the warmup or
        cooldown period.

        :param result: The result to add to the aggregator.
        """
        is_error = result.type_ == "request_complete" and result.response.error
        self.add_base_result(result, is_error=is_error)

        if result.type_ == "request_complete":
            self._update_generative_stats_from_result(result)

    def compile(self) -> GenerativeBenchmark:
        """
        Compile the benchmark results and statistics into a GenerativeBenchmark object.
        This is required to be implemented by subclasses to finalize the benchmark
        and return the compiled object.
        """
        completed, errored = self._compile_results()

        return GenerativeBenchmark.from_stats(
            run_id=self.run_id,
            completed=completed,
            errored=errored,
            args=BenchmarkArgs(
                profile=self.profile,
                strategy_index=self.strategy_index,
                strategy=self.strategy,
                max_number=self.max_number,
                max_duration=self.max_duration,
                warmup_number=self.warmup_number,
                warmup_duration=self.warmup_duration,
                cooldown_number=self.cooldown_number,
                cooldown_duration=self.cooldown_duration,
            ),
            run_stats=BenchmarkRunStats(
                start_time=self.start_time,
                end_time=time.time(),
                total=self.completed_requests,
                total_completed=self.successful_requests,
                total_errored=self.errored_requests,
                queued_time_avg=self.queued_time.mean,
                scheduled_time_avg=self.scheduled_time.mean,
                worker_time_avg=self.worker_time.mean,
                worker_delay_avg=self.targeted_worker_start_delay.mean,
                resolve_delay_avg=self.targeted_request_delay.mean,
            ),
            worker=self.worker_description,
            requests_loader=self.request_loader_description,
            extras=self.extras,
        )

    def _update_generative_stats_from_result(
        self, result: SchedulerResult[GenerationRequest, ResponseSummary]
    ):
        if self.request_latency.count == 0:
            self.request_latency.start_time = self.start_time
            self.targeted_request_delay.start_time = self.start_time
            self.time_to_first_token.start_time = self.start_time
            self.inter_token_latency.start_time = self.start_time
            self.prompt_tokens.start_time = self.start_time
            self.output_tokens.start_time = self.start_time
            self.total_tokens.start_time = self.start_time

        self.request_latency.update(
            result.response.end_time - result.response.start_time
            if result.response.end_time and result.response.start_time
            else 0.0
        )
        self.targeted_request_delay.update(
            result.response.start_time - result.request_info.targeted_start_time
            if result.response.start_time
            else 0.0
        )
        self.time_to_first_token.update(
            (result.response.first_iter_time - result.response.start_time) * 1000.0
            if result.response.first_iter_time and result.response.start_time
            else 0.0
        )
        if result.response.output_tokens > 1:
            self.inter_token_latency.update(
                (result.response.last_iter_time - result.response.first_iter_time)
                * 1000.0,
                count=result.response.output_tokens - 1,
            )
        self.prompt_tokens.update(
            result.response.prompt_tokens or 0,
        )
        self.output_tokens.update(
            result.response.output_tokens or 0,
        )
        self.total_tokens.update(
            (result.response.prompt_tokens or 0) + (result.response.output_tokens or 0),
        )

    def _compile_results(
        self,
    ) -> Tuple[List[Union[GenerativeTextResponseStats, GenerativeTextErrorStats]]]:
        completed: List[GenerativeTextResponseStats] = []
        errored: List[GenerativeTextErrorStats] = []

        for result in self.results:
            prompt_tokens = self._compile_tokens_count(
                value=str(result.request.content),
                requests_tokens=result.response.request_prompt_tokens,
                response_tokens=result.response.response_prompt_tokens,
                preferred_tokens_source=settings.preferred_prompt_tokens_source,
            )
            output_tokens = self._compile_tokens_count(
                value=result.response.value,
                requests_tokens=result.response.request_output_tokens,
                response_tokens=result.response.response_output_tokens,
                preferred_tokens_source=settings.preferred_output_tokens_source,
            )

            if result.response.error:
                errored.append(
                    GenerativeTextErrorStats(
                        error=result.response.error,
                        request_id=result.request.request_id,
                        request_type=result.request.request_type,
                        prompt=str(result.request.content),
                        prompt_tokens=prompt_tokens,
                        output=result.response.value,
                        output_tokens=output_tokens,
                        start_time=result.response.start_time,
                        end_time=result.response.end_time,
                        first_token_time=result.response.first_iter_time,
                        last_token_time=result.response.last_iter_time,
                    )
                )
            else:
                completed.append(
                    GenerativeTextResponseStats(
                        request_id=result.request.request_id,
                        request_type=result.request.request_type,
                        prompt=str(result.request.content),
                        prompt_tokens=prompt_tokens,
                        output=result.response.value,
                        output_tokens=output_tokens,
                        start_time=result.response.start_time,
                        end_time=result.response.end_time,
                        first_token_time=result.response.first_iter_time,
                        last_token_time=result.response.last_iter_time,
                    )
                )

        return completed, errored

    def _compile_tokens_count(
        self,
        value: str,
        requests_tokens: Optional[int],
        response_tokens: Optional[int],
        preferred_tokens_source: Optional[Literal["request", "response"]],
    ) -> int:
        if preferred_tokens_source is None and (requests_tokens or response_tokens):
            return (
                response_tokens or requests_tokens
            )  # trust response first if no preference
        elif preferred_tokens_source == "response" and response_tokens:
            return response_tokens
        elif preferred_tokens_source == "request" and requests_tokens:
            return requests_tokens
        elif self.processor is None:
            # no processor available, fall back on unpreferred source or 0
            return response_tokens or requests_tokens or 0

        self.processor = check_load_processor(
            self.processor,
            error_msg="Processor/Tokenizer is required for calculating token counts.",
        )
        # no tokens that matched the preferred source,
        # calculate locally based on the value
        return len(self.processor.tokenize(value))
