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
from guidellm.objects import RunningStats, Serializable, TimeRunningStats
from guidellm.request import GenerationRequest
from guidellm.scheduler import (
    REQ,
    RES,
    SchedulerRequestResult,
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

    results: List[SchedulerRequestResult[GenerationRequest, ResponseSummary]] = Field(
        default_factory=list,
        description=(
            "The list of all results from the benchmark (complete, incomplete, error), "
            "that were not within the warmup or cooldown periods."
        ),
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

    scheduler_created_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests created for this "
            "benchmark run. This includes all requests created, regardless of "
            "their status."
        ),
        default_factory=RunningStats,
    )
    scheduler_queued_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests pending in queue "
            "for this benchmark run. This includes requests that are waiting to "
            "be scheduled."
        ),
        default_factory=RunningStats,
    )
    scheduler_scheduled_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests scheduled (actively "
            "running but waiting for the desired start time) for this benchmark run."
        ),
        default_factory=RunningStats,
    )
    scheduler_processing_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests actively being "
            "processed by the worker for this benchmark run."
        ),
        default_factory=RunningStats,
    )
    scheduler_completed_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests completed for this "
            "benchmark run. This includes requests within the warmup and cooldown "
            "period, if any, along with the final results."
        ),
        default_factory=RunningStats,
    )

    successful_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests that completed "
            "successfully without error. This is a subset of the completed requests "
            "for any that did not error. This includes requests within the warmup "
            "and cooldown period, if any, along with the final results."
        ),
        default_factory=RunningStats,
    )
    incomplete_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests that were incomplete "
            "or preempted during processing. This includes requests "
            "within the warmup and cooldown period, if any, along with the final "
            "results."
        ),
        default_factory=RunningStats,
    )
    errored_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests that errored during "
            "processing. This is a subset of the completed requests for any that "
            "errored. This includes requests within the warmup and cooldown period, "
            "if any, along with the final results."
        ),
        default_factory=RunningStats,
    )

    queued_time: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent in queue for all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was created to when it was dequeued by the worker."
        ),
        default_factory=TimeRunningStats,
    )
    scheduled_time_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent from when a request was "
            "dequeued by the worker to when it was actually scheduled by the worker"
            "for all requests that completed within the benchmark run. "
            "This should be as close to 0 as possible, any additional time is "
            "overheads from the system or the worker."
        ),
        default_factory=TimeRunningStats,
    )
    scheduled_time_sleep: TimeRunningStats = Field(
        description=(
            "The running statistics for the time for each request spent sleeping til "
            "the desired start time was reached for all requests that completed within "
            "the benchmark run. This is the time from when the request was scheduled "
            "to when the desired start time was reached. "
        ),
        default_factory=TimeRunningStats,
    )
    worker_start_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the time delay between when the request was "
            "scheduled and when the worker actually started processing subtracting any "
            "sleep time for all requests that completed within the benchmark run. "
            "This should be as close to 0 as possible, any additional time is "
            "overheads from the system or the worker."
        ),
        default_factory=TimeRunningStats,
    )
    worker_time: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent processing all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was started to when it was completed."
        ),
        default_factory=TimeRunningStats,
    )
    worker_start_time_targeted_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay between the targeted start time and "
            "the actual start time for requests that completed within the benchmark "
            "run. This represents delays from the best case desired start time. "
            "For async strategies, this represents delays from the ideal system. "
            "For sync strategies, since those are doubled in queue, this should be "
            "as close to the time for a request to be processed as possible."
        ),
        default_factory=TimeRunningStats,
    )
    request_start_time_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay between the actual request being "
            "made and the time the worker started on the request for all requests "
            "that completed within the benchmark run. This time should be as close to "
            "0 as possible, any additional time is overhead from the system or "
            "the worker."
        ),
        default_factory=TimeRunningStats,
    )
    request_start_time_targeted_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay between the targeted start time and "
            "the actual start time for all requests that completed within the "
            "benchmark run. This represents delays from the best case desired start "
            "time. For async strategies, this represents delays from the ideal system. "
            "For sync strategies, since those are duplicated in queue, this should be "
            "as close to the time for a request to be processed."
        ),
        default_factory=TimeRunningStats,
    )
    request_time_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay in time between the total request "
            "time and the worker time. This should be as close to 0 as possible, any "
            "additional time is overhead from the system or the worker. "
        ),
        default_factory=TimeRunningStats,
    )
    request_time: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent processing all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was created to when it was completed."
        ),
        default_factory=TimeRunningStats,
    )

    def add_result(
        self,
        result: SchedulerRequestResult[REQ, RES],
    ) -> bool:
        """
        Add a result to the aggregator. This will update the internal statistics
        and add the result to the list of results if it is not within the warmup or
        cooldown period.

        :param result: The result to add to the aggregator.
        :return: True if the result was added, False if it was added because it
            did not fit within the warmup or cooldown period, was not requested,
            or is not finished
        """
        # Add base scheduler statistics to the aggregator
        self.scheduler_created_requests += max(0, result.run_info.created_requests)
        self.scheduler_queued_requests += max(0, result.run_info.queued_requests)
        self.scheduler_scheduled_requests += max(0, result.run_info.scheduled_requests)
        self.scheduler_processing_requests += max(
            0, result.run_info.processing_requests
        )
        self.scheduler_completed_requests += max(0, result.run_info.completed_requests)

        if result.type_ != "request_complete" or (
            result.request_info.canceled and not result.request_info.requested
        ):
            # If the result is not completed yet, don't add to the results
            # If the result was canceled and not started, ignore it
            return False

        # add base result statistics given this was not preempted and it's completed
        if result.request_info.completed:
            self.successful_requests += 1
        elif result.request_info.canceled:
            self.incomplete_requests += 1
        elif result.request_info.errored:
            self.errored_requests += 1
        else:
            raise ValueError(
                "Unexpected state: request_info must be either "
                "completed, canceled, or errored."
            )

        self.queued_time += (
            result.request_info.dequeued_time - result.request_info.queued_time
        )
        self.scheduled_time_delay += (
            result.request_info.scheduled_time - result.request_info.dequeued_time
        )
        sleep_time = max(
            0.0,
            result.request_info.targeted_start_time
            - result.request_info.scheduled_time,
        )
        self.scheduled_time_sleep += sleep_time
        time_to_worker_start = (
            result.request_info.worker_start - result.request_info.scheduled_time
        )
        self.worker_start_delay += time_to_worker_start - sleep_time
        self.worker_time += (
            result.request_info.worker_end - result.request_info.worker_start
        )
        self.worker_start_time_targeted_delay += (
            result.request_info.worker_start - result.request_info.targeted_start_time
        )

        # Add result to the list of results provided we are not in warmup or cooldown
        total_completed = (
            self.successful_requests.total
            + self.incomplete_requests.total
            + self.errored_requests.total
        )
        global_start_time = self.scheduler_created_requests.start_time

        if (self.warmup_number and total_completed <= self.warmup_number) or (
            self.warmup_duration
            and result.request_info.worker_start
            <= (global_start_time + self.warmup_duration)
        ):
            # within warmup period
            self.in_warmup = True
            return True

        if (
            self.cooldown_number
            and total_completed > self.max_number - self.cooldown_number
        ) or (
            self.cooldown_duration
            and result.request_info.worker_start
            >= global_start_time + self.max_duration - self.cooldown_duration
        ):
            # within cooldown period
            self.in_cooldown = True
            return True

        self.in_warmup = False
        self.in_cooldown = False
        self.results.append(result)

        return True

    @abstractmethod
    def compile(self) -> BENCH:
        """
        Compile the benchmark results and statistics into a Benchmark object.
        This is required to be implemented by subclasses to finalize the benchmark
        and return the compiled object.
        """
        ...


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
    processor_args: Optional[Dict[str, Any]] = Field(
        description=(
            "Additional arguments to pass to the tokenizer if it requires "
            "any specific configuration for loading or processing."
        ),
    )

    time_to_first_token: TimeRunningStats = Field(
        description=(
            "The running statistics for the time from the start of the request to the "
            "first token being generated for all requests that completed within the "
            "benchmark run."
        ),
        default_factory=TimeRunningStats,
    )
    inter_token_latency: TimeRunningStats = Field(
        description=(
            "The running statistics for the time between each token being generated "
            "for all requests that completed within the benchmark run."
        ),
        default_factory=TimeRunningStats,
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

    def add_result(
        self, result: SchedulerRequestResult[GenerationRequest, ResponseSummary]
    ) -> bool:
        """
        Add a result to the aggregator. This will update the internal statistics
        and add the result to the list of results if it is not within the warmup or
        cooldown period.

        :param result: The result to add to the aggregator.
        """
        if not super().add_result(result):
            return False

        self.request_start_time_delay += (
            result.response.start_time - result.request_info.worker_start
        )
        self.request_start_time_targeted_delay += (
            result.response.start_time - result.request_info.targeted_start_time
        )
        self.request_time_delay += (
            (result.response.start_time - result.request_info.worker_start)
            + result.request_info.worker_end
            - result.response.end_time
        )
        self.request_time += result.response.end_time - result.response.start_time

        self.time_to_first_token += (
            (result.response.first_iter_time - result.response.start_time) * 1000.0
            if result.response.first_iter_time
            else 0.0
        )
        self.inter_token_latency.update(
            (result.response.last_iter_time - result.response.first_iter_time) * 1000.0
            if result.response.last_iter_time and result.response.first_iter_time
            else 0.0,
            count=(result.response.output_tokens or 1) - 1,
        )
        self.prompt_tokens += result.response.prompt_tokens or 0
        self.output_tokens += result.response.output_tokens or 0
        self.total_tokens += (result.response.prompt_tokens or 0) + (
            result.response.output_tokens or 0
        )

        return True

    def compile(self) -> GenerativeBenchmark:
        """
        Compile the benchmark results and statistics into a GenerativeBenchmark object.
        This is required to be implemented by subclasses to finalize the benchmark
        and return the compiled object.
        """
        successful, incomplete, errored = self._compile_results()

        return GenerativeBenchmark.from_stats(
            run_id=self.run_id,
            successful=successful,
            incomplete=incomplete,
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
                start_time=self.scheduler_created_requests.start_time,
                end_time=time.time(),
                total_successful=self.successful_requests.total,
                total_incomplete=self.incomplete_requests.total,
                total_errored=self.errored_requests.total,
                queued_time_avg=self.queued_time.mean,
                scheduled_time_delay_avg=self.scheduled_time_delay.mean,
                scheduled_time_sleep_avg=self.scheduled_time_sleep.mean,
                worker_start_delay_avg=self.worker_start_delay.mean,
                worker_time_avg=self.worker_time.mean,
                worker_start_time_targeted_delay_avg=self.worker_start_time_targeted_delay.mean,
                request_start_time_delay_avg=self.request_start_time_delay.mean,
                request_start_time_targeted_delay_avg=self.request_start_time_targeted_delay.mean,
                request_time_delay_avg=self.request_time_delay.mean,
                request_time_avg=self.request_time.mean,
            ),
            worker=self.worker_description,
            requests_loader=self.request_loader_description,
            extras=self.extras,
        )

    def _compile_results(
        self,
    ) -> Tuple[
        List[GenerativeTextResponseStats],
        List[GenerativeTextErrorStats],
        List[GenerativeTextErrorStats],
    ]:
        successful: List[GenerativeTextResponseStats] = []
        incomplete: List[GenerativeTextErrorStats] = []
        error: List[GenerativeTextErrorStats] = []

        for result in self.results:
            prompt_tokens = self._compile_tokens_count(
                value=str(result.request.content),
                requests_tokens=result.response.request_prompt_tokens,
                response_tokens=result.response.response_prompt_tokens,
                preferred_tokens_source=settings.preferred_prompt_tokens_source,
                errored=result.request_info.errored,
            )
            output_tokens = self._compile_tokens_count(
                value=result.response.value,
                requests_tokens=result.response.request_output_tokens,
                response_tokens=result.response.response_output_tokens,
                preferred_tokens_source=settings.preferred_output_tokens_source,
                errored=result.request_info.errored,
            )

            if result.request_info.canceled:
                incomplete.append(
                    GenerativeTextErrorStats(
                        error=result.response.error,
                        request_id=result.request.request_id,
                        request_type=result.request.request_type,
                        scheduler_info=result.request_info,
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
            elif result.request_info.errored:
                error.append(
                    GenerativeTextErrorStats(
                        error=result.response.error,
                        request_id=result.request.request_id,
                        request_type=result.request.request_type,
                        scheduler_info=result.request_info,
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
                successful.append(
                    GenerativeTextResponseStats(
                        request_id=result.request.request_id,
                        request_type=result.request.request_type,
                        scheduler_info=result.request_info,
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

        return successful, incomplete, error

    def _compile_tokens_count(
        self,
        value: str,
        requests_tokens: Optional[int],
        response_tokens: Optional[int],
        preferred_tokens_source: Optional[Literal["request", "response", "local"]],
        errored: bool,
    ) -> int:
        if not errored and preferred_tokens_source == "response" and response_tokens:
            return response_tokens or 0

        if not errored and preferred_tokens_source == "request" and requests_tokens:
            return requests_tokens or 0

        if preferred_tokens_source in {"response", "request"} and (
            self.processor is None or errored or response_tokens or requests_tokens
        ):
            # we had a preferred tokens source that isn't local and we either
            # have the data to return something or we don't have the ability
            # to calculate locally
            return response_tokens or requests_tokens or 0

        self.processor = check_load_processor(
            self.processor,
            processor_args=self.processor_args,
            error_msg="Processor/Tokenizer is required for calculating token counts.",
        )
        return len(self.processor.tokenize(value))
