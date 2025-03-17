from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Generic, List, TypeVar

from pydantic import Field

from guidellm.backend import ResponseSummary
from guidellm.benchmark.benchmark import BENCH, Benchmark, GenerativeBenchmark
from guidellm.objects import Serializable
from guidellm.request import GenerationRequest
from guidellm.scheduler import (
    REQ,
    RES,
    SchedulerResult,
)

__all__ = [
    "AGG",
    "BenchmarkAggregator",
    "GenerativeBenchmarkAggregator",
]


class BenchmarkAggregator(Generic[BENCH, REQ, RES], ABC, Serializable):
    created_requests: int = 0
    queued_requests: int = 0
    scheduled_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0
    successful_requests: int = 0
    errored_requests: int = 0

    queued_time: float = 0.0
    scheduled_time: float = 0.0
    worker_time: float = 0.0
    targeted_worker_start_delay: float = 0.0
    process_idle_time: DefaultDict[int, float] = defaultdict(float)
    process_idle_time_scratch: DefaultDict[int, float] = defaultdict(float)

    def add_base_result(
        self, result: SchedulerResult[REQ, RES], is_error: bool = False
    ):
        self.created_requests = result.run_info.created_requests
        self.queued_requests = result.run_info.queued_requests
        self.scheduled_requests = result.run_info.scheduled_requests
        self.processing_requests = result.run_info.processing_requests
        self.completed_requests = result.run_info.completed_requests

        if result.type_ != "request_complete":
            return

        if is_error:
            self.errored_requests += 1
        else:
            self.successful_requests += 1

        self.queued_time += (
            result.request_info.scheduled_time - result.request_info.queued_time
        )
        self.scheduled_time += (
            result.request_info.worker_start - result.request_info.scheduled_time
        )

        self.worker_time += (
            result.request_info.worker_end - result.request_info.worker_start
        )
        self.worker_schedule_delay_total += (
            result.request_info.worker_start - result.request_info.targeted_start_time
        )

        first_process_request = (
            result.request_info.process_id not in self.process_idle_time_scratch
        )
        if not first_process_request:
            self.process_idle_time_scratch[result.request_info.process_id] -= (
                result.request_info.worker_start
            )
            self.process_idle_time[result.request_info.process_id] = (
                self.process_idle_time_scratch[result.request_info.process_id]
            )
        self.process_idle_time_scratch[result.request_info.process_id] += (
            result.request_info.worker_end
        )

    def add_result(self, result: SchedulerResult[REQ, RES]):
        self.add_base_result(result)

    @abstractmethod
    def compile(self) -> Benchmark[BENCH]: ...


AGG = TypeVar("AGG", bound=BenchmarkAggregator)


class GenerativeBenchmarkAggregator(
    BenchmarkAggregator[GenerativeBenchmark, GenerationRequest, ResponseSummary]
):
    results: List[SchedulerResult[GenerationRequest, ResponseSummary]] = Field(
        default_factory=list,
        description="The list of results for the benchmark.",
    )

    request_time_total: float = 0.0
    targeted_request_delay_total: float = 0.0
    time_to_first_token_total: float = 0.0
    inter_token_latency_total: float = 0.0
    prompt_tokens_total: int = 0
    output_tokens_total: int = 0

    def add_result(self, result: SchedulerResult[GenerationRequest, ResponseSummary]):
        is_error = bool(result.response.error)
        self.add_base_result(result, is_error=is_error)

        if result.type_ != "request_complete":
            return

        self.results.append(result)

        if not is_error:
            self.request_time_total += (result.response.end_time or 0.0) - (
                result.response.start_time or 0.0
            )
            self.targeted_request_delay_total += (result.response.start_time or 0.0) - (
                result.request_info.targeted_start_time or 0.0
            )
            self.time_to_first_token_total += (
                result.response.first_iter_time or 0.0
            ) - (result.response.start_time or 0.0)
            self.inter_token_latency_total += (
                result.response.last_iter_time or 0.0
            ) - (result.response.first_iter_time or 0.0)
            self.prompt_tokens_total += result.response.prompt_tokens or 0
            self.output_tokens_total += result.response.output_tokens or 0

    def compile(self) -> GenerativeBenchmark:
        pass  # TODO
