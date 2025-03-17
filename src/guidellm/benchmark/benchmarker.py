import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generic, Iterable, Literal, Optional

from guidellm.backend import Backend, ResponseSummary
from guidellm.benchmark.aggregator import AGG, BENCH, GenerativeBenchmarkAggregator
from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.benchmark.profile import Profile
from guidellm.objects import Serializable
from guidellm.request import GenerationRequest
from guidellm.scheduler import (
    REQ,
    RES,
    GenerativeRequestsWorker,
    RequestsWorker,
    Scheduler,
    SchedulerResult,
    SchedulingStrategy,
)

__all__ = ["Benchmarker", "BenchmarkerResult", "GenerativeBenchmarker"]


class BenchmarkerResult(Generic[AGG, BENCH, REQ, RES], Serializable):
    type_: Literal[
        "run_start",
        "run_complete",
        "scheduler_start",
        "scheduler_update",
        "scheduler_complete",
        "benchmark_compiled",
    ]
    start_time: float
    end_number: int
    profile: Profile
    current_index: int
    current_strategy: Optional[SchedulingStrategy] = None
    current_aggregator: Optional[AGG[BENCH, REQ, RES]] = None
    current_benchmark: Optional[BENCH] = None
    current_result: Optional[SchedulerResult[REQ, RES]] = None


class Benchmarker(Generic[AGG, BENCH, REQ, RES], ABC):
    def __init__(
        self,
        worker: RequestsWorker[REQ, RES],
        request_loader: Iterable[REQ],
        requests_loader_description: Optional[Serializable] = None,
        benchmark_save_extras: Optional[Dict[str, Any]] = None,
    ):
        self.scheduler: Scheduler[REQ, RES] = Scheduler(
            worker=worker, request_loader=request_loader
        )
        self.requests_loader_description = requests_loader_description
        self.benchmark_save_extras = benchmark_save_extras

    async def run(
        self,
        profile: Profile,
        max_number_per_strategy: Optional[int],
        max_duration_per_strategy: Optional[float],
        warmup_number_per_strategy: Optional[float],
        warmup_duration_per_strategy: Optional[float],
        cooldown_number_per_strategy: Optional[int],
        cooldown_duration_per_strategy: Optional[float],
    ) -> AsyncGenerator[BenchmarkerResult[AGG, BENCH, REQ, RES], None]:
        start_time = time.time()
        end_number = len(profile.strategy_types)
        current_index = -1

        yield BenchmarkerResult(
            type_="run_start",
            start_time=start_time,
            end_number=end_number,
            profile=profile,
            current_index=current_index,
            current_strategy=None,
            current_aggregator=None,
            current_benchmark=None,
            current_result=None,
        )

        while scheduling_strategy := profile.next_strategy():
            current_index += 1
            aggregator: AGG[BENCH, REQ, RES] = self.create_benchmark_aggregator(
                profile=profile,
                current_index=current_index,
                strategy=scheduling_strategy,
                max_number=max_number_per_strategy,
                max_duration=max_duration_per_strategy,
                warmup_number=warmup_number_per_strategy,
                warmup_duration=warmup_duration_per_strategy,
                cooldown_number=cooldown_number_per_strategy,
                cooldown_duration=cooldown_duration_per_strategy,
            )

            yield BenchmarkerResult(
                type_="scheduler_start",
                start_time=start_time,
                end_number=end_number,
                profile=profile,
                current_index=current_index,
                current_strategy=scheduling_strategy,
                current_aggregator=aggregator,
                current_benchmark=None,
                current_result=None,
            )

            async for result in self.scheduler.run(
                scheduling_strategy=scheduling_strategy,
                max_number=max_number_per_strategy,
                max_duration=max_duration_per_strategy,
            ):
                aggregator.add_result(result)

                yield BenchmarkerResult(
                    type_="scheduler_update",
                    start_time=start_time,
                    end_number=end_number,
                    profile=profile,
                    current_index=current_index,
                    current_strategy=scheduling_strategy,
                    current_aggregator=aggregator,
                    current_benchmark=None,
                    current_result=result,
                )

            yield BenchmarkerResult(
                type_="scheduler_complete",
                start_time=start_time,
                end_number=end_number,
                profile=profile,
                current_index=current_index,
                current_strategy=scheduling_strategy,
                current_aggregator=aggregator,
                current_benchmark=None,
                current_result=None,
            )

            benchmark: BENCH = aggregator.compile()
            profile.completed_strategy(
                average_rate=benchmark.requests_per_second.completed.mean,
                average_concurrency=benchmark.requests_concurrency.completed.mean,
            )

            yield BenchmarkerResult(
                type_="benchmark_compiled",
                start_time=start_time,
                end_number=end_number,
                profile=profile,
                current_index=current_index,
                current_strategy=scheduling_strategy,
                current_aggregator=None,
                current_benchmark=benchmark,
                current_result=None,
            )

        yield BenchmarkerResult(
            type_="run_complete",
            start_time=start_time,
            end_number=end_number,
            profile=profile,
            current_index=current_index,
            current_strategy=None,
            current_aggregator=None,
            current_benchmark=None,
            current_result=None,
        )

    @abstractmethod
    def create_benchmark_aggregator(
        self,
        profile: Profile,
        current_index: int,
        strategy: SchedulingStrategy,
        max_number: Optional[int],
        max_duration: Optional[float],
        warmup_number: Optional[float],
        warmup_duration: Optional[float],
        cooldown_number: Optional[int],
        cooldown_duration: Optional[float],
    ) -> AGG[BENCH, REQ, RES]: ...


class GenerativeBenchmarker(
    Benchmarker[
        GenerativeBenchmarkAggregator,
        GenerativeBenchmark,
        GenerationRequest,
        ResponseSummary,
    ],
):
    def __init__(
        self,
        backend: Backend,
        request_loader: Iterable[GenerationRequest],
    ):
        super().__init__(
            worker=GenerativeRequestsWorker(backend), request_loader=request_loader
        )

    def create_benchmark_aggregator(
        self,
        profile: Profile,
        current_index: int,
        strategy: SchedulingStrategy,
        max_number: Optional[int],
        max_duration: Optional[float],
        warmup_number: Optional[float],
        warmup_duration: Optional[float],
        cooldown_number: Optional[int],
        cooldown_duration: Optional[float],
    ) -> GenerativeBenchmarkAggregator:
        return GenerativeBenchmarkAggregator()  # TODO
