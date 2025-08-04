import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Union,
)

from transformers import PreTrainedTokenizerBase  # type: ignore  # noqa: PGH003

from guidellm.backend import Backend
from guidellm.benchmark.aggregator import (
    AggregatorT,
    BenchmarkT,
    GenerativeBenchmarkAggregator,
)
from guidellm.benchmark.benchmark import BenchmarkArgs, GenerativeBenchmark
from guidellm.benchmark.profile import Profile
from guidellm.request import (
    GenerativeRequestLoaderDescription,
)
from guidellm.scheduler import (
    BackendT,
    Environment,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
    SchedulingStrategy,
)
from guidellm.utils import ThreadSafeSingletonMixin

__all__ = ["Benchmarker", "GenerativeBenchmarker"]


"""
Scheduler:
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        env: Environment,
        **constraints: dict[
            str, Union[int, float, str, ConstraintsResolveArgs, CallableConstraint]
        ],

CallableConstraint = Callable[
    [SchedulerState, ScheduledRequestInfo], SchedulerUpdateAction
]
"""


CallableConstraintInitializer = Callable[
    [AggregatorT, BenchmarkT],
    Callable[[SchedulerState, ScheduledRequestInfo], SchedulerUpdateAction],
]


class Benchmarker(
    Generic[AggregatorT, BenchmarkT, RequestT, ResponseT], ABC, ThreadSafeSingletonMixin
):
    async def run(
        self,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, ResponseT],
        profile: Profile,
        environment: Environment,
        aggregator: type[AggregatorT],
    ) -> AsyncGenerator[
        BenchmarkerResult[AggregatorT, BenchmarkT, RequestT, ResponseT], None
    ]:
        try:
            requests_loader_size = len(self.scheduler.request_loader)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            requests_loader_size = None

        strategy_limits = BenchmarkerStrategyLimits(
            requests_loader_size=requests_loader_size,
            max_number_per_strategy=max_number_per_strategy,
            max_duration_per_strategy=max_duration_per_strategy,
            warmup_percent_per_strategy=warmup_percent_per_strategy,
            cooldown_percent_per_strategy=cooldown_percent_per_strategy,
        )
        start_time = time.time()
        end_number = len(profile.strategy_types)
        current_index = -1
        run_id = str(uuid.uuid4())

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
            aggregator = self.create_benchmark_aggregator(
                run_id=run_id,
                profile=profile,
                strategy_index=current_index,
                strategy=scheduling_strategy,
                limits=strategy_limits,
            )

            async for result in self.scheduler.run(
                scheduling_strategy=scheduling_strategy,
                max_number=max_number_per_strategy,
                max_duration=max_duration_per_strategy,
            ):
                if result.type_ == "run_start":
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
                elif result.type_ == "run_complete":
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
                elif isinstance(result, SchedulerRequestResult):
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
                else:
                    raise ValueError(f"Unexpected result type: {type(result)}")

            benchmark: BenchmarkT = aggregator.compile()
            profile.completed_strategy(
                average_rate=benchmark.metrics.requests_per_second.successful.mean,
                average_concurrency=benchmark.metrics.request_concurrency.successful.mean,
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
        run_id: str,
        profile: Profile,
        strategy_index: int,
        strategy: SchedulingStrategy,
        limits: BenchmarkerStrategyLimits,
    ) -> AggregatorT: ...


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
        request_loader_description: GenerativeRequestLoaderDescription,
        benchmark_save_extras: Optional[dict[str, Any]] = None,
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]] = None,
        processor_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            worker=GenerativeRequestsWorker(backend),
            request_loader=request_loader,
            requests_loader_description=request_loader_description,
            benchmark_save_extras=benchmark_save_extras,
        )
        self.processor = processor
        self.processor_args = processor_args

    def create_benchmark_aggregator(
        self,
        run_id: str,
        profile: Profile,
        strategy_index: int,
        strategy: SchedulingStrategy,
        limits: BenchmarkerStrategyLimits,
    ) -> GenerativeBenchmarkAggregator:
        return GenerativeBenchmarkAggregator(
            run_id=run_id,
            args=BenchmarkArgs(
                profile=profile,
                strategy_index=strategy_index,
                strategy=strategy,
                max_number=limits.max_number,
                max_duration=limits.max_duration,
                warmup_number=limits.warmup_number,
                warmup_duration=limits.warmup_duration,
                cooldown_number=limits.cooldown_number,
                cooldown_duration=limits.cooldown_duration,
            ),
            worker_description=self.worker.description,  # type: ignore[arg-type]
            request_loader_description=self.requests_loader_description,  # type: ignore[arg-type]
            extras=self.benchmark_save_extras or {},
            processor=self.processor,
            processor_args=self.processor_args,
        )
