from abc import ABC
from collections.abc import AsyncIterator, Iterable
from typing import (
    Generic,
    Optional,
    Union,
)

from guidellm.benchmark.aggregator import (
    AggregatorT,
    BenchmarkT,
)
from guidellm.benchmark.profile import Profile
from guidellm.scheduler import (
    BackendT,
    Environment,
    RequestT,
    RequestTimingsT,
    ResponseT,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.utils import ThreadSafeSingletonMixin

__all__ = ["Benchmarker"]


class Benchmarker(
    Generic[AggregatorT, BenchmarkT, RequestT, RequestTimingsT, ResponseT],
    ABC,
    ThreadSafeSingletonMixin,
):
    async def run(
        self,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, ResponseT],
        profile: Profile,
        environment: Environment,
        aggregator_class: type[AggregatorT],
    ) -> AsyncIterator[
        tuple[
            Optional[BenchmarkT],
            AggregatorT,
            SchedulingStrategy,
            Optional[SchedulerState],
        ]
    ]:
        with self.thread_lock:
            strategies_generator = profile.strategies_generator()
            strategy, constraints = next(strategies_generator)

            while strategy is not None:
                aggregator = aggregator_class(
                    strategy=strategy, constraints=constraints
                )
                yield None, aggregator, strategy, None

                async for (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) in Scheduler[BackendT, RequestT, RequestTimingsT, ResponseT].run(
                    requests=requests,
                    backend=backend,
                    strategy=strategy,
                    env=environment,
                    **constraints,
                ):
                    aggregator.update(
                        response=response,
                        request=request,
                        request_info=request_info,
                    )
                    yield None, aggregator, strategy, scheduler_state

            benchmark = aggregator.compile()
            yield benchmark, aggregator, strategy, None
            strategy, constraints = strategies_generator.send((benchmark, aggregator))
