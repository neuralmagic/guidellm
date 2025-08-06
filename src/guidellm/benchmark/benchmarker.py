import uuid
from abc import ABC
from collections.abc import AsyncIterator, Iterable
from typing import (
    Any,
    Generic,
    Optional,
    Union,
)

from guidellm.benchmark.aggregator import Aggregator, CompilableAggregator
from guidellm.benchmark.benchmark import BenchmarkT
from guidellm.benchmark.profile import Profile
from guidellm.scheduler import (
    BackendT,
    Constraint,
    Environment,
    RequestT,
    RequestTimingsT,
    ResponseT,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.utils import InfoMixin, ThreadSafeSingletonMixin

__all__ = ["Benchmarker"]


class Benchmarker(
    Generic[BenchmarkT, RequestT, RequestTimingsT, ResponseT],
    ABC,
    ThreadSafeSingletonMixin,
):
    async def run(
        self,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, RequestTimingsT, ResponseT],
        profile: Profile,
        environment: Environment,
        benchmark_aggregators: dict[
            str,
            Union[
                Aggregator[ResponseT, RequestT, RequestTimingsT],
                CompilableAggregator[ResponseT, RequestT, RequestTimingsT],
            ],
        ],
        benchmark_class: type[BenchmarkT],
    ) -> AsyncIterator[
        tuple[
            dict[str, Any],
            Optional[BenchmarkT],
            SchedulingStrategy,
            Optional[SchedulerState],
        ]
    ]:
        with self.thread_lock:
            run_id = str(uuid.uuid4())
            strategies_generator = profile.strategies_generator()
            strategy, constraints = next(strategies_generator)

            while strategy is not None:
                yield {}, None, strategy, None
                aggregators_state = {key: {} for key in benchmark_aggregators}

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
                    aggregators_update = {}
                    for key, aggregator in benchmark_aggregators.items():
                        update = aggregator(
                            aggregators_state[key],
                            response,
                            request,
                            request_info,
                            scheduler_state,
                        )
                        if update:
                            aggregators_update.update(update)
                    yield aggregators_update, None, strategy, scheduler_state

                benchmark_kwargs = self._compile_benchmark_kwargs(
                    run_id=run_id,
                    run_index=len(profile.completed_strategies),
                    profile=profile,
                    requests=requests,
                    backend=backend,
                    environment=environment,
                    aggregators=benchmark_aggregators,
                    aggregators_state=aggregators_state,
                    strategy=strategy,
                    constraints=constraints,
                    scheduler_state=scheduler_state,
                )
                benchmark = benchmark_class(**benchmark_kwargs)
                yield {}, benchmark, strategy, None

                strategy, constraints = strategies_generator.send(benchmark)

    @classmethod
    def _compile_benchmark_kwargs(
        cls,
        run_id: str,
        profile: Profile,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, ResponseT],
        environment: Environment,
        aggregators: dict[
            str,
            Union[
                Aggregator[ResponseT, RequestT, RequestTimingsT],
                CompilableAggregator[ResponseT, RequestT, RequestTimingsT],
            ],
        ],
        aggregators_state: dict[str, dict[str, Any]],
        strategy: SchedulingStrategy,
        constraints: dict[str, Union[Any, dict[str, Any], Constraint]],
        scheduler_state: Optional[SchedulerState],
    ) -> dict[str, Any]:
        benchmark_kwargs = {
            "run_id": run_id,
            "run_index": len(profile.completed_strategies) - 1,
            "scheduler": {
                "strategy": strategy,
                "constraints": {
                    key: InfoMixin.extract_from_obj(val) for key, val in constraints
                },
                "state": scheduler_state,
            },
            "benchmarker": {
                "profile": profile,
                "requests": InfoMixin.extract_from_obj(requests),
                "backend": InfoMixin.extract_from_obj(backend),
                "environment": InfoMixin.extract_from_obj(environment),
                "aggregators": {
                    key: InfoMixin.extract_from_obj(aggregator)
                    for key, aggregator in aggregators.items()
                },
            },
            "system": {},
            "extras": {},
        }
        for key, aggregator in aggregators.items():
            if not isinstance(aggregator, CompilableAggregator):
                continue

            compiled = aggregator.compile(aggregators_state[key])

            if key not in benchmark_kwargs:
                benchmark_kwargs[key] = compiled
                continue

            existing_val = benchmark_kwargs[key]
            if not (isinstance(existing_val, dict) and isinstance(compiled, dict)):
                raise ValueError(
                    f"Key '{key}' already exists with value {existing_val} "
                    f"(type: {type(existing_val).__name__}) and cannot be "
                    f"overwritten with {compiled} (type: {type(compiled).__name__})"
                )
            existing_val.update(compiled)

        return benchmark_kwargs
