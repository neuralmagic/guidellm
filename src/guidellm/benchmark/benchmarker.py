"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine that coordinates request scheduling,
data aggregation, and result compilation across different execution strategies
and environments.

Classes:
    Benchmarker: Abstract benchmark orchestrator for request processing workflows.

Type Variables:
    BenchmarkT: Generic benchmark result type.
    RequestT: Generic request object type.
    RequestTimingsT: Generic request timing object type.
    ResponseT: Generic response object type.
"""

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
    BackendInterface,
    Constraint,
    Environment,
    MeasuredRequestTimingsT,
    RequestT,
    ResponseT,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.utils import InfoMixin, ThreadSafeSingletonMixin

__all__ = ["Benchmarker"]


class Benchmarker(
    Generic[BenchmarkT, RequestT, MeasuredRequestTimingsT, ResponseT],
    ABC,
    ThreadSafeSingletonMixin,
):
    """
    Abstract benchmark orchestrator for request processing workflows.

    Coordinates the execution of benchmarking runs across different scheduling
    strategies, aggregating metrics and compiling results. Manages the complete
    benchmark lifecycle from request submission through result compilation.

    Implements thread-safe singleton pattern to ensure consistent state across
    concurrent benchmark operations.
    """

    async def run(
        self,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendInterface[RequestT, MeasuredRequestTimingsT, ResponseT],
        profile: Profile,
        environment: Environment,
        benchmark_aggregators: dict[
            str,
            Union[
                Aggregator[ResponseT, RequestT, MeasuredRequestTimingsT],
                CompilableAggregator[ResponseT, RequestT, MeasuredRequestTimingsT],
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
        """
        Execute benchmark runs across multiple scheduling strategies.

        Orchestrates the complete benchmark workflow: iterates through scheduling
        strategies from the profile, executes requests through the scheduler,
        aggregates metrics, and compiles final benchmark results.

        :param requests: Request datasets for processing across strategies.
        :param backend: Backend interface for request processing.
        :param profile: Benchmark profile defining strategies and constraints.
        :param environment: Execution environment for coordination.
        :param benchmark_aggregators: Metric aggregation functions by name.
        :param benchmark_class: Class for constructing final benchmark objects.
        :yield: Tuples of (metrics_update, benchmark_result, strategy, state).
        :raises Exception: If benchmark execution or compilation fails.
        """
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
                ) in Scheduler().run(
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

                try:
                    strategy, constraints = strategies_generator.send(benchmark)
                except StopIteration:
                    # No more strategies, exit the loop
                    strategy = None

    @classmethod
    def _compile_benchmark_kwargs(
        cls,
        run_id: str,
        run_index: int,
        profile: Profile,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendInterface[RequestT, MeasuredRequestTimingsT, ResponseT],
        environment: Environment,
        aggregators: dict[
            str,
            Union[
                Aggregator[ResponseT, RequestT, MeasuredRequestTimingsT],
                CompilableAggregator[ResponseT, RequestT, MeasuredRequestTimingsT],
            ],
        ],
        aggregators_state: dict[str, dict[str, Any]],
        strategy: SchedulingStrategy,
        constraints: dict[str, Union[Any, dict[str, Any], Constraint]],
        scheduler_state: Optional[SchedulerState],
    ) -> dict[str, Any]:
        """
        Compile benchmark construction parameters from execution results.

        Aggregates metadata from scheduler execution and compiles it into
        structured parameters for benchmark object construction.

        :param run_id: Unique identifier for the benchmark run.
        :param run_index: Index of this strategy in the benchmark profile.
        :param profile: Benchmark profile containing strategy configuration.
        :param requests: Request datasets used for the benchmark.
        :param backend: Backend interface used for request processing.
        :param environment: Execution environment for coordination.
        :param aggregators: Metric aggregation functions by name.
        :param aggregators_state: Current state of metric aggregators.
        :param strategy: Scheduling strategy that was executed.
        :param constraints: Runtime constraints applied during execution.
        :param scheduler_state: Final state of scheduler execution.
        :return: Dictionary of parameters for benchmark object construction.
        :raises ValueError: If aggregator output conflicts with existing keys.
        """
        from guidellm import logger

        logger.debug(
            f"_compile_benchmark_kwargs called with constraints type: {type(constraints)}"
        )
        logger.debug(f"constraints content: {constraints}")
        logger.debug(f"constraints has items method: {hasattr(constraints, 'items')}")

        constraints_items = (
            constraints.items() if hasattr(constraints, "items") else constraints
        )
        logger.debug(f"constraints_items type: {type(constraints_items)}")

        try:
            constraints_list = list(constraints_items)
            logger.debug(f"constraints_list: {constraints_list}")
            logger.debug(
                f"Sample constraint item: {constraints_list[0] if constraints_list else 'No items'}"
            )
        except Exception as e:
            logger.error(f"Error converting constraints to list: {e}")

        benchmark_kwargs = {
            "run_id": run_id,
            "run_index": run_index,
            "scheduler": {
                "strategy": strategy,
                "constraints": {
                    key: InfoMixin.extract_from_obj(val)
                    for key, val in constraints_items
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
            "env_args": {},  # Environment arguments - empty for now
            "system": {},
            "extras": {},
        }

        for key, aggregator in aggregators.items():
            if not isinstance(aggregator, CompilableAggregator):
                continue

            compiled = aggregator.compile(aggregators_state[key], scheduler_state)

            # Handle key mapping for scheduler_stats -> run_stats
            if key == "scheduler_stats" and "scheduler_stats" in compiled:
                compiled["run_stats"] = compiled.pop("scheduler_stats")

            # Flatten aggregator results into top-level benchmark_kwargs
            for field_key, field_value in compiled.items():
                benchmark_kwargs[field_key] = field_value

        return benchmark_kwargs
