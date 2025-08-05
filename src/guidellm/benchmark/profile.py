"""
Benchmarking profile configurations for coordinating multi-strategy execution.

Provides configurable profile abstractions for orchestrating sequential and
parallel execution of different scheduling strategies during benchmarking,
with automatic strategy generation and constraint management.

Classes:
    Profile: Abstract base for multi-strategy benchmarking profiles.
    SynchronousProfile: Single synchronous strategy execution profile.
    ConcurrentProfile: Fixed-concurrency strategy execution profile.
    ThroughputProfile: Maximum throughput strategy execution profile.
    AsyncProfile: Rate-based asynchronous strategy execution profile.
    SweepProfile: Adaptive multi-strategy sweep execution profile.

Type Aliases:
    ProfileType: Literal type for supported profile configurations.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Generic, Literal, Optional, Union

import numpy as np
from pydantic import Field, computed_field

from guidellm.benchmark.aggregator import AggregatorT
from guidellm.benchmark.benchmark import BenchmarkT
from guidellm.objects import StandardBaseModel
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    CallableConstraint,
    CallableConstraintInitializer,
    ConcurrentStrategy,
    ConstraintsInitializerFactory,
    SchedulingStrategy,
    StrategyT,
    StrategyType,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.utils import RegistryMixin

__all__ = [
    "AsyncProfile",
    "ConcurrentProfile",
    "Profile",
    "ProfileType",
    "SweepProfile",
    "SynchronousProfile",
    "ThroughputProfile",
]

ProfileType = Literal["synchronous", "concurrent", "throughput", "async", "sweep"]


class Profile(
    StandardBaseModel,
    ABC,
    Generic[StrategyT, AggregatorT, BenchmarkT],
    RegistryMixin,
):
    """
    Abstract base for multi-strategy benchmarking execution profiles.

    Coordinates sequential execution of scheduling strategies with automatic
    strategy generation, constraint management, and completion tracking for
    comprehensive benchmarking workflows.
    """

    @classmethod
    def create(
        cls,
        rate_type: str,
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> "Profile":
        """
        Create a profile instance based on the specified type.

        :param rate_type: The type of profile to create.
        :param rate: Rate parameter for profile configuration.
        :param random_seed: Random seed for stochastic strategies.
        :param kwargs: Additional arguments for profile configuration.
        :return: Configured profile instance for the specified type.
        :raises ValueError: If the profile type is not registered.
        """
        profile_class: type[Profile] = cls.get_registered_object(rate_type)
        resolved_kwargs = profile_class.resolve_args(
            rate_type=rate_type, rate=rate, random_seed=random_seed, **kwargs
        )

        return profile_class(**resolved_kwargs)

    @classmethod
    @abstractmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve and validate arguments for profile construction.

        :param rate_type: The type of the profile.
        :param rate: Rate parameter for configuration.
        :param random_seed: Random seed for stochastic strategies.
        :param kwargs: Additional arguments to resolve.
        :return: Dictionary of resolved arguments for profile construction.
        """
        ...

    type_: Literal["profile"] = Field(
        description="The type of benchmarking profile to use",
    )
    completed_strategies: list[SchedulingStrategy] = Field(
        default_factory=list,
        description="The strategies that have completed execution",
    )
    constraints: Optional[
        dict[str, Union[Any, dict[str, Any], CallableConstraintInitializer]]
    ] = Field(
        default=None,
        description="Runtime constraints to apply during strategy execution",
    )

    @computed_field  # type: ignore[misc]
    @property
    def strategy_types(self) -> list[StrategyType]:
        """Get the types of all completed strategies in execution order."""
        return [strat.type_ for strat in self.completed_strategies]

    def strategies_generator(
        self,
    ) -> Generator[
        tuple[
            Optional[StrategyT],
            Optional[dict[str, Union[Any, dict[str, Any], CallableConstraint]]],
        ],
        tuple[AggregatorT, BenchmarkT],
        None,
    ]:
        """
        Generate strategies and constraints for sequential profile execution.

        :return: Generator yielding (strategy, constraints) tuples and
            receiving (aggregator, benchmark) results from each execution.
        """
        prev_strategy: Optional[StrategyT] = None
        prev_aggregator: Optional[AggregatorT] = None
        prev_benchmark: Optional[BenchmarkT] = None

        while (
            strategy := self.next_strategy(
                prev_strategy, prev_aggregator, prev_benchmark
            )
        ) is not None:
            constraints = self.next_strategy_constraints(
                strategy, prev_strategy, prev_aggregator, prev_benchmark
            )
            prev_aggregator, prev_benchmark = yield (
                strategy,
                constraints,
            )
            prev_strategy = strategy
            self.completed_strategies.append(prev_strategy)

    @abstractmethod
    def next_strategy(
        self,
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[StrategyT]:
        """
        Generate the next strategy to execute in the profile sequence.

        :param prev_strategy: The previously completed strategy.
        :param prev_aggregator: Result aggregator from the previous strategy.
        :param prev_benchmark: Benchmark results from the previous strategy.
        :return: Next strategy to execute, or None if profile is complete.
        """
        ...

    def next_strategy_constraints(
        self,
        next_strategy: Optional[StrategyT],
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[dict[str, Union[Any, dict[str, Any], CallableConstraint]]]:
        """
        Generate constraints for the next strategy execution.

        :param next_strategy: The next strategy to be executed.
        :param prev_strategy: The previously completed strategy.
        :param prev_aggregator: Result aggregator from the previous strategy.
        :param prev_benchmark: Benchmark results from the previous strategy.
        :return: Constraints dictionary for the next strategy, or None.
        """
        return (
            ConstraintsInitializerFactory.resolve(self.constraints)
            if next_strategy and self.constraints
            else None
        )


@Profile.register("synchronous")
class SynchronousProfile(Profile[StrategyT, AggregatorT, BenchmarkT]):
    """Single synchronous strategy execution profile."""

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for synchronous profile construction.

        :param rate_type: The type/strategy of the profile (ignored).
        :param rate: Rate parameter (must be None, will be stripped).
        :param random_seed: Random seed (ignored and stripped).
        :param kwargs: Additional arguments to pass through.
        :return: Dictionary of resolved arguments.
        :raises ValueError: If rate is not None.
        """
        if rate is not None:
            raise ValueError("SynchronousProfile does not accept a rate parameter")

        return kwargs

    @property
    def strategy_types(self) -> list[StrategyType]:
        """Get the single synchronous strategy type."""
        return [self.type_]

    def next_strategy(
        self,
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[StrategyT]:
        """
        Generate synchronous strategy or None if already completed.

        :param prev_strategy: The previously completed strategy (unused).
        :param prev_aggregator: Result aggregator from the previous strategy (unused).
        :param prev_benchmark: Benchmark results from the previous strategy (unused).
        :return: SynchronousStrategy for the first execution, None afterward.
        """
        if len(self.completed_strategies) >= 1:
            return None

        return SynchronousStrategy()


@Profile.register("concurrent")
class ConcurrentProfile(Profile[StrategyT, AggregatorT, BenchmarkT]):
    """Fixed-concurrency strategy execution profile with configurable stream counts."""

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: Union[int, list[int]] = Field(
        description="Number of concurrent streams for request scheduling",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "before completion-based timing"
        ),
        ge=0,
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for concurrent profile construction.

        :param rate_type: The type/strategy of the profile (ignored).
        :param rate: Rate parameter, remapped to streams.
        :param random_seed: Random seed (ignored and stripped).
        :param kwargs: Additional arguments to pass through.
        :return: Dictionary of resolved arguments.
        :raises ValueError: If rate is None.
        """
        kwargs["streams"] = rate
        return kwargs

    @property
    def strategy_types(self) -> list[StrategyType]:
        """Get concurrent strategy types for each configured stream count."""
        num_strategies = len(self.streams) if isinstance(self.streams, list) else 1
        return [self.type_] * num_strategies

    def next_strategy(
        self,
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[StrategyT]:
        """
        Generate concurrent strategy for the next stream count.

        :param prev_strategy: The previously completed strategy (unused).
        :param prev_aggregator: Result aggregator from the previous strategy (unused).
        :param prev_benchmark: Benchmark results from the previous strategy (unused).
        :return: ConcurrentStrategy with next stream count, or None if complete.
        """
        streams = self.streams if isinstance(self.streams, list) else [self.streams]

        if len(self.completed_strategies) >= len(streams):
            return None

        return ConcurrentStrategy(
            streams=streams[len(self.completed_strategies)],
            startup_duration=self.startup_duration,
        )


@Profile.register("throughput")
class ThroughputProfile(Profile[StrategyT, AggregatorT, BenchmarkT]):
    """
    Maximum throughput strategy execution profile with optional concurrency limits.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: Optional[int] = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "before full throughput scheduling"
        ),
        ge=0,
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for throughput profile construction.

        :param rate_type: The type/strategy of the profile (ignored).
        :param rate: Rate parameter to remap to max_concurrency.
        :param random_seed: Random seed (ignored and stripped).
        :param kwargs: Additional arguments to pass through.
        :return: Dictionary of resolved arguments.
        """
        # Remap rate to max_concurrency, strip out random_seed
        kwargs.pop("random_seed", None)
        if rate is not None:
            kwargs["max_concurrency"] = rate
        return kwargs

    @property
    def strategy_types(self) -> list[StrategyType]:
        """Get the single throughput strategy type."""
        return [self.type_]

    def next_strategy(
        self,
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[StrategyT]:
        """
        Generate throughput strategy or None if already completed.

        :param prev_strategy: The previously completed strategy (unused).
        :param prev_aggregator: Result aggregator from the previous strategy (unused).
        :param prev_benchmark: Benchmark results from the previous strategy (unused).
        :return: ThroughputStrategy for the first execution, None afterward.
        """
        if len(self.completed_strategies) >= 1:
            return None

        return ThroughputStrategy(
            max_concurrency=self.max_concurrency,
            startup_duration=self.startup_duration,
        )


@Profile.register(["async", "constant", "poisson"])
class AsyncProfile(Profile[StrategyT, AggregatorT, BenchmarkT]):
    """
    Rate-based asynchronous strategy execution profile with configurable patterns.
    """

    type_: Literal["async"] = "async"  # type: ignore[assignment]
    strategy_type: Literal["constant", "poisson"] = Field(
        description="Type of asynchronous strategy pattern to use",
    )
    rate: Union[float, list[float]] = Field(
        description="Request scheduling rate in requests per second",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "to converge quickly to desired rate"
        ),
        ge=0,
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
        gt=0,
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution strategy",
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: Union[ProfileType, StrategyT],
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for async profile construction.

        :param rate_type: The type/strategy of the profile.
        :param rate: Rate parameter for the profile.
        :param random_seed: Random seed for stochastic strategies.
        :param kwargs: Additional arguments to pass through.
        :return: Dictionary of resolved arguments.
        :raises ValueError: If rate is None.
        """
        if rate is None:
            raise ValueError("AsyncProfile requires a rate parameter")

        kwargs["strategy_type"] = (
            rate_type
            if rate_type in ["constant", "poisson"]
            else kwargs.get("strategy_type", "constant")
        )
        kwargs["rate"] = rate
        kwargs["random_seed"] = random_seed
        return kwargs

    @property
    def strategy_types(self) -> list[StrategyType]:
        """Get async strategy types for each configured rate."""
        num_strategies = len(self.rate) if isinstance(self.rate, list) else 1
        return [self.strategy_type] * num_strategies

    def next_strategy(
        self,
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[StrategyT]:
        """
        Generate async strategy for the next configured rate.

        :param prev_strategy: The previously completed strategy (unused).
        :param prev_aggregator: Result aggregator from the previous strategy (unused).
        :param prev_benchmark: Benchmark results from the previous strategy (unused).
        :return: AsyncConstantStrategy or AsyncPoissonStrategy for next rate,
            or None if all rates completed.
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'.
        """
        rate = self.rate if isinstance(self.rate, list) else [self.rate]

        if len(self.completed_strategies) >= len(rate):
            return None

        current_rate = rate[len(self.completed_strategies)]

        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=current_rate,
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=current_rate,
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")


@Profile.register("sweep")
class SweepProfile(Profile[StrategyT, AggregatorT, BenchmarkT]):
    """
    Adaptive multi-strategy sweep execution profile with rate discovery.
    """

    type_: Literal["sweep"] = "sweep"  # type: ignore[assignment]
    sweep_size: int = Field(
        description="Number of strategies to generate for the sweep",
    )
    strategy_type: Literal["constant", "poisson"] = "constant"
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds for distributing startup requests "
            "to converge quickly to desired rate"
        ),
        ge=0,
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
        gt=0,
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution strategy",
    )
    synchronous_rate: float = Field(
        default=-1.0,
        description="Measured rate from synchronous strategy execution",
    )
    throughput_rate: float = Field(
        default=-1.0,
        description="Measured rate from throughput strategy execution",
    )
    async_rates: list[float] = Field(
        default_factory=list,
        description="Generated rates for async strategy sweep",
    )
    measured_rates: list[float] = Field(
        default_factory=list,
        description="Calculated interpolated rates between synchronous and throughput",
    )

    @classmethod
    def resolve_args(
        cls,
        rate_type: str,
        rate: Optional[Union[float, int, list[float, int]]],
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Resolve arguments for sweep profile construction.

        :param rate_type: The type/strategy for async strategies in the sweep.
        :param rate: Rate parameter (ignored for sweep).
        :param random_seed: Random seed for stochastic strategies.
        :param kwargs: Additional arguments to pass through.
        :return: Dictionary of resolved arguments.
        """
        kwargs["sweep_size"] = kwargs.get("sweep_size", rate)
        kwargs["random_seed"] = random_seed
        if rate_type in ["constant", "poisson"]:
            kwargs["strategy_type"] = rate_type
        return kwargs

    @property
    def strategy_types(self) -> list[StrategyType]:
        """Get strategy types for the complete sweep sequence."""
        types = ["synchronous", "throughput"]
        types += [self.strategy_type] * (self.sweep_size - len(types))
        return types

    def next_strategy(
        self,
        prev_strategy: Optional[StrategyT],
        prev_aggregator: Optional[AggregatorT],
        prev_benchmark: Optional[BenchmarkT],
    ) -> Optional[StrategyT]:
        """
        Generate the next strategy in the adaptive sweep sequence.

        Executes synchronous and throughput strategies first to measure
        baseline rates, then generates interpolated rates for async strategies.

        :param prev_strategy: The previously completed strategy.
        :param prev_aggregator: Result aggregator from the previous strategy (unused).
        :param prev_benchmark: Benchmark results from the previous strategy.
        :return: Next strategy in sweep sequence, or None if complete.
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'.
        """
        if prev_strategy is None:
            return SynchronousStrategy()

        if prev_strategy.type_ == "synchronous":
            self.synchronous_rate = (
                prev_benchmark.metrics.requests_per_second.successful.mean
            )

            return ThroughputStrategy(
                max_concurrency=self.max_concurrency,
                startup_duration=self.startup_duration,
            )

        if prev_strategy.type_ == "throughput":
            self.throughput_rate = (
                prev_benchmark.metrics.requests_per_second.successful.mean
            )
            self.measured_rates = list(
                np.linspace(
                    self.synchronous_rate,
                    self.throughput_rate,
                    self.sweep_size - 1,
                )
            )[1:]  # don't rerun synchronous

        if len(self.completed_strategies) >= self.sweep_size:
            return None

        next_rate_index = len(
            [
                strat
                for strat in self.completed_strategies
                if strat.type_ == self.strategy_type
            ]
        )

        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=self.measured_rates[next_rate_index],
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
            )
        elif self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=self.measured_rates[next_rate_index],
                startup_duration=self.startup_duration,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
