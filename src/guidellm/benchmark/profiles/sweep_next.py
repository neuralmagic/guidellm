"""Sweep profile that finds the top of its range with a ramp."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import numpy as np

from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.logger import logger
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    AsyncRampStrategy,
    Constraint,
    ConstraintInitializer,
    RampSaturationConstraint,
    SchedulingStrategy,
    SynchronousStrategy,
)
from guidellm.schemas.benchmark.profiles import SweepNextProfileArgs

from .profile import Profile, ProfileFactory

__all__ = ["SweepNextProfile"]

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark

# Ramp sizing, relative to the synchronous step's request latency.
RAMP_DOUBLING_LATENCIES = 4.0
RAMP_MIN_DOUBLING_SECONDS = 2.0
RAMP_CONSTRAINT_KEY = "ramp_saturation"
RAMP_NOT_SATURATED_WARNING = (
    "The sweep ramp stopped before the server saturated, so its peak of "
    "{:.2f} req/s is a lower bound and the constant steps may stop short of "
    "capacity. Give each step more time, for example with a higher --max-seconds."
)


@ProfileFactory.register("sweep_next")
class SweepNextProfile(Profile):
    """
    Sweep the rate range between the synchronous rate and a ramp-measured peak.

    Runs like the sweep profile, but replaces its fixed-concurrency throughput
    step with a ramp that raises the request rate from the synchronous rate
    until completions fall behind what was sent. The highest completion rate
    reached sets the top of the range for the interpolated async steps.
    """

    args: SweepNextProfileArgs

    def __init__(
        self,
        args: SweepNextProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args
        self.synchronous_rate = -1.0
        self.peak_rate = -1.0
        self.measured_rates: list[float] = []
        self._ramp_doubling_interval: float | None = None
        self._ramp_peak_source: str | None = None

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Strategy types for the complete sweep sequence
        """
        types = ["synchronous", "ramp"]
        types += [self.args.strategy_type] * (self.args.sweep_size - len(types))
        return types

    @property
    def conclusion(self) -> dict[str, Any] | None:
        """
        :return: The peak the ramp found, or None before the ramp has run
        """
        if self._ramp_peak_source is None:
            return None

        return {
            "kind": "sweep_ramp",
            "synchronous_rate": self.synchronous_rate,
            "peak_rate": self.peak_rate,
            "peak_source": self._ramp_peak_source,
            "doubling_interval": self._ramp_doubling_interval,
        }

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> (
        AsyncConstantStrategy
        | AsyncPoissonStrategy
        | AsyncRampStrategy
        | SynchronousStrategy
        | None
    ):
        """
        Generate next strategy in the sweep sequence.

        Runs the synchronous step, then the ramp, then interpolated async rates
        between the two measured rates. If a failure constraint is triggered
        during the async phase, all remaining higher rates are skipped.

        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Next strategy in sweep sequence, or None if complete
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'
        """
        if prev_strategy is None:
            return SynchronousStrategy()

        if prev_strategy.type_ == "synchronous":
            self.synchronous_rate = prev_benchmark.request_throughput.successful.mean
            return self._ramp_strategy(prev_benchmark)  # type: ignore[arg-type]

        if prev_strategy.type_ == "ramp":
            self.peak_rate = self._peak_rate(prev_benchmark)  # type: ignore[arg-type]
            if self.synchronous_rate <= 0 and self.peak_rate <= 0:
                raise RuntimeError(
                    "Invalid rates in sweep; aborting. "
                    "Were there any successful requests?"
                )
            self.measured_rates = list(
                np.linspace(
                    self.synchronous_rate,
                    self.peak_rate,
                    self.args.sweep_size - 1,
                )
            )[1:]  # don't rerun synchronous

        # Stop escalation if a constraint with stopping_scope='all' triggered
        # during the async phase. The ramp is excluded because it intentionally
        # pushes beyond sustainable load. Synchronous never reaches here.
        if (
            prev_strategy.type_ != "ramp"
            and self._should_stop_escalating(prev_benchmark)  # type: ignore[arg-type]
        ):
            return None

        next_index = len(self.completed_strategies) - 1 - 1  # synchronous and ramp
        next_rate = (
            self.measured_rates[next_index]
            if next_index < len(self.measured_rates)
            else None
        )

        if next_rate is None or next_rate <= 0:
            # Stop if we don't have another valid rate to run
            return None

        if self.args.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=next_rate,
                max_concurrency=self.args.max_concurrency,
                rampup_duration=self.args.rampup_duration,
            )
        if self.args.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=next_rate,
                max_concurrency=self.args.max_concurrency,
                random_seed=self.random_seed,
            )
        raise ValueError(f"Invalid strategy type: {self.args.strategy_type}")

    def next_strategy_constraints(
        self,
        next_strategy: SchedulingStrategy | None,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> dict[str, Constraint] | None:
        """
        Add saturation detection to the ramp step.

        The ramp has no end of its own. It stops sending once completions fall
        short of what was sent one latency earlier, then measures the peak while
        the requests already sent complete. User constraints still apply
        alongside it.

        :param next_strategy: Strategy to be executed next
        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Constraints dictionary for next strategy, or None
        """
        constraints = super().next_strategy_constraints(
            next_strategy, prev_strategy, prev_benchmark
        )
        if not isinstance(next_strategy, AsyncRampStrategy):
            return constraints

        constraints = dict(constraints or {})
        constraints[RAMP_CONSTRAINT_KEY] = RampSaturationConstraint(
            window_seconds=next_strategy.doubling_interval
        )
        return constraints

    def _ramp_strategy(self, synchronous: Benchmark) -> AsyncRampStrategy:
        latency = synchronous.request_latency.successful.mean
        self._ramp_doubling_interval = max(
            RAMP_DOUBLING_LATENCIES * latency, RAMP_MIN_DOUBLING_SECONDS
        )

        if self.synchronous_rate <= 0:
            raise RuntimeError(
                "Invalid synchronous rate in sweep; cannot start the ramp. "
                "Were there any successful requests?"
            )

        return AsyncRampStrategy(
            start_rate=self.synchronous_rate,
            doubling_interval=self._ramp_doubling_interval,
            max_concurrency=self.args.max_concurrency,
        )

    def _peak_rate(self, benchmark: Benchmark) -> float:
        if isinstance(benchmark, GenerativeBenchmark):
            action = benchmark.scheduler_state.scheduler_constraints.get(
                RAMP_CONSTRAINT_KEY
            )
            peak = action.metadata.get("peak_rate") if action else None
            if peak:
                # A duration limit can end the ramp before it saturates.
                self._ramp_peak_source = (
                    "saturated"
                    if action and action.metadata.get("saturated_at") is not None
                    else "not_saturated"
                )
                if self._ramp_peak_source == "not_saturated":
                    logger.warning(RAMP_NOT_SATURATED_WARNING, peak)
                return peak

        # No full window completed, so use the mean rate.
        mean_rate = benchmark.request_throughput.successful.mean
        self._ramp_peak_source = "mean_rate"
        logger.warning(RAMP_NOT_SATURATED_WARNING, mean_rate)
        return mean_rate
