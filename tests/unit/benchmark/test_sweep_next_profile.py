"""Unit tests for the sweep_next benchmark profile."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from guidellm.benchmark.profiles import ProfileFactory, SweepNextProfile
from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncRampStrategy,
    RampSaturationConstraint,
    SchedulerState,
    SchedulerUpdateAction,
    SynchronousStrategy,
)
from guidellm.schemas.benchmark import SweepNextProfileArgs


def _generative_benchmark(
    rate: float,
    latency: float = 0.5,
    constraint_actions: dict[str, SchedulerUpdateAction] | None = None,
):
    """A GenerativeBenchmark stand-in exposing only what the sweep reads."""
    benchmark = Mock(spec=GenerativeBenchmark)
    benchmark.request_throughput = SimpleNamespace(
        successful=SimpleNamespace(mean=rate)
    )
    benchmark.request_latency = SimpleNamespace(
        successful=SimpleNamespace(mean=latency)
    )
    state = SchedulerState()
    state.scheduler_constraints = constraint_actions or {}
    benchmark.scheduler_state = state
    return benchmark


def _profile(**overrides) -> SweepNextProfile:
    args = SweepNextProfileArgs.model_validate(
        {"kind": "sweep_next", "sweep_size": 4, **overrides}
    )
    return SweepNextProfile(args, random_seed=42, constraints=None)


def _run_to_ramp(profile: SweepNextProfile, sync_rate=2.0, latency=0.5):
    sync = SynchronousStrategy()
    profile.completed_strategies.append(sync)
    ramp = profile.next_strategy(sync, _generative_benchmark(sync_rate, latency))
    profile.completed_strategies.append(ramp)
    return ramp


class TestSweepNextProfile:
    @pytest.mark.smoke
    def test_registered_as_its_own_profile(self):
        """
        sweep_next is created by its own kind and leaves sweep as it was.

        ## WRITTEN BY AI ##
        """
        profile = ProfileFactory.create(
            SweepNextProfileArgs.model_validate({"kind": "sweep_next"}),
            random_seed=42,
            constraints=None,
        )

        assert isinstance(profile, SweepNextProfile)
        assert profile.strategy_types[:2] == ["synchronous", "ramp"]

    @pytest.mark.smoke
    @pytest.mark.parametrize(("latency", "interval"), [(0.5, 2.0), (1.5, 6.0)])
    def test_ramp_is_sized_from_the_synchronous_step(self, latency, interval):
        """
        The ramp starts at the synchronous rate and doubles every four
        synchronous latencies, with a two second floor.

        ## WRITTEN BY AI ##
        """
        profile = _profile()

        ramp = _run_to_ramp(profile, sync_rate=2.0, latency=latency)

        assert isinstance(ramp, AsyncRampStrategy)
        assert ramp.start_rate == 2.0
        assert ramp.doubling_interval == pytest.approx(interval)

    @pytest.mark.sanity
    def test_ramp_gets_saturation_detection(self):
        """
        Only the ramp step gets the saturation constraint, sized to its interval.

        ## WRITTEN BY AI ##
        """
        profile = _profile()
        ramp = _run_to_ramp(profile)

        ramp_constraints = profile.next_strategy_constraints(
            ramp, SynchronousStrategy(), None
        )
        constant_constraints = profile.next_strategy_constraints(
            AsyncConstantStrategy(rate=1.0), ramp, None
        )

        tracker = (ramp_constraints or {})["ramp_saturation"]
        assert isinstance(tracker, RampSaturationConstraint)
        assert tracker.window_seconds == ramp.doubling_interval
        assert "ramp_saturation" not in (constant_constraints or {})

    @pytest.mark.sanity
    def test_constant_steps_span_up_to_the_ramp_peak(self):
        """
        The peak the ramp measured sets the top of the range and the conclusion.

        ## WRITTEN BY AI ##
        """
        profile = _profile()
        ramp = _run_to_ramp(profile, sync_rate=2.0)
        peak_action = SchedulerUpdateAction(
            metadata={"peak_rate": 20.0, "saturated_at": 12.0}
        )

        first = profile.next_strategy(
            ramp,
            _generative_benchmark(
                5.0, constraint_actions={"ramp_saturation": peak_action}
            ),
        )

        assert isinstance(first, AsyncConstantStrategy)
        assert profile.measured_rates == pytest.approx([11.0, 20.0])
        assert profile.conclusion == {
            "kind": "sweep_ramp",
            "synchronous_rate": 2.0,
            "peak_rate": 20.0,
            "peak_source": "saturated",
            "doubling_interval": ramp.doubling_interval,
        }

    @pytest.mark.sanity
    def test_falls_back_to_the_mean_rate_without_a_peak(self):
        """
        If the ramp ended before measuring a full window, the mean rate is used.

        ## WRITTEN BY AI ##
        """
        profile = _profile()
        ramp = _run_to_ramp(profile, sync_rate=2.0)

        profile.next_strategy(ramp, _generative_benchmark(8.0))

        assert profile.peak_rate == 8.0
        assert (profile.conclusion or {})["peak_source"] == "mean_rate"

    @pytest.mark.regression
    def test_ramp_stop_does_not_end_the_sweep(self):
        """
        A stop during the ramp does not end escalation, since the ramp pushes
        past sustainable load on purpose.

        ## WRITTEN BY AI ##
        """
        profile = _profile()
        ramp = _run_to_ramp(profile)
        stopped = _generative_benchmark(8.0)
        stopped.scheduler_state.end_queuing_constraints = {
            "max_errors": SchedulerUpdateAction(
                request_queuing="stop", stopping_scope="all"
            )
        }

        assert isinstance(profile.next_strategy(ramp, stopped), AsyncConstantStrategy)
