from __future__ import annotations

import pytest

from guidellm.benchmark.schemas import BenchmarkConfig
from guidellm.benchmark.schemas.accumulator import (
    GenerativeBenchmarkTimings,
    GenerativeRequestsAccumulator,
)
from guidellm.scheduler import (
    SchedulerProgress,
    SchedulerState,
    SynchronousStrategy,
)
from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    UsageMetrics,
)
from guidellm.schemas.benchmark import TransientPhaseConfig


def _make_stats(
    request_id: str = "req-1",
    output: str | None = "some output",
    reasoning_output: str | None = None,
    request_args: str | None = "args",
) -> GenerativeRequestStats:
    """Build a minimal GenerativeRequestStats for testing."""
    info = RequestInfo(request_id=request_id, status="completed")
    info.timings.request_start = 0.0
    info.timings.request_end = 1.0
    info.timings.resolve_end = 1.0
    return GenerativeRequestStats(
        request_id=request_id,
        request_args=request_args,
        output=output,
        reasoning_output=reasoning_output,
        info=info,
        input_metrics=UsageMetrics(text_tokens=5),
        output_metrics=UsageMetrics(text_tokens=10),
    )


class TestClearStatsData:
    """
    Tests for GenerativeRequestsAccumulator.clear_stats_data.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_clears_output_and_reasoning_output(self):
        """
        clear_stats_data clears output, reasoning_output, and tool_calls
        when clear_nonsampled_outputs is True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=False,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.tool_calls is None
        assert stats.request_args == "args"

    @pytest.mark.smoke
    def test_clears_request_args(self):
        """
        clear_stats_data clears request_args when
        clear_nonsampled_request_args is True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=False,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args is None
        assert stats.output == "answer"
        assert stats.reasoning_output == "thinking..."

    @pytest.mark.smoke
    def test_clears_both(self):
        """
        clear_stats_data clears all fields when both flags are True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args is None
        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.tool_calls is None

    @pytest.mark.smoke
    def test_clears_by_index(self):
        """
        clear_stats_data accepts an integer index to look up the stats
        in requests_stats.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(reasoning_output="step 1")
        acc.requests_stats.append(stats)

        acc.clear_stats_data(0)

        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.request_args is None

    @pytest.mark.smoke
    def test_preserves_when_both_flags_false(self):
        """
        clear_stats_data preserves all fields when both flags are False.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=False,
            clear_nonsampled_request_args=False,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args == "args"
        assert stats.output == "answer"
        assert stats.reasoning_output == "thinking..."


def _make_timed_stats(
    request_id: str,
    request_start: float,
    request_end: float,
) -> GenerativeRequestStats:
    """Build a GenerativeRequestStats with explicit request timing bounds."""
    info = RequestInfo(request_id=request_id, status="completed")
    info.timings.request_start = request_start
    info.timings.request_end = request_end
    info.timings.resolve_end = request_end
    return GenerativeRequestStats(
        request_id=request_id,
        info=info,
        input_metrics=UsageMetrics(text_tokens=5),
        output_metrics=UsageMetrics(text_tokens=10),
    )


class TestGetWithinRange:
    """
    Tests for GenerativeRequestsAccumulator.get_within_range.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.sanity
    def test_partial_overlap_includes_events_crossing_boundary(self):
        """
        Default closed=False includes events that partially overlap the window.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator()
        acc.requests_stats = [
            _make_timed_stats("before", 5.0, 12.0),
            _make_timed_stats("inside", 11.0, 18.0),
            _make_timed_stats("after", 18.0, 24.0),
        ]

        in_range = acc.get_within_range(10.0, 20.0)

        assert [stats.request_id for stats in in_range] == [
            "before",
            "inside",
            "after",
        ]

    @pytest.mark.sanity
    def test_closed_range_requires_full_enclosure(self):
        """
        closed=True excludes boundary-crossing events and keeps only fully
        enclosed ones.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator()
        acc.requests_stats = [
            _make_timed_stats("before", 5.0, 12.0),
            _make_timed_stats("inside", 11.0, 18.0),
            _make_timed_stats("after", 18.0, 24.0),
        ]

        in_range = acc.get_within_range(10.0, 20.0, closed=True)

        assert [stats.request_id for stats in in_range] == ["inside"]


def _run_synchronous_timings(
    num_requests: int, warmup: int | None, cooldown: int | None
) -> tuple[GenerativeBenchmarkTimings, list[tuple[float, float]]]:
    """
    Feed a synchronous run's request updates through GenerativeBenchmarkTimings.

    Request i starts at i and ends at i + 0.5; each request sends an in-progress
    update and then a completed update, as the scheduler does.
    """
    config = BenchmarkConfig(
        run_id="transient-phases",
        run_index=0,
        strategy=SynchronousStrategy(),
        constraints={},
        profile={},
        requests={},
        backend={},
        environment={},
        warmup=TransientPhaseConfig(value=warmup, mode="requests"),
        cooldown=TransientPhaseConfig(value=cooldown, mode="requests"),
    )
    timings = GenerativeBenchmarkTimings()
    state = SchedulerState(
        start_requests_time=1.0,
        progress=SchedulerProgress(
            total_requests=num_requests, remaining_requests=num_requests
        ),
    )
    spans = []

    for index in range(1, num_requests + 1):
        start, end = float(index), index + 0.5
        spans.append((start, end))

        info = RequestInfo(request_id=f"req-{index}", status="in_progress")
        info.timings.request_start = start
        timings.update_estimate(info, state, config)

        state.processed_requests = index
        state.progress.remaining_requests = num_requests - index
        if index == num_requests:
            state.end_processing_time = end
        info = RequestInfo(request_id=f"req-{index}", status="completed")
        info.timings.request_start = start
        info.timings.request_end = end
        info.timings.resolve_end = end
        timings.update_estimate(info, state, config)

    return timings, spans


class TestTransientPhasesByRequests:
    """
    Request-count warmup and cooldown exclude exactly the configured requests.
    """

    @pytest.mark.regression
    @pytest.mark.parametrize(
        ("warmup", "cooldown", "expected"),
        [
            (None, None, list(range(1, 11))),
            (2, None, list(range(3, 11))),
            (None, 2, list(range(1, 9))),
            (2, 2, list(range(3, 9))),
            (3, 1, list(range(4, 10))),
        ],
    )
    def test_measured_requests(
        self, warmup: int | None, cooldown: int | None, expected: list[int]
    ):
        """
        Requests overlapping the measured window are the ones left after removing
        warmup requests from the start and cooldown requests from the end.
        """
        timings, spans = _run_synchronous_timings(10, warmup, cooldown)
        start = timings.finalized_measure_start
        end = timings.finalized_measure_end

        measured = [
            index
            for index, (req_start, req_end) in enumerate(spans, start=1)
            if req_end >= start and req_start <= end
        ]

        assert measured == expected
