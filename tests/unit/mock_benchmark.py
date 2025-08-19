# TODO: Review Cursor generated code (start)
"""Mock benchmark objects for unit testing."""
# TODO: Review Cursor generated code (end)

# TODO: Review Cursor generated code (start)
from guidellm.backend import GenerationRequestTimings

# TODO: Review Cursor generated code (end)
from guidellm.benchmark import (
    BenchmarkSchedulerStats,
    GenerativeBenchmark,
    # TODO: Review Cursor generated code (start)
    GenerativeMetrics,
    # TODO: Review Cursor generated code (end)
    GenerativeRequestStats,
)

# TODO: Review Cursor generated code (start)
from guidellm.benchmark.objects import BenchmarkerDict, SchedulerDict
from guidellm.benchmark.profile import SynchronousProfile
from guidellm.scheduler import ScheduledRequestInfo, SchedulerState, SynchronousStrategy
from guidellm.utils import (
    DistributionSummary,
    Percentiles,
    StandardBaseDict,
    StatusBreakdown,
    StatusDistributionSummary,
)

# TODO: Review Cursor generated code (end)

__all__ = ["mock_generative_benchmark"]


# TODO: Review Cursor generated code (start)
def _create_mock_percentiles() -> Percentiles:
    """Create mock percentiles for testing."""
    return Percentiles(
        p001=0.1,
        p01=1.0,
        p05=5.0,
        p10=10.0,
        p25=25.0,
        p50=50.0,
        p75=75.0,
        p90=90.0,
        p95=95.0,
        p99=99.0,
        p999=99.9,
    )


# TODO: Review Cursor generated code (end)


# TODO: Review Cursor generated code (start)
def _create_mock_distribution() -> DistributionSummary:
    """Create mock distribution summary for testing."""
    return DistributionSummary(
        mean=50.0,
        median=50.0,
        mode=50.0,
        variance=10.0,
        std_dev=3.16,
        min=10.0,
        max=100.0,
        count=100,
        total_sum=5000.0,
        percentiles=_create_mock_percentiles(),
    )


# TODO: Review Cursor generated code (end)


# TODO: Review Cursor generated code (start)
def _create_status_dist() -> StatusDistributionSummary:
    """Create mock status distribution summary for testing."""
    dist = _create_mock_distribution()
    return StatusDistributionSummary(
        successful=dist,
        incomplete=dist,
        errored=dist,
        total=dist,
    )


# TODO: Review Cursor generated code (end)


def mock_generative_benchmark() -> GenerativeBenchmark:
    # TODO: Review Cursor generated code (start)
    """Create a minimal mock GenerativeBenchmark for testing purposes."""
    return GenerativeBenchmark(
        run_id="test-run-gen",
        run_index=0,
        scheduler=SchedulerDict(
            # TODO: Review Cursor generated code (end)
            strategy=SynchronousStrategy(),
            # TODO: Review Cursor generated code (start)
            constraints={},
            state=SchedulerState(node_id=0, num_processes=1),
        ),
        benchmarker=BenchmarkerDict(
            profile=SynchronousProfile.create("synchronous", rate=None),
            requests={},
            backend={},
            environment={},
            aggregators={},
            # TODO: Review Cursor generated code (end)
        ),
        # TODO: Review Cursor generated code (start)
        env_args=StandardBaseDict(),
        extras=StandardBaseDict(),
        # TODO: Review Cursor generated code (end)
        run_stats=BenchmarkSchedulerStats(
            # TODO: Review Cursor generated code (start)
            start_time=1,
            end_time=2,
            # TODO: Review Cursor generated code (end)
            requests_made=StatusBreakdown(
                # TODO: Review Cursor generated code (start)
                successful=1,
                incomplete=0,
                errored=0,
                total=1,
                # TODO: Review Cursor generated code (end)
            ),
            # TODO: Review Cursor generated code (start)
            queued_time_avg=0.1,
            worker_resolve_start_delay_avg=0.1,
            worker_resolve_time_avg=0.1,
            worker_resolve_end_delay_avg=0.1,
            finalized_delay_avg=0.1,
            worker_targeted_start_delay_avg=0.1,
            request_start_delay_avg=0.1,
            request_time_avg=0.1,
            request_targeted_delay_avg=0.1,
            # TODO: Review Cursor generated code (end)
        ),
        # TODO: Review Cursor generated code (start)
        start_time=1000.0,
        end_time=2000.0,
        metrics=GenerativeMetrics(
            requests_per_second=_create_status_dist(),
            request_concurrency=_create_status_dist(),
            request_latency=_create_status_dist(),
            prompt_token_count=_create_status_dist(),
            output_token_count=_create_status_dist(),
            total_token_count=_create_status_dist(),
            time_to_first_token_ms=_create_status_dist(),
            time_per_output_token_ms=_create_status_dist(),
            inter_token_latency_ms=_create_status_dist(),
            output_tokens_per_second=_create_status_dist(),
            tokens_per_second=_create_status_dist(),
            # TODO: Review Cursor generated code (end)
        ),
        # TODO: Review Cursor generated code (start)
        request_totals=StatusBreakdown(
            successful=1,
            incomplete=0,
            errored=0,
            total=1,
            # TODO: Review Cursor generated code (end)
        ),
        # TODO: Review Cursor generated code (start)
        requests=StatusBreakdown(
            successful=[
                GenerativeRequestStats(
                    scheduler_info=ScheduledRequestInfo(
                        request_timings=GenerationRequestTimings(
                            request_start=1,
                            first_iteration=2,
                            last_iteration=6,
                            request_end=6,
                        )
                    ),
                    request_id="a",
                    request_type="text_completions",
                    prompt="p",
                    request_args={},
                    output="o",
                    iterations=1,
                    prompt_tokens=1,
                    output_tokens=2,
                )
            ],
            incomplete=[],
            errored=[],
            total=None,
        ),
    )  # TODO: Review Cursor generated code (end)
