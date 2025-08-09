"""
Benchmark result aggregation and compilation interfaces.

Provides protocols and implementations for collecting, processing, and compiling
benchmark data from scheduler executions into final metrics and statistics.

Classes:
    Aggregator: Protocol for processing benchmark data updates.
    CompilableAggregator: Protocol for aggregators that can compile final results.
    SchedulerStatsAggregator: Aggregates scheduler timing and performance metrics.
    GenerativeRequestsStatsProgressAggregator: Tracks generation metrics during run.
    GenerativeRequestsAggregator: Compiles complete generative benchmark results.

Functions:
    add_aggregate_metric: Helper for accumulating timing and count metrics.

Type Variables:
    RequestT: Generic request object type.
    ResponseT: Generic response object type.
    RequestTimingsT: Generic request timing object type.
"""

import math
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import Field

from guidellm.backend import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.benchmark.benchmark import (
    BenchmarkSchedulerStats,
    GenerativeMetrics,
    GenerativeRequestStats,
)
from guidellm.config import settings
from guidellm.objects import (
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
)
from guidellm.scheduler import (
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)

__all__ = [
    "Aggregator",
    "CompilableAggregator",
    "GenerativeRequestsAggregator",
    "GenerativeRequestsStatsProgressAggregator",
    "SchedulerStatsAggregator",
    "add_aggregate_metric",
]


@runtime_checkable
class Aggregator(Protocol[ResponseT, RequestT, RequestTimingsT]):
    """
    Protocol for processing benchmark data updates during execution.

    Defines the interface for aggregators that collect and process request/response
    data from scheduler executions. Implementations update aggregation state with
    each completed request for eventual compilation into final metrics.
    """

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[ResponseT],
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        """
        Process a completed request and update aggregation state.

        :param agg_state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
        """
        ...


@runtime_checkable
class CompilableAggregator(Aggregator[ResponseT, RequestT, RequestTimingsT]):
    """
    Protocol for aggregators that compile final results from aggregated state.

    Extends the Aggregator protocol with the ability to transform accumulated
    state into final benchmark results and metrics after execution completes.
    """

    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile aggregated state into final benchmark results.

        :param agg_state: The accumulated aggregation state.
        :param scheduler_state: Final scheduler execution state.
        :return: Compiled benchmark results and metrics.
        """


def add_aggregate_metric(
    base_key: str,
    agg_state: dict[str, Any],
    end_val: Optional[Union[int, float]],
    start_val: Optional[Union[int, float]] = 0.0,
    count: int = 1,
):
    """
    Add timing or count metrics to aggregation state.

    Accumulates delta values and counts for computing averages and totals.
    Creates entries for "{base_key}_total" and "{base_key}_count" in agg_state.

    :param base_key: Base key name for the metric.
    :param agg_state: Aggregation state dictionary to update.
    :param end_val: End value for calculating delta, or None to skip.
    :param start_val: Start value for calculating delta, defaults to 0.0.
    :param count: Number of occurrences to count, defaults to 1.
    """
    if start_val is None or end_val is None:
        return

    delta_val = end_val - start_val
    agg_state[f"{base_key}_total"] = agg_state.get(f"{base_key}_total", 0) + delta_val
    agg_state[f"{base_key}_count"] = agg_state.get(f"{base_key}_count", 0) + count


class SchedulerStatsAggregator(
    CompilableAggregator[ResponseT, RequestT, RequestTimingsT]
):
    """
    Aggregates scheduler timing and performance metrics.

    Collects timing data for various scheduler phases including queuing,
    resolution, and processing delays to generate performance statistics.
    """

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[ResponseT],
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        """
        Aggregate scheduler timing metrics for a completed request.

        :param agg_state: Current aggregation state to update.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state for intermediate reporting.
        """
        if response is None:
            return None

        add_aggregate_metric(
            "queued_time",
            agg_state,
            request_info.scheduler_timings.dequeued,
            request_info.scheduler_timings.queued,
        )
        add_aggregate_metric(
            "worker_resolve_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.scheduler_timings.scheduled,
        )
        add_aggregate_metric(
            "worker_resolve_time",
            agg_state,
            request_info.scheduler_timings.resolve_end,
            request_info.scheduler_timings.resolve_start,
        )
        add_aggregate_metric(
            "worker_resolve_end_delay",
            agg_state,
            request_info.scheduler_timings.resolve_end,
            request_info.request_timings.request_end,
        )
        add_aggregate_metric(
            "finalized_delay",
            agg_state,
            request_info.scheduler_timings.finalized,
            request_info.scheduler_timings.resolve_end,
        )
        add_aggregate_metric(
            "worker_targeted_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.scheduler_timings.targeted_start,
        )
        add_aggregate_metric(
            "request_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.request_timings.request_start,
        )
        add_aggregate_metric(
            "request_time",
            agg_state,
            request_info.request_timings.request_end,
            request_info.request_timings.request_start,
        )
        add_aggregate_metric(
            "request_targeted_start_delay",
            agg_state,
            request_info.request_timings.request_start,
            request_info.scheduler_timings.targeted_start,
        )

        return agg_state

    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[Literal["scheduler_stats"], BenchmarkSchedulerStats]:
        """
        Compile scheduler timing metrics into benchmark statistics.

        :param agg_state: Accumulated timing data and counts.
        :param scheduler_state: Final scheduler execution state.
        :return: Dictionary containing compiled scheduler statistics.
        """
        return {
            "scheduler_stats": BenchmarkSchedulerStats(
                start_time=scheduler_state.start_time,
                end_time=scheduler_state.end_time,
                requests_made=StatusBreakdown(
                    successful=scheduler_state.successful_requests,
                    incomplete=scheduler_state.cancelled_requests,
                    errored=scheduler_state.errored_requests,
                ),
                queued_time_avg=(
                    agg_state.get("queued_time_total", 0.0)
                    / agg_state.get("queued_time_count", 1)
                ),
                worker_resolve_start_delay_avg=(
                    agg_state.get("worker_resolve_start_delay_total", 0.0)
                    / agg_state.get("worker_resolve_start_delay_count", 1)
                ),
                worker_resolve_time_avg=(
                    agg_state.get("worker_resolve_time_total", 0.0)
                    / agg_state.get("worker_resolve_time_count", 1)
                ),
                worker_resolve_end_delay_avg=(
                    agg_state.get("worker_resolve_end_delay_total", 0.0)
                    / agg_state.get("worker_resolve_end_delay_count", 1)
                ),
                finalized_delay_avg=(
                    agg_state.get("finalized_delay_total", 0.0)
                    / agg_state.get("finalized_delay_count", 1)
                ),
                worker_targeted_start_delay_avg=(
                    agg_state.get("worker_targeted_start_delay_total", 0.0)
                    / agg_state.get("worker_targeted_start_delay_count", 1)
                ),
                request_start_delay_avg=(
                    agg_state.get("request_start_delay_total", 0.0)
                    / agg_state.get("request_start_delay_count", 1)
                ),
                request_time_avg=(
                    agg_state.get("request_time_total", 0.0)
                    / agg_state.get("request_time_count", 1)
                ),
                request_targeted_delay_avg=(
                    agg_state.get("request_targeted_delay_total", 0.0)
                    / agg_state.get("request_targeted_delay_count", 1)
                ),
            ),
        }


class GenerativeRequestsStatsProgressAggregator(
    Aggregator[GenerationResponse, GenerationRequest, GenerationRequestTimings]
):
    """
    Tracks generative model metrics during benchmark execution.

    Aggregates token-level metrics including time to first token, inter-token
    latency, and token counts for real-time progress monitoring.
    """

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[GenerationResponse],
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        """
        Aggregate generative model metrics for a completed request.

        :param agg_state: Current aggregation state to update.
        :param response: Generation response with token and timing data.
        :param request: The processed generation request.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state for progress reporting.
        """
        if response is None:
            return None

        if (
            request_info.status == "completed"
            and request_info.request_timings.request_end is not None
        ):
            agg_state["requests_per_second"] = scheduler_state.successful_requests / (
                request_info.request_timings.request_end - scheduler_state.start_time
            )
            add_aggregate_metric(
                "request_latency",
                agg_state,
                request_info.request_timings.request_end,
                request_info.request_timings.request_start,
            )

        if (
            request_info.status == "completed"
            and request_info.request_timings.first_iteration is not None
            and request_info.request_timings.last_iteration is not None
            and response.output_tokens
        ):
            add_aggregate_metric(
                "time_per_output_token",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.request_start,
                count=response.output_tokens,
            )

        if (
            request_info.request_timings.first_iteration is not None
            and request_info.request_timings.request_start is not None
        ):
            add_aggregate_metric(
                "time_to_first_token",
                agg_state,
                request_info.request_timings.first_iteration,
                request_info.request_timings.request_start,
            )

        if (
            request_info.request_timings.first_iteration is not None
            and request_info.request_timings.last_iteration is not None
            and response.output_tokens is not None
            and response.output_tokens > 1
        ):
            add_aggregate_metric(
                "inter_token_latency",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.first_iteration,
                count=response.output_tokens - 1,
            )

        if response.prompt_tokens is not None:
            add_aggregate_metric(
                "prompt_tokens",
                agg_state,
                response.prompt_tokens,
            )
            if request_info.request_timings.request_end is not None:
                agg_state["prompt_tokens_per_second"] = agg_state[
                    "prompt_tokens_total"
                ] / (
                    request_info.request_timings.request_end
                    - scheduler_state.start_time
                )

        if response.output_tokens is not None:
            add_aggregate_metric(
                "output_tokens",
                agg_state,
                response.output_tokens,
            )
            if request_info.request_timings.request_end is not None:
                agg_state["output_tokens_per_second"] = agg_state[
                    "output_tokens_total"
                ] / (
                    request_info.request_timings.request_end
                    - scheduler_state.start_time
                )

        if response.total_tokens is not None:
            add_aggregate_metric(
                "total_tokens",
                agg_state,
                response.total_tokens,
            )
            if request_info.request_timings.request_end is not None:
                agg_state["total_tokens_per_second"] = agg_state[
                    "total_tokens_total"
                ] / (
                    request_info.request_timings.request_end
                    - scheduler_state.start_time
                )

        return agg_state


class GenerativeRequestsAggregator(
    StandardBaseModel,
    CompilableAggregator[
        GenerationResponse, GenerationRequest, GenerationRequestTimings
    ],
):
    """
    Compiles complete generative benchmark results with warmup/cooldown filtering.

    Aggregates request data during execution and compiles comprehensive metrics
    including timing distributions, token statistics, and throughput measurements.
    Supports filtering warmup and cooldown periods from final results.
    """

    warmup_requests: Optional[int] = Field(
        default=None,
        description="Number of warmup requests to ignore at benchmark start",
    )
    warmup_duration: Optional[float] = Field(
        default=None,
        description="Warmup duration in seconds to ignore at benchmark start",
    )
    cooldown_requests: Optional[int] = Field(
        default=None,
        description="Number of cooldown requests to ignore at benchmark end",
    )
    cooldown_duration: Optional[float] = Field(
        default=None,
        description="Cooldown duration in seconds to ignore at benchmark end",
    )
    _in_cooldown: bool = False
    _in_warmup: bool = False

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[GenerationResponse],
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        """
        Collect completed requests for final compilation.

        Filters requests based on warmup/cooldown settings and categorizes by
        completion status for comprehensive benchmark analysis.

        :param agg_state: Current aggregation state to update.
        :param response: Generation response data.
        :param request: The processed generation request.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: None, as this aggregator only collects for final compilation.
        """
        status = {
            "requests_in_warmup": False,
            "requests_in_cooldown": False,
        }

        if (
            response is None
            or request_info.status not in {"completed", "canceled", "errored"}
            or (request_info.status == "canceled" and request_info.started_at is None)
        ):
            # Ignore requests that don't have a response yet.
            # Ignore requests that were canceled before they started.
            return status

        if (
            self.warmup_requests is not None
            and self.warmup_requests >= scheduler_state.processed_requests
        ) or (
            self.warmup_duration is not None
            and request_info.request_timings.request_end is not None
            and (
                scheduler_state.start_time + self.warmup_duration
                >= request_info.request_timings.request_end
            )
        ):
            status["requests_in_warmup"] = True

            return status

        if (
            self.cooldown_requests is not None
            and scheduler_state.remaining_requests is not None
            and self.cooldown_requests >= scheduler_state.remaining_requests
        ) or (
            self.cooldown_duration is not None
            and scheduler_state.remaining_duration is not None
            and self.cooldown_duration >= scheduler_state.remaining_duration
        ):
            return status["requests_in_cooldown"]

        if "completed" not in agg_state:
            agg_state["completed"] = []
            agg_state["errored"] = []
            agg_state["incomplete"] = []

        if request_info.status == "completed":
            agg_state["completed"].append((response, request, request_info))
        elif request_info.status == "canceled":
            agg_state["incomplete"].append((response, request, request_info))
        else:
            agg_state["errored"].append((response, request, request_info))

        return status

    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile aggregated requests into comprehensive benchmark results.

        Transforms collected request data into detailed metrics including timing
        distributions, token statistics, throughput measurements, and status breakdowns.

        :param agg_state: Accumulated request data categorized by completion status.
        :param scheduler_state: Final scheduler execution state.
        :return: Complete benchmark results with metrics and request statistics.
        """
        successful: list[GenerativeRequestStats] = [
            self._create_generate_stats(response, request, request_info)
            for (response, request, request_info) in agg_state.get("completed", [])
        ]
        incomplete: list[GenerativeRequestStats] = [
            self._create_generate_stats(response, request, request_info)
            for (response, request, request_info) in agg_state.get("incomplete", [])
        ]
        errored: list[GenerativeRequestStats] = [
            self._create_generate_stats(response, request, request_info)
            for (response, request, request_info) in agg_state.get("errored", [])
        ]
        total: list[GenerativeRequestStats] = successful + incomplete + errored
        total_types = list[Literal["successful", "incomplete", "error"]] = [
            *["successful"] * len(successful),
            *["incomplete"] * len(incomplete),
            *["error"] * len(errored),
        ]
        start_time = min(
            [math.inf]
            + [
                req.scheduler_info.request_timings.request_start
                for req in total
                if req.scheduler_info.request_timings.request_start is not None
            ]
        )
        end_time = max(
            [-1 * math.inf]
            + [
                req.scheduler_info.request_timings.request_end
                for req in total
                if req.scheduler_info.request_timings.request_end is not None
            ]
        )

        return {
            "start_time": start_time,
            "end_time": end_time,
            "request_totals": StatusBreakdown(
                successful=len(successful),
                incomplete=len(incomplete),
                errored=len(errored),
                total=len(total),
            ),
            "requests": StatusBreakdown(
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            "metrics": GenerativeMetrics(
                requests_per_second=(
                    StatusDistributionSummary.from_request_times(
                        request_types=total_types,
                        requests=[
                            (
                                req.scheduler_info.request_timings.request_start,
                                req.scheduler_info.request_timings.request_end,
                            )
                            for req in total
                            if (
                                req.scheduler_info.request_timings.request_start
                                is not None
                                and req.scheduler_info.request_timings.request_end
                                is not None
                            )
                        ],
                        distribution_type="rate",
                    )
                ),
                request_concurrency=(
                    StatusDistributionSummary.from_request_times(
                        request_types=total_types,
                        requests=[
                            (
                                req.scheduler_info.request_timings.request_start,
                                req.scheduler_info.request_timings.request_end,
                            )
                            for req in total
                            if (
                                req.scheduler_info.request_timings.request_start
                                is not None
                                and req.scheduler_info.request_timings.request_end
                                is not None
                            )
                        ],
                        distribution_type="concurrency",
                    )
                ),
                request_latency=(
                    StatusDistributionSummary.from_values(
                        value_types=total_types,
                        values=[
                            req.request_latency
                            for req in total
                            if req.request_latency is not None
                        ],
                    )
                ),
                prompt_token_count=(
                    StatusDistributionSummary.from_values(
                        value_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.prompt_tokens is not None
                        ],
                        values=[
                            req.prompt_tokens
                            for req in total
                            if req.prompt_tokens is not None
                        ],
                    )
                ),
                output_token_count=(
                    StatusDistributionSummary.from_values(
                        value_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.output_tokens is not None
                        ],
                        values=[
                            req.output_tokens
                            for req in total
                            if req.output_tokens is not None
                        ],
                    )
                ),
                total_token_count=(
                    StatusDistributionSummary.from_values(
                        value_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.prompt_tokens is not None
                            or req.output_tokens is not None
                        ],
                        values=(
                            (req.prompt_tokens or 0) + (req.output_tokens or 0)
                            for req in total
                            if req.prompt_tokens is not None
                            or req.output_tokens is not None
                        ),
                    )
                ),
                time_to_first_token_ms=(
                    StatusDistributionSummary.from_values(
                        value_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.time_to_first_token_ms is not None
                        ],
                        values=[
                            req.time_to_first_token_ms
                            for req in total
                            if req.time_to_first_token_ms is not None
                        ],
                    )
                ),
                time_per_output_token_ms=(
                    StatusDistributionSummary.from_values(
                        value_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.time_per_output_token_ms is not None
                        ],
                        values=[
                            req.time_per_output_token_ms
                            for req in total
                            if req.time_per_output_token_ms is not None
                        ],
                        weights=[
                            req.output_tokens
                            for req in total
                            if req.time_per_output_token_ms is not None
                        ],
                    )
                ),
                inter_token_latency_ms=(
                    StatusDistributionSummary.from_values(
                        value_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.inter_token_latency_ms is not None
                        ],
                        values=[
                            req.inter_token_latency_ms
                            for req in total
                            if req.inter_token_latency_ms is not None
                        ],
                        weights=[
                            req.output_tokens - 1
                            for req in total
                            if req.inter_token_latency_ms is not None
                        ],
                    )
                ),
                output_tokens_per_second=(
                    StatusDistributionSummary.from_iterable_request_times(
                        request_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.output_tokens_per_second is not None
                        ],
                        requests=[
                            (
                                req.scheduler_info.request_timings.request_start,
                                req.scheduler_info.request_timings.request_end,
                            )
                            for req in total
                            if req.output_tokens_per_second is not None
                        ],
                        first_iter_times=[
                            req.scheduler_info.request_timings.first_iteration
                            for req in total
                            if req.output_tokens_per_second is not None
                            and req.scheduler_info.request_timings.first_iteration
                            is not None
                        ],
                        iter_counts=[
                            req.output_tokens
                            for req in total
                            if req.output_tokens_per_second is not None
                            and req.output_tokens is not None
                        ],
                    )
                ),
                tokens_per_second=(
                    StatusDistributionSummary.from_iterable_request_times(
                        request_types=[
                            type_
                            for type_, req in zip(total_types, total)
                            if req.tokens_per_second is not None
                        ],
                        requests=[
                            (
                                req.scheduler_info.request_timings.request_start,
                                req.scheduler_info.request_timings.request_end,
                            )
                            for req in total
                            if req.tokens_per_second is not None
                        ],
                        first_iter_times=[
                            req.scheduler_info.request_timings.first_iteration
                            for req in total
                            if req.tokens_per_second is not None
                        ],
                        iter_counts=[
                            req.output_tokens
                            for req in total
                            if req.tokens_per_second is not None
                        ],
                        first_iter_counts=[
                            req.prompt_tokens
                            for req in total
                            if req.tokens_per_second is not None
                        ],
                    )
                ),
            ),
        }

    @classmethod
    def _create_generate_stats(
        cls,
        response: GenerationResponse,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
    ) -> GenerativeRequestStats:
        prompt_tokens = response.preferred_prompt_tokens(
            settings.preferred_prompt_tokens_source
        )
        output_tokens = response.preferred_output_tokens(
            settings.preferred_output_tokens_source
        )

        return GenerativeRequestStats(
            request_id=request.request_id,
            request_type=request.request_type,
            prompt=str(request.content),
            request_args=response.request_args,
            output=response.value,
            iterations=response.iterations,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                prompt_tokens + output_tokens
                if prompt_tokens is not None and output_tokens is not None
                else None
            ),
            scheduler_info=request_info,
        )
