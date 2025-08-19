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

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    runtime_checkable,
)

import numpy
from pydantic import Field, PrivateAttr

from guidellm.backend import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.benchmark.objects import (
    BenchmarkSchedulerStats,
    GenerativeMetrics,
    GenerativeRequestStats,
)
from guidellm.config import settings
from guidellm.scheduler import (
    MeasuredRequestTimingsT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.utils import (
    InfoMixin,
    PydanticClassRegistryMixin,
    StatusBreakdown,
    StatusDistributionSummary,
    all_defined,
    safe_divide,
    safe_getattr,
    safe_subtract,
)

__all__ = [
    "Aggregator",
    "CompilableAggregator",
    "GenerativeRequestsAggregator",
    "GenerativeStatsProgressAggregator",
    "SchedulerStatsAggregator",
    "SerializableAggregator",
]


@runtime_checkable
class Aggregator(Protocol[ResponseT, RequestT, MeasuredRequestTimingsT]):
    """
    Protocol for processing benchmark data updates during execution.

    Defines the interface for aggregators that collect and process request/response
    data from scheduler executions. Implementations update aggregation state with
    each completed request for eventual compilation into final metrics.
    """

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Process a completed request and update aggregation state.

        :param agg_state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
        """


@runtime_checkable
class CompilableAggregator(Protocol[ResponseT, RequestT, MeasuredRequestTimingsT]):
    """
    Protocol for aggregators that compile final results from aggregated state.

    Extends the Aggregator protocol with the ability to transform accumulated
    state into final benchmark results and metrics after execution completes.
    """

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Process a completed request and update aggregation state.

        :param agg_state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
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


class SerializableAggregator(
    PydanticClassRegistryMixin[type["SerializableAggregator"]],
    ABC,
    Generic[ResponseT, RequestT, MeasuredRequestTimingsT],
):
    schema_discriminator: ClassVar[str] = "type_"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[SerializableAggregator]:
        if cls.__name__ == "SerializableAggregator":
            return cls

        return SerializableAggregator

    @classmethod
    def add_aggregate_metric(
        cls,
        base_key: str,
        agg_state: dict[str, Any],
        end_val: int | float | None,
        start_val: int | float | None = 0.0,
        count: int = 1,
    ) -> int | float | None:
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
        if not all_defined(end_val, start_val):
            return None

        delta_val = end_val - start_val
        agg_state[f"{base_key}_total"] = (
            agg_state.get(f"{base_key}_total", 0) + delta_val
        )
        agg_state[f"{base_key}_count"] = agg_state.get(f"{base_key}_count", 0) + count

        return agg_state[f"{base_key}_total"]

    @classmethod
    def add_aggregate_metric_rate(
        cls, base_key: str, agg_state: dict[str, Any]
    ) -> float:
        """
        Calculate the rate of a metric by dividing the total by the count.

        :param base_key: Base key name for the metric.
        :param agg_state: Aggregation state dictionary to update.
        """
        agg_state[f"{base_key}_rate"] = safe_divide(
            agg_state.get(f"{base_key}_total"), agg_state.get(f"{base_key}_count")
        )

        return agg_state[f"{base_key}_rate"]

    @classmethod
    @abstractmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for constraint creation.

        Must be implemented by subclasses to handle their specific parameter patterns.

        :param args: Positional arguments passed to the constraint
        :param kwargs: Keyword arguments passed to the constraint
        :return: Validated dictionary of parameters for constraint creation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...

    @classmethod
    def resolve(
        cls,
        aggregators: dict[
            str,
            Any | dict[str, Any] | Aggregator | CompilableAggregator,
        ],
    ) -> dict[str, Aggregator | CompilableAggregator]:
        """
        Resolve mixed aggregator specifications to callable aggregators.

        :param aggregators: Dictionary mapping aggregator keys to specifications
        :return: Dictionary mapping aggregator keys to callable functions
        :raises ValueError: If any key is not registered in the factory
        """
        resolved = {}

        for key, val in aggregators.items():
            if isinstance(val, (Aggregator, CompilableAggregator)):
                resolved[key] = val
            else:
                aggregator_class = cls.get_registered_object(key)
                kwargs = aggregator_class.validated_kwargs(**val)
                resolved[key] = aggregator_class(**kwargs)

        return resolved

    type_: Literal["aggregator"] = Field(default="aggregator", description="")

    @abstractmethod
    def __call__(
        self,
        agg_state: dict[str, Any],
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Process a completed request and update aggregation state.

        :param agg_state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
        """

    @abstractmethod
    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile aggregated state into final benchmark results.

        :param agg_state: The accumulated aggregation state.
        :param scheduler_state: Final scheduler execution state.
        :return: Compiled benchmark results and metrics.
        """


@SerializableAggregator.register("scheduler_stats")
class SchedulerStatsAggregator(
    SerializableAggregator[ResponseT, RequestT, MeasuredRequestTimingsT], InfoMixin
):
    """
    Aggregates scheduler timing and performance metrics.

    Collects timing data for various scheduler phases including queuing,
    resolution, and processing delays to generate performance statistics.
    """

    @classmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        return {}

    type_: Literal["scheduler_stats"] = Field(default="scheduler_stats")

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Aggregate scheduler timing metrics for a completed request.

        :param agg_state: Current aggregation state to update.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state for intermediate reporting.
        """
        if request_info.status not in ("completed", "errored", "cancelled"):
            # Only compile scheduler stats for processed requests
            return None

        self.add_aggregate_metric(
            "queued_time",
            agg_state,
            request_info.scheduler_timings.dequeued,
            request_info.scheduler_timings.queued,
        )
        self.add_aggregate_metric(
            "worker_resolve_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.scheduler_timings.scheduled_at,
        )
        self.add_aggregate_metric(
            "worker_resolve_time",
            agg_state,
            request_info.scheduler_timings.resolve_end,
            request_info.scheduler_timings.resolve_start,
        )
        self.add_aggregate_metric(
            "worker_resolve_end_delay",
            agg_state,
            request_info.scheduler_timings.resolve_end,
            safe_getattr(request_info.request_timings, "request_end"),
        )
        self.add_aggregate_metric(
            "finalized_delay",
            agg_state,
            request_info.scheduler_timings.finalized,
            request_info.scheduler_timings.resolve_end,
        )
        self.add_aggregate_metric(
            "worker_targeted_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.scheduler_timings.targeted_start,
        )
        self.add_aggregate_metric(
            "request_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            safe_getattr(request_info.request_timings, "request_start"),
        )
        self.add_aggregate_metric(
            "request_time",
            agg_state,
            safe_getattr(request_info.request_timings, "request_end"),
            safe_getattr(request_info.request_timings, "request_start"),
        )
        self.add_aggregate_metric(
            "request_targeted_start_delay",
            agg_state,
            safe_getattr(request_info.request_timings, "request_start"),
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
            "run_stats": BenchmarkSchedulerStats(
                start_time=scheduler_state.start_time,
                end_time=scheduler_state.end_time,
                requests_made=StatusBreakdown[int, int, int, int](
                    successful=scheduler_state.successful_requests,
                    incomplete=scheduler_state.cancelled_requests,
                    errored=scheduler_state.errored_requests,
                    total=(
                        scheduler_state.successful_requests
                        + scheduler_state.cancelled_requests
                        + scheduler_state.errored_requests
                    ),
                ),
                queued_time_avg=self.add_aggregate_metric_rate(
                    "queued_time", agg_state
                ),
                worker_resolve_start_delay_avg=self.add_aggregate_metric_rate(
                    "worker_resolve_start_delay", agg_state
                ),
                worker_resolve_time_avg=self.add_aggregate_metric_rate(
                    "worker_resolve_time", agg_state
                ),
                worker_resolve_end_delay_avg=self.add_aggregate_metric_rate(
                    "worker_resolve_end_delay", agg_state
                ),
                finalized_delay_avg=self.add_aggregate_metric_rate(
                    "finalized_delay", agg_state
                ),
                worker_targeted_start_delay_avg=self.add_aggregate_metric_rate(
                    "worker_targeted_start_delay", agg_state
                ),
                request_start_delay_avg=self.add_aggregate_metric_rate(
                    "request_start_delay", agg_state
                ),
                request_time_avg=self.add_aggregate_metric_rate(
                    "request_time", agg_state
                ),
                request_targeted_delay_avg=self.add_aggregate_metric_rate(
                    "request_targeted_delay", agg_state
                ),
            ),
        }


@SerializableAggregator.register("generative_stats_progress")
class GenerativeStatsProgressAggregator(
    SerializableAggregator[
        GenerationResponse, GenerationRequest, GenerationRequestTimings
    ]
):
    """
    Tracks generative model metrics during benchmark execution.

    Aggregates token-level metrics including time to first token, inter-token
    latency, and token counts for real-time progress monitoring.
    """

    @classmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        return {}

    type_: Literal["generative_stats_progress"] = Field(
        default="generative_stats_progress"
    )

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Aggregate generative model metrics for a completed request.

        :param agg_state: Current aggregation state to update.
        :param response: Generation response with token and timing data.
        :param request: The processed generation request.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state for progress reporting.
        """
        if request_info.status == "completed":
            prefix = "completed_"
        elif request_info.status == "errored":
            prefix = "errored_"
        elif request_info.status == "cancelled":
            prefix = "cancelled_"
        else:
            # Only compile progress stats for processed requests
            return None

        start_time = scheduler_state.start_time
        end_time = (
            safe_getattr(request_info.request_timings, "request_end")
            or request_info.scheduler_timings.resolve_end
        )
        duration = safe_subtract(end_time, start_time)

        if all_defined(end_time):
            # request rate
            requests = (
                scheduler_state.successful_requests
                if request_info.status == "completed"
                else scheduler_state.cancelled_requests
                if request_info.status == "cancelled"
                else scheduler_state.errored_requests
            )
            agg_state[f"{prefix}requests_per_second"] = safe_divide(requests, duration)
            agg_state["requests_per_second"] = safe_divide(
                scheduler_state.processed_requests, duration
            )

        if all_defined(
            safe_getattr(request_info.request_timings, "request_end"),
            safe_getattr(request_info.request_timings, "request_start"),
        ):
            # request latency
            self.add_aggregate_metric(
                f"{prefix}request_latency",
                agg_state,
                request_info.request_timings.request_end,
                request_info.request_timings.request_start,
            )
            self.add_aggregate_metric_rate(f"{prefix}request_latency", agg_state)
            self.add_aggregate_metric(
                "request_latency",
                agg_state,
                request_info.request_timings.request_end,
                request_info.request_timings.request_start,
            )
            self.add_aggregate_metric_rate("request_latency", agg_state)

        if all_defined(
            safe_getattr(request_info.request_timings, "last_iteration"),
            safe_getattr(request_info.request_timings, "request_start"),
        ):
            # TPOT
            self.add_aggregate_metric(
                f"{prefix}time_per_output_token",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.request_start,
            )
            self.add_aggregate_metric_rate(f"{prefix}time_per_output_token", agg_state)
            self.add_aggregate_metric(
                "time_per_output_token",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.request_start,
            )
            self.add_aggregate_metric_rate("time_per_output_token", agg_state)

        if all_defined(
            safe_getattr(request_info.request_timings, "first_iteration"),
            safe_getattr(request_info.request_timings, "request_start"),
        ):
            # TTFT
            self.add_aggregate_metric(
                f"{prefix}time_to_first_token",
                agg_state,
                request_info.request_timings.first_iteration,
                request_info.request_timings.request_start,
            )
            self.add_aggregate_metric_rate(f"{prefix}time_to_first_token", agg_state)
            self.add_aggregate_metric(
                "time_to_first_token",
                agg_state,
                request_info.request_timings.first_iteration,
                request_info.request_timings.request_start,
            )
            self.add_aggregate_metric_rate("time_to_first_token", agg_state)

        if (
            all_defined(
                safe_getattr(request_info.request_timings, "first_iteration"),
                safe_getattr(request_info.request_timings, "last_iteration"),
                safe_getattr(response, "output_tokens"),
            )
            and response.output_tokens > 1
        ):
            # ITL
            self.add_aggregate_metric(
                f"{prefix}inter_token_latency",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.first_iteration,
                count=response.output_tokens - 1,
            )
            self.add_aggregate_metric_rate(f"{prefix}inter_token_latency", agg_state)
            self.add_aggregate_metric(
                "inter_token_latency",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.first_iteration,
                count=response.output_tokens - 1,
            )
            self.add_aggregate_metric_rate("inter_token_latency", agg_state)

        if all_defined(safe_getattr(response, "prompt_tokens")):
            # Prompt tokens totals
            self.add_aggregate_metric(
                f"{prefix}prompt_tokens", agg_state, response.prompt_tokens
            )
            agg_state[f"{prefix}prompt_tokens_per_request"] = (
                self.add_aggregate_metric_rate(f"{prefix}prompt_tokens", agg_state)
            )
            self.add_aggregate_metric(
                f"{prefix}prompt_tokens", agg_state, response.prompt_tokens
            )
            agg_state["prompt_tokens_per_request"] = self.add_aggregate_metric_rate(
                "prompt_tokens", agg_state
            )

            if all_defined(end_time):
                # Prompt tokens rate
                agg_state[f"{prefix}prompt_tokens_rate"] = safe_divide(
                    agg_state[f"{prefix}prompt_tokens_total"], duration
                )
                agg_state["prompt_tokens_rate"] = safe_divide(
                    agg_state["prompt_tokens_total"], duration
                )

        if all_defined(safe_getattr(response, "output_tokens")):
            # Output tokens totals
            self.add_aggregate_metric(
                f"{prefix}output_tokens", agg_state, response.output_tokens
            )
            agg_state[f"{prefix}output_tokens_per_request"] = (
                self.add_aggregate_metric_rate(f"{prefix}output_tokens", agg_state)
            )
            self.add_aggregate_metric(
                "output_tokens", agg_state, response.output_tokens
            )
            agg_state["output_tokens_per_request"] = self.add_aggregate_metric_rate(
                "output_tokens", agg_state
            )

            if all_defined(end_time):
                # Output tokens rate
                agg_state[f"{prefix}output_tokens_rate"] = safe_divide(
                    agg_state[f"{prefix}output_tokens_total"], duration
                )
                agg_state["output_tokens_rate"] = safe_divide(
                    agg_state["output_tokens_total"], duration
                )

        if all_defined(safe_getattr(response, "total_tokens")):
            # Total tokens totals
            self.add_aggregate_metric(
                f"{prefix}total_tokens", agg_state, response.total_tokens
            )
            agg_state[f"{prefix}total_tokens_per_request"] = (
                self.add_aggregate_metric_rate(f"{prefix}total_tokens", agg_state)
            )
            self.add_aggregate_metric("total_tokens", agg_state, response.total_tokens)
            agg_state["total_tokens_per_request"] = self.add_aggregate_metric_rate(
                "total_tokens", agg_state
            )

            if all_defined(end_time):
                # Total tokens rate
                agg_state[f"{prefix}total_tokens_rate"] = safe_divide(
                    agg_state[f"{prefix}total_tokens_total"], duration
                )
                agg_state["total_tokens_rate"] = safe_divide(
                    agg_state["total_tokens_total"], duration
                )

        return agg_state

    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile progress metrics into final results.

        GenerativeStatsProgressAggregator is primarily for progress tracking,
        so compilation returns the aggregated state as-is.

        :param agg_state: The accumulated aggregation state.
        :param scheduler_state: Final scheduler execution state.
        :return: The aggregated state as final results.
        """
        return {}


@SerializableAggregator.register("generative_requests")
class GenerativeRequestsAggregator(
    SerializableAggregator[
        GenerationResponse, GenerationRequest, GenerationRequestTimings
    ],
):
    """
    Compiles complete generative benchmark results with warmup/cooldown filtering.

    Aggregates request data during execution and compiles comprehensive metrics
    including timing distributions, token statistics, and throughput measurements.
    Supports filtering warmup and cooldown periods from final results.
    """

    @classmethod
    def validated_kwargs(
        cls,
        request_samples: int | None = 20,
        warmup: int | float | None = None,
        cooldown: int | float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "request_samples": request_samples,
            "warmup": warmup,
            "cooldown": cooldown,
        }

    type_: Literal["generative_requests"] = Field(default="generative_requests")

    request_samples: int | None = Field(default=20, description="")
    warmup: int | float | None = Field(
        default=None,
        description="Number of warmup requests to ignore at benchmark start",
    )
    cooldown: int | float | None = Field(
        default=None,
        description="Number of cooldown requests to ignore at benchmark end",
    )
    _in_cooldown: bool = PrivateAttr(False)
    _in_warmup: bool = PrivateAttr(False)

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
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

        # Skip invalid requests
        if request_info.status not in {"completed", "canceled", "errored"} or (
            request_info.status == "canceled"
            and safe_getattr(request_info.scheduler_timings, "resolve_start") is None
            # Canceled requests that never started should not be kept
        ):
            return status

        if self._is_in_warmup(request_info, scheduler_state):
            status["requests_in_warmup"] = True
            return status

        if self._is_in_cooldown(request_info, scheduler_state):
            status["requests_in_cooldown"] = True
            return status

        if "completed" not in agg_state:
            agg_state["completed"] = []
            agg_state["errored"] = []
            agg_state["incomplete"] = []

        # Categorize request by status
        if request_info.status == "completed":
            agg_state["completed"].append((response, request, request_info))
        elif request_info.status == "canceled":
            agg_state["incomplete"].append((response, request, request_info))
        else:
            agg_state["errored"].append((response, request, request_info))

        return status

    def compile(
        self,
        agg_state: dict[str, Any],
        scheduler_state: SchedulerState,  # noqa: ARG002
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

        # Use all requests for metrics calculations (not sampled)
        total: list[GenerativeRequestStats] = successful + incomplete + errored
        total_types: list[Literal["successful", "incomplete", "error"]] = [
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
                successful=(
                    list(
                        numpy.random.choice(
                            successful, size=self.request_samples, replace=False
                        )
                    )
                    if self.request_samples
                    else successful
                ),
                incomplete=(
                    list(
                        numpy.random.choice(
                            incomplete, size=self.request_samples, replace=False
                        )
                    )
                    if self.request_samples
                    else incomplete
                ),
                errored=(
                    list(
                        numpy.random.choice(
                            errored, size=self.request_samples, replace=False
                        )
                    )
                    if self.request_samples
                    else errored
                ),
            ),
            "metrics": GenerativeMetrics(
                requests_per_second=self._calculate_requests_per_second(
                    statuses=total_types, requests=total
                ),
                request_concurrency=self._calculate_request_concurrency(
                    statuses=total_types, requests=total
                ),
                request_latency=self._calculate_request_latency(
                    statuses=total_types, requests=total
                ),
                prompt_token_count=self._calculate_prompt_token_count(
                    statuses=total_types, requests=total
                ),
                output_token_count=self._calculate_output_token_count(
                    statuses=total_types, requests=total
                ),
                total_token_count=self._calculate_total_token_count(
                    statuses=total_types, requests=total
                ),
                time_to_first_token_ms=self._calculate_time_to_first_token_ms(
                    statuses=total_types, requests=total
                ),
                time_per_output_token_ms=self._calculate_time_per_output_token_ms(
                    statuses=total_types, requests=total
                ),
                inter_token_latency_ms=self._calculate_inter_token_latency_ms(
                    statuses=total_types, requests=total
                ),
                output_tokens_per_second=self._calculate_output_tokens_per_second(
                    statuses=total_types, requests=total
                ),
                tokens_per_second=self._calculate_tokens_per_second(
                    statuses=total_types, requests=total
                ),
            ),
        }

    def _is_in_warmup(
        self,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> bool:
        """Check if the current request is within the warmup period."""
        if self.warmup is None:
            return False

        if 0 < self.warmup < 1:  # Percentage-based warmup
            return (
                scheduler_state.remaining_fraction is not None
                and scheduler_state.remaining_fraction > (1 - self.warmup)
            )

        if self.warmup >= 1:  # Count/time-based warmup
            if scheduler_state.processed_requests < self.warmup:
                return True

            current_time = request_info.scheduler_timings.targeted_start
            return (
                current_time is not None
                and (current_time - scheduler_state.start_time) < self.warmup
            )

        return False

    def _is_in_cooldown(
        self,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> bool:
        """Check if the current request is within the cooldown period."""
        if self.cooldown is None:
            return False

        if 0 < self.cooldown < 1:  # Percentage-based cooldown
            return (
                scheduler_state.remaining_fraction is not None
                and scheduler_state.remaining_fraction < self.cooldown
            )

        if self.cooldown >= 1:  # Count/time-based cooldown
            if scheduler_state.remaining_requests < self.cooldown:
                return True

            current_time = (
                request_info.scheduler_timings.resolve_end
                or request_info.scheduler_timings.targeted_start
            )
            return (
                current_time is not None
                and scheduler_state.remaining_duration is not None
                and scheduler_state.remaining_duration < self.cooldown
            )

        return False

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

    @classmethod
    def _calculate_requests_per_second(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_times = []

        for status, request in zip(statuses, requests):
            if not all_defined(
                safe_getattr(request.scheduler_info.request_timings, "request_start"),
                safe_getattr(request.scheduler_info.request_timings, "request_end"),
            ):
                continue

            filtered_statuses.append(status)
            filtered_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )

        return StatusDistributionSummary.from_request_times(
            request_types=filtered_statuses,
            requests=filtered_times,
            distribution_type="rate",
        )

    @classmethod
    def _calculate_request_concurrency(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_times = []

        for status, request in zip(statuses, requests):
            if not all_defined(
                safe_getattr(request.scheduler_info.request_timings, "request_start"),
                safe_getattr(request.scheduler_info.request_timings, "request_end"),
            ):
                continue

            filtered_statuses.append(status)
            filtered_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )

        return StatusDistributionSummary.from_request_times(
            request_types=filtered_statuses,
            requests=filtered_times,
            distribution_type="concurrency",
        )

    @classmethod
    def _calculate_request_latency(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.request_latency):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.request_latency)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_prompt_token_count(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.prompt_tokens):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.prompt_tokens)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_output_token_count(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.output_tokens):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.output_tokens)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_total_token_count(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.total_tokens):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.total_tokens)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_time_to_first_token_ms(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.time_to_first_token_ms):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.time_to_first_token_ms)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_time_per_output_token_ms(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []
        filtered_weights = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.time_to_first_token_ms):
                continue

            # Add time to first token separately to better reflect in distribution
            filtered_statuses.append(status)
            filtered_values.append(request.time_to_first_token_ms)
            filtered_weights.append(1)

            if not all_defined(request.inter_token_latency_ms):
                continue

            # Add tokens after the first token to get the full distribution
            filtered_statuses.append(status)
            filtered_values.append(request.inter_token_latency_ms)
            filtered_weights.append(request.output_tokens - 1)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
            weights=filtered_weights,
        )

    @classmethod
    def _calculate_inter_token_latency_ms(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []
        filtered_weights = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.inter_token_latency_ms):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.inter_token_latency_ms)
            filtered_weights.append(request.output_tokens - 1)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
            weights=filtered_weights,
        )

    @classmethod
    def _calculate_output_tokens_per_second(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_request_times = []
        filtered_first_iter_times = []
        filtered_iter_counts = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.output_tokens_per_second):
                continue

            filtered_statuses.append(status)
            filtered_request_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )
            filtered_first_iter_times.append(
                request.scheduler_info.request_timings.first_iteration
            )
            filtered_iter_counts.append(request.output_tokens)

        return StatusDistributionSummary.from_iterable_request_times(
            request_types=filtered_statuses,
            requests=filtered_request_times,
            first_iter_times=filtered_first_iter_times,
            iter_counts=filtered_iter_counts,
        )

    @classmethod
    def _calculate_tokens_per_second(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_request_times = []
        filtered_first_iter_times = []
        filtered_iter_counts = []
        filtered_first_iter_counts = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.tokens_per_second):
                continue

            filtered_statuses.append(status)
            filtered_request_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )
            filtered_first_iter_times.append(
                request.scheduler_info.request_timings.first_iteration
            )
            filtered_iter_counts.append(request.output_tokens - 1)
            filtered_first_iter_counts.append(request.prompt_tokens + 1)

        return StatusDistributionSummary.from_iterable_request_times(
            request_types=filtered_statuses,
            requests=filtered_request_times,
            first_iter_times=filtered_first_iter_times,
            iter_counts=filtered_iter_counts,
            first_iter_counts=filtered_first_iter_counts,
        )
