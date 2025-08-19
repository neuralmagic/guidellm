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

# TODO: Review Cursor generated code (start)
from loguru import logger

# TODO: Review Cursor generated code (end)
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
        start_val: int | float | None,
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
        agg_state[f"{base_key}_total"] = (
            agg_state.get(f"{base_key}_total", 0) + delta_val
        )
        agg_state[f"{base_key}_count"] = agg_state.get(f"{base_key}_count", 0) + count

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
        if response is None:
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
            # TODO: Review Cursor generated code (start)
            request_info.scheduler_timings.scheduled_at,
            # TODO: Review Cursor generated code (end)
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
            request_info.request_timings.request_end,
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
            request_info.request_timings.request_start,
        )
        self.add_aggregate_metric(
            "request_time",
            agg_state,
            request_info.request_timings.request_end,
            request_info.request_timings.request_start,
        )
        self.add_aggregate_metric(
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
                    total=(
                        scheduler_state.successful_requests
                        + scheduler_state.cancelled_requests
                        + scheduler_state.errored_requests
                    ),
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
        if response is None:
            return None

        if (
            request_info.status == "completed"
            and request_info.request_timings.request_end is not None
        ):
            agg_state["requests_per_second"] = scheduler_state.successful_requests / (
                request_info.request_timings.request_end - scheduler_state.start_time
            )
            self.add_aggregate_metric(
                "request_latency",
                agg_state,
                request_info.request_timings.request_end,
                request_info.request_timings.request_start,
            )

        if (
            request_info.status == "completed"
            # TODO: Review Cursor generated code (start)
            and request_info.request_timings is not None
            and hasattr(request_info.request_timings, "first_iteration")
            # TODO: Review Cursor generated code (end)
            and request_info.request_timings.first_iteration is not None
            # TODO: Review Cursor generated code (start)
            and hasattr(request_info.request_timings, "last_iteration")
            # TODO: Review Cursor generated code (end)
            and request_info.request_timings.last_iteration is not None
            and response.output_tokens
        ):
            self.add_aggregate_metric(
                "time_per_output_token",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.request_start,
                count=response.output_tokens,
            )

        if (
            # TODO: Review Cursor generated code (start)
            request_info.request_timings is not None
            and hasattr(request_info.request_timings, "first_iteration")
            and request_info.request_timings.first_iteration is not None
            # TODO: Review Cursor generated code (end)
            and request_info.request_timings.request_start is not None
        ):
            self.add_aggregate_metric(
                "time_to_first_token",
                agg_state,
                request_info.request_timings.first_iteration,
                request_info.request_timings.request_start,
            )

        if (
            # TODO: Review Cursor generated code (start)
            request_info.request_timings is not None
            and hasattr(request_info.request_timings, "first_iteration")
            and request_info.request_timings.first_iteration is not None
            and hasattr(request_info.request_timings, "last_iteration")
            # TODO: Review Cursor generated code (end)
            and request_info.request_timings.last_iteration is not None
            and response.output_tokens is not None
            and response.output_tokens > 1
        ):
            self.add_aggregate_metric(
                "inter_token_latency",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.first_iteration,
                count=response.output_tokens - 1,
            )

        if response.prompt_tokens is not None:
            self.add_aggregate_metric(
                "prompt_tokens",
                agg_state,
                response.prompt_tokens,
                0,
            )
            if request_info.request_timings.request_end is not None:
                agg_state["prompt_tokens_per_second"] = agg_state[
                    "prompt_tokens_total"
                ] / (
                    request_info.request_timings.request_end
                    - scheduler_state.start_time
                )

        if response.output_tokens is not None:
            self.add_aggregate_metric(
                "output_tokens",
                agg_state,
                response.output_tokens,
                0,
            )
            if request_info.request_timings.request_end is not None:
                agg_state["output_tokens_per_second"] = agg_state[
                    "output_tokens_total"
                ] / (
                    request_info.request_timings.request_end
                    - scheduler_state.start_time
                )

        if response.total_tokens is not None:
            self.add_aggregate_metric(
                "total_tokens",
                agg_state,
                response.total_tokens,
                0,
            )
            if request_info.request_timings.request_end is not None:
                agg_state["total_tokens_per_second"] = agg_state[
                    "total_tokens_total"
                ] / (
                    request_info.request_timings.request_end
                    - scheduler_state.start_time
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
        if (
            response is None
            or request_info.status not in {"completed", "canceled", "errored"}
            or (
                request_info.status == "canceled"
                and request_info.scheduler_timings.resolve_start is None
            )
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
        # TODO: Review Cursor generated code (start)
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: agg_state keys: {list(agg_state.keys())}"
        )
        completed_data = agg_state.get("completed", [])
        incomplete_data = agg_state.get("incomplete", [])
        errored_data = agg_state.get("errored", [])
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: completed={len(completed_data)}, incomplete={len(incomplete_data)}, errored={len(errored_data)}"
        )
        # TODO: Review Cursor generated code (end)

        successful: list[GenerativeRequestStats] = [
            self._create_generate_stats(response, request, request_info)
            # TODO: Review Cursor generated code (start)
            for (response, request, request_info) in completed_data
            # TODO: Review Cursor generated code (end)
        ]
        # TODO: Review Cursor generated code (start)
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: Created {len(successful)} successful request stats"
        )
        # TODO: Review Cursor generated code (end)
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

        # TODO: Review Cursor generated code (start)
        # Debug logging before StatusBreakdown creation
        successful_requests = (
            (
                list(
                    numpy.random.choice(
                        successful, size=self.request_samples, replace=False
                    )
                )
                if self.request_samples is not None
                and len(successful) >= self.request_samples
                else successful
            )
            if successful
            else []
        )
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        incomplete_requests = (
            (
                list(
                    numpy.random.choice(
                        incomplete, size=self.request_samples, replace=False
                    )
                )
                if self.request_samples is not None
                and len(incomplete) >= self.request_samples
                else incomplete
            )
            if incomplete
            else []
        )
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        errored_requests = (
            (
                list(
                    numpy.random.choice(
                        errored, size=self.request_samples, replace=False
                    )
                )
                if self.request_samples is not None
                and len(errored) >= self.request_samples
                else errored
            )
            if errored
            else []
        )
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Rebuild total and total_types from sampled lists to match for metrics calculations
        total: list[GenerativeRequestStats] = (
            successful_requests + incomplete_requests + errored_requests
        )
        total_types: list[Literal["successful", "incomplete", "error"]] = [
            *["successful"] * len(successful_requests),
            *["incomplete"] * len(incomplete_requests),
            *["error"] * len(errored_requests),
        ]
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: About to create StatusBreakdown with successful={len(successful_requests)}, incomplete={len(incomplete_requests)}, errored={len(errored_requests)}"
        )
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: request_samples={self.request_samples}"
        )
        if successful_requests:
            logger.debug(
                f"DEBUG GenerativeRequestsAggregator.compile: First successful request type: {type(successful_requests[0])}"
            )
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Create the StatusBreakdown object and test it
        requests_breakdown = StatusBreakdown(
            successful=successful_requests,
            incomplete=incomplete_requests,
            errored=errored_requests,
        )
        logger.debug(
            "DEBUG GenerativeRequestsAggregator.compile: StatusBreakdown created"
        )
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: StatusBreakdown.successful type: {type(requests_breakdown.successful)}"
        )
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: StatusBreakdown.successful length: {len(requests_breakdown.successful) if requests_breakdown.successful is not None else 'None'}"
        )
        logger.debug(
            f"DEBUG GenerativeRequestsAggregator.compile: StatusBreakdown.successful is None: {requests_breakdown.successful is None}"
        )
        # TODO: Review Cursor generated code (end)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "request_totals": StatusBreakdown(
                successful=len(successful),
                incomplete=len(incomplete),
                errored=len(errored),
                total=len(total),
            ),
            # TODO: Review Cursor generated code (start)
            "requests": requests_breakdown,
            # TODO: Review Cursor generated code (end)
            "metrics": GenerativeMetrics(
                requests_per_second=(
                    StatusDistributionSummary.from_request_times(
                        # TODO: Review Cursor generated code (start)
                        request_types=[
                            req_type
                            for req, req_type in zip(total, total_types)
                            if (
                                req.scheduler_info.request_timings.request_start
                                is not None
                                and req.scheduler_info.request_timings.request_end
                                is not None
                            )
                        ],
                        # TODO: Review Cursor generated code (end)
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
                        # TODO: Review Cursor generated code (start)
                        request_types=[
                            req_type
                            for req, req_type in zip(total, total_types)
                            if (
                                req.scheduler_info.request_timings.request_start
                                is not None
                                and req.scheduler_info.request_timings.request_end
                                is not None
                            )
                        ],
                        # TODO: Review Cursor generated code (end)
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
                        # TODO: Review Cursor generated code (start)
                        value_types=[
                            req_type
                            for req, req_type in zip(total, total_types)
                            if req.request_latency is not None
                        ],
                        # TODO: Review Cursor generated code (end)
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
                        values=[
                            (req.prompt_tokens or 0) + (req.output_tokens or 0)
                            for req in total
                            if req.prompt_tokens is not None
                            or req.output_tokens is not None
                        ],
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
                            # TODO: Review Cursor generated code (start)
                            if (
                                req.scheduler_info.request_timings is not None
                                and hasattr(
                                    req.scheduler_info.request_timings,
                                    "first_iteration",
                                )
                                and req.scheduler_info.request_timings.first_iteration
                                is not None
                            )
                            else req.scheduler_info.request_timings.request_start
                            # TODO: Review Cursor generated code (end)
                            for req in total
                            if req.output_tokens_per_second is not None
                        ],
                        iter_counts=[
                            # TODO: Review Cursor generated code (start)
                            req.output_tokens if req.output_tokens is not None else 1
                            # TODO: Review Cursor generated code (end)
                            for req in total
                            if req.output_tokens_per_second is not None
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
                            # TODO: Review Cursor generated code (start)
                            if (
                                req.scheduler_info.request_timings is not None
                                and hasattr(
                                    req.scheduler_info.request_timings,
                                    "first_iteration",
                                )
                                and req.scheduler_info.request_timings.first_iteration
                                is not None
                            )
                            else req.scheduler_info.request_timings.request_start
                            # TODO: Review Cursor generated code (end)
                            for req in total
                            if req.tokens_per_second is not None
                        ],
                        iter_counts=[
                            # TODO: Review Cursor generated code (start)
                            req.output_tokens if req.output_tokens is not None else 1
                            # TODO: Review Cursor generated code (end)
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

        # TODO: Review Cursor generated code (start)
        # Debug timing data
        timings = request_info.request_timings
        # TODO: Review Cursor generated code (end)

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
