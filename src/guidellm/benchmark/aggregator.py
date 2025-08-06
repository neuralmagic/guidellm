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
    GenerativeRequestStats,
)
from guidellm.config import settings
from guidellm.objects import (
    StandardBaseModel,
    StatusBreakdown,
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
    "RequestsPerformanceStatsAggregator",
    "SchedulerPerformanceStatsAggregator",
]


@runtime_checkable
class Aggregator(Protocol[ResponseT, RequestT, RequestTimingsT]):
    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[ResponseT],
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        """
        Process a new update to the aggregator state in place.
        Optionally return updates for intermediate reporting before compilation.
        """
        ...


@runtime_checkable
class CompilableAggregator(Aggregator[ResponseT, RequestT, RequestTimingsT]):
    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[str, Any]: ...

    def info(self) -> dict[str, Any]: ...


def _add_aggregation_diff_state(
    base_key: str,
    agg_state: dict[str, Any],
    end_val: Optional[Union[int, float]],
    start_val: Optional[Union[int, float]] = 0.0,
    count: int = 1,
):
    if start_val is None or end_val is None:
        return

    delta_val = end_val - start_val
    agg_state[f"{base_key}_total"] = agg_state.get(f"{base_key}_total", 0) + delta_val
    agg_state[f"{base_key}_count"] = agg_state.get(f"{base_key}_count", 0) + count


class SchedulerStatsAggregator(
    CompilableAggregator[ResponseT, RequestT, RequestTimingsT]
):
    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[ResponseT],
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        if response is None:
            return None

        _add_aggregation_diff_state(
            "queued_time",
            agg_state,
            request_info.scheduler_timings.dequeued,
            request_info.scheduler_timings.queued,
        )
        _add_aggregation_diff_state(
            "worker_resolve_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.scheduler_timings.dequeued,
        )
        _add_aggregation_diff_state(
            "worker_resolve_time",
            agg_state,
            request_info.scheduler_timings.resolve_end,
            request_info.scheduler_timings.resolve_start,
        )
        _add_aggregation_diff_state(
            "worker_resolve_end_delay",
            agg_state,
            request_info.scheduler_timings.resolve_end,
            request_info.request_timings.request_end,
        )
        _add_aggregation_diff_state(
            "finalized_delay",
            agg_state,
            request_info.scheduler_timings.finalized,
            request_info.scheduler_timings.resolve_end,
        )
        _add_aggregation_diff_state(
            "worker_targeted_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.scheduler_timings.targeted_start,
        )
        _add_aggregation_diff_state(
            "request_start_delay",
            agg_state,
            request_info.scheduler_timings.resolve_start,
            request_info.request_timings.request_start,
        )
        _add_aggregation_diff_state(
            "request_time",
            agg_state,
            request_info.request_timings.request_end,
            request_info.request_timings.request_start,
        )
        _add_aggregation_diff_state(
            "request_targeted_delay",
            agg_state,
            request_info.request_timings.request_start,
            request_info.scheduler_timings.targeted_start,
        )

        return agg_state

    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[Literal["scheduler_stats"], BenchmarkSchedulerStats]:
        """
        Compile the current state of the aggregator into a more compact representation.
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
    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[GenerationResponse],
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        if response is None:
            return None

        if (
            request_info.request_timings.first_iteration is not None
            and request_info.request_timings.request_start is not None
        ):
            _add_aggregation_diff_state(
                "time_to_first_token",
                agg_state,
                request_info.request_timings.first_iteration,
                request_info.request_timings.request_start,
            )

        if (
            request_info.request_timings.first_iteration is not None
            and request_info.request_timings.last_iteration is not None
            and response.output_tokens is not None
        ):
            _add_aggregation_diff_state(
                "time_per_output_token",
                agg_state,
                request_info.request_timings.last_iteration,
                request_info.request_timings.request_start,
                count=response.output_tokens,
            )

            if response.output_tokens > 1:
                _add_aggregation_diff_state(
                    "inter_token_latency",
                    agg_state,
                    request_info.request_timings.last_iteration,
                    request_info.request_timings.first_iteration,
                    count=response.output_tokens - 1,
                )

        if response.prompt_tokens is not None:
            _add_aggregation_diff_state(
                "prompt_tokens",
                agg_state,
                response.prompt_tokens,
            )

        if response.output_tokens is not None:
            _add_aggregation_diff_state(
                "output_tokens",
                agg_state,
                response.output_tokens,
            )

        if response.total_tokens is not None:
            _add_aggregation_diff_state(
                "total_tokens",
                agg_state,
                response.total_tokens,
            )

        return agg_state


class GenerativeRequestsAggregator(
    StandardBaseModel,
    CompilableAggregator[
        GenerationResponse, GenerationRequest, GenerationRequestTimings
    ],
):
    warmup_requests: Optional[int] = Field(
        default=None,
        description="The number of warmup requests to ignore at the start of the benchmark.",
    )
    warmup_duration: Optional[float] = Field(
        default=None,
        description="The number of warmup seconds to ignore requests at the start of the benchmark.",
    )
    cooldown_requests: Optional[int] = Field(
        default=None,
        description="The number of cooldown requests to ignore at the end of the benchmark.",
    )
    cooldown_duration: Optional[float] = Field(
        default=None,
        description="The number of cooldown seconds to ignore requests at the end of the benchmark.",
    )

    def __call__(
        self,
        agg_state: dict[str, Any],
        response: Optional[GenerationResponse],
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> Optional[dict[str, Any]]:
        if response is None:
            return None

        if (
            self.warmup_requests is not None
            and self.warmup_requests >= scheduler_state.processed_requests
        ):
            return None

        if (
            self.warmup_duration is not None
            and request_info.request_timings.request_end is not None
            and (
                scheduler_state.start_time + self.warmup_duration
                >= request_info.request_timings.request_end
            )
        ):
            return None

        if (
            self.cooldown_requests is not None
            and scheduler_state.remaining_requests is not None
            and self.cooldown_requests >= scheduler_state.remaining_requests
        ):
            return None

        if (
            self.cooldown_duration is not None
            and scheduler_state.remaining_duration is not None
            and self.cooldown_duration >= scheduler_state.remaining_duration
        ):
            return None

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

        return None

    def compile(
        self, agg_state: dict[str, Any], scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        successful = [
            self._create_generate_stats(response, request, request_info)
            for (response, request, request_info) in agg_state.get("completed", [])
        ]
        incomplete = [
            self._create_generate_stats(response, request, request_info)
            for (response, request, request_info) in agg_state.get("incomplete", [])
        ]
        error = [
            self._create_generate_stats(response, request, request_info)
            for (response, request, request_info) in agg_state.get("errored", [])
        ]

        return {
            "requests": StatusBreakdown(
                successful=successful,
                incomplete=incomplete,
                errored=error,
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
