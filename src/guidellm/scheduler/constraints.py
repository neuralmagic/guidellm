"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior in GuideLLM,
enabling limits on request queuing and processing based on time, error rates,
and request counts with configurable thresholds and evaluation windows.

Classes:
    ConstraintsInitializerFactory: Registry for constraint initializer functions.
    MaxNumberConstraint: Limits execution by maximum request count.
    MaxNumberConstraintInitializer: Factory for MaxNumberConstraint instances.
    MaxDurationConstraint: Limits execution by maximum time duration.
    MaxDurationConstraintInitializer: Factory for MaxDurationConstraint instances.
    MaxErrorsConstraint: Limits execution by maximum absolute error count.
    MaxErrorsConstraintInitializer: Factory for MaxErrorsConstraint instances.
    MaxErrorRateConstraint: Limits execution by sliding window error rate.
    MaxErrorRateConstraintInitializer: Factory for MaxErrorRateConstraint instances.
    MaxGlobalErrorRateConstraint: Limits execution by global error rate.
    MaxGlobalErrorRateConstraintInitializer: Factory for MaxGlobalErrorRateConstraint.

Type Aliases:
    CallableConstraint: Function signature for constraint evaluation.
    CallableConstraintInitializer: Function signature for constraint factory.
"""

import time
from typing import Any, Callable, Optional, Union

from pydantic import Field

from guidellm.config import settings
from guidellm.objects import StandardBaseModel
from guidellm.scheduler.objects import (
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.utils import RegistryMixin

__all__ = [
    "CallableConstraint",
    "CallableConstraintInitializer",
    "ConstraintsInitializerFactory",
    "MaxDurationConstraint",
    "MaxDurationConstraintInitializer",
    "MaxErrorRateConstraint",
    "MaxErrorRateConstraintInitializer",
    "MaxErrorsConstraint",
    "MaxErrorsConstraintInitializer",
    "MaxGlobalErrorRateConstraint",
    "MaxGlobalErrorRateConstraintInitializer",
    "MaxNumberConstraint",
    "MaxNumberConstraintInitializer",
]

CallableConstraint = Callable[
    [SchedulerState, ScheduledRequestInfo], SchedulerUpdateAction
]

CallableConstraintInitializer = Callable[..., CallableConstraint]


class ConstraintsInitializerFactory(RegistryMixin[CallableConstraintInitializer]):
    """
    Registry factory for creating and managing constraint initializers.

    Provides a centralized registry for constraint initializer functions, enabling
    dynamic constraint creation from configuration keys and parameters. Supports
    resolution of mixed constraint specifications including callables, dictionaries,
    and scalar values.
    """

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> CallableConstraintInitializer:
        """
        Create a constraint initializer for the specified key.

        :param key: Registered constraint initializer key.
        :param args: Positional arguments for initializer creation.
        :param kwargs: Keyword arguments for initializer creation.
        :return: Configured constraint initializer function.
        :raises ValueError: If the key is not registered in the factory.
        """
        if not cls.is_registered(key):
            raise ValueError(f"Unknown constraint initializer key: {key}")

        return cls.get_registered_object(key)(*args, **kwargs)

    @classmethod
    def resolve(
        cls,
        initializers: dict[
            str, Union[Any, dict[str, Any], CallableConstraintInitializer]
        ],
    ) -> dict[str, CallableConstraint]:
        """
        Resolve a dictionary of mixed constraint specifications to callable constraints.

        Handles three types of specifications:
        - Callable objects: Used directly as constraints
        - Dictionary values: Expanded as keyword arguments to initializers
        - Scalar values: Passed as single arguments to initializers

        :param initializers: Dictionary mapping constraint keys to specifications.
        :return: Dictionary mapping constraint keys to callable constraint functions.
        :raises ValueError: If any key is not registered in the factory.
        """
        return {
            key: (
                val
                if callable(val)
                else cls.create(key, **val)
                if isinstance(val, dict)
                else cls.create(key, val)
            )
            for key, val in initializers.items()
        }

    @classmethod
    def create_constraint(cls, key: str, *args, **kwargs) -> CallableConstraint:
        """
        Create a constraint instance for the specified key.

        :param key: Registered constraint initializer key.
        :param args: Positional arguments for constraint creation.
        :param kwargs: Keyword arguments for constraint creation.
        :return: Configured constraint function ready for evaluation.
        :raises ValueError: If the key is not registered in the factory.
        """
        initializer = cls.create(key)

        return initializer(*args, **kwargs)

    @classmethod
    def resolve_constraints(
        cls,
        constraints: dict[str, Union[Any, dict[str, Any], CallableConstraint]],
    ) -> dict[str, CallableConstraint]:
        """
        Resolve constraints from a dictionary of mixed constraint specifications.

        Handles three types of constraint specifications:
        - Callable objects: Used directly as constraints
        - Dictionary values: Expanded as keyword arguments to constraint initializers
        - Scalar values: Passed as single arguments to constraint initializers

        :param constraints: Dictionary mapping constraint keys to specifications.
        :return: Dictionary mapping constraint keys to callable constraint functions.
        :raises ValueError: If any constraint key is not registered in the factory.
        """
        return {
            key: (
                val
                if callable(val)
                else cls.create_constraint(key, **val)
                if isinstance(val, dict)
                else cls.create_constraint(key, val)
            )
            for key, val in constraints.items()
        }


class _MaxNumberBase(StandardBaseModel):
    """Base configuration for maximum number constraints."""

    max_num: Union[int, float] = Field(
        ge=0, description="Maximum number of requests allowed"
    )


class MaxNumberConstraint(_MaxNumberBase, CallableConstraint):
    """
    Constraint that limits execution based on maximum request counts.

    Stops request queuing when the number of created requests reaches the limit,
    and stops processing when the number of processed requests reaches the limit.
    Supports both integer and floating-point limits for flexibility.
    """

    def __call__(
        self, state: SchedulerState, _request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current scheduler state.

        :param state: Current scheduler state with request counts.
        :param _request_info: Individual request information (unused).
        :return: Action indicating whether to continue or stop operations.
        """
        create_exceeded = state.created_requests >= self.max_num
        processed_exceeded = state.processed_requests >= self.max_num

        return SchedulerUpdateAction(
            request_queuing="stop" if create_exceeded else "continue",
            request_processing="stop_local" if processed_exceeded else "continue",
            metadata={
                "max_number": self.max_num,
                "create_exceeded": create_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
            },
        )


@ConstraintsInitializerFactory.register("max_number")
class MaxNumberConstraintInitializer(_MaxNumberBase, CallableConstraintInitializer):
    """Factory for creating MaxNumberConstraint instances from configuration."""

    def __call__(self, **_kwargs) -> CallableConstraint:
        """
        Create a MaxNumberConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxNumberConstraint instance.
        """
        return MaxNumberConstraint(
            max_num=self.max_num,
        )


class _MaxDurationBase(StandardBaseModel):
    """Base configuration for maximum duration constraints."""

    max_duration: Union[int, float] = Field(
        ge=0, description="Maximum duration in seconds"
    )


class MaxDurationConstraint(_MaxDurationBase, CallableConstraint):
    """
    Constraint that limits execution based on maximum time duration.

    Stops both request queuing and processing when the elapsed time since
    scheduler startup exceeds the configured maximum duration. Uses wall-clock
    time for accurate duration measurement regardless of processing load.
    """

    def __call__(
        self, state: SchedulerState, _request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current scheduler state and elapsed time.

        :param state: Current scheduler state with start time.
        :param _request_info: Individual request information (unused).
        :return: Action indicating whether to continue or stop operations.
        """
        current_time = time.time()
        elapsed = current_time - state.start_time
        duration_exceeded = elapsed >= self.max_duration

        return SchedulerUpdateAction(
            request_queuing="stop" if duration_exceeded else "continue",
            request_processing="stop_local" if duration_exceeded else "continue",
            metadata={
                "max_duration": self.max_duration,
                "elapsed_time": elapsed,
                "duration_exceeded": duration_exceeded,
                "start_time": state.start_time,
                "current_time": current_time,
            },
        )


@ConstraintsInitializerFactory.register("max_duration")
class MaxDurationConstraintInitializer(_MaxDurationBase, CallableConstraintInitializer):
    """Factory for creating MaxDurationConstraint instances from configuration."""

    def __call__(self, **_kwargs) -> CallableConstraint:
        """
        Create a MaxDurationConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxDurationConstraint instance.
        """
        return MaxDurationConstraint(
            max_duration=self.max_duration,
        )


class _MaxErrorsBase(StandardBaseModel):
    """Base configuration for maximum errors constraints."""

    max_errors: Union[int, float] = Field(
        gt=0, description="Maximum number of errors allowed"
    )


class MaxErrorsConstraint(_MaxErrorsBase, CallableConstraint):
    """
    Constraint that limits execution based on absolute error count.

    Stops both request queuing and processing (across all nodes) when the
    total number of errored requests exceeds the configured threshold.
    Uses global stop to prevent error propagation across distributed nodes.
    """

    def __call__(
        self, state: SchedulerState, _request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against current error count.

        :param state: Current scheduler state with error counts.
        :param _request_info: Individual request information (unused).
        :return: Action indicating whether to continue or stop operations.
        """
        errors_exceeded = state.errored_requests >= self.max_errors

        return SchedulerUpdateAction(
            request_queuing="stop" if errors_exceeded else "continue",
            request_processing="stop_all" if errors_exceeded else "continue",
            metadata={
                "max_errors": self.max_errors,
                "errors_exceeded": errors_exceeded,
                "current_errors": state.errored_requests,
            },
        )


@ConstraintsInitializerFactory.register("max_errors")
class MaxErrorsConstraintInitializer(_MaxErrorsBase, CallableConstraintInitializer):
    """Factory for creating MaxErrorsConstraint instances from configuration."""

    def __call__(self, **_kwargs) -> CallableConstraint:
        """
        Create a MaxErrorsConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxErrorsConstraint instance.
        """
        return MaxErrorsConstraint(
            max_errors=self.max_errors,
        )


class _MaxErrorRateBase(StandardBaseModel):
    """Base configuration for sliding window error rate constraints."""

    max_error_rate: Union[int, float] = Field(
        ge=0, le=1, description="Maximum error rate allowed (0.0 to 1.0)"
    )
    window_size: Union[int, float] = Field(
        default_factory=lambda: settings.constraint_max_error_rate_min_processed,
        gt=0,
        description="Size of sliding window for calculating error rate",
    )


class MaxErrorRateConstraint(_MaxErrorRateBase, CallableConstraint):
    """
    Constraint that limits execution based on sliding window error rate.

    Maintains a sliding window of recent request outcomes and stops execution
    when the error rate within that window exceeds the configured threshold.
    Provides more responsive error detection than global error rate constraints.
    """

    error_window: list[bool] = Field(
        default_factory=list,
        description="Sliding window tracking error status of recent requests",
    )

    def __call__(
        self, state: SchedulerState, request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against sliding window error rate.

        Updates the error window with the current request status and calculates
        the error rate within the window. Stops execution if rate exceeds threshold.

        :param state: Current scheduler state with request counts.
        :param request_info: Individual request with completion status.
        :return: Action indicating whether to continue or stop operations.
        """
        if request_info.status in ["completed", "errored", "cancelled"]:
            self.error_window.append(request_info.status == "errored")
            if len(self.error_window) > self.window_size:
                self.error_window.pop(0)

        error_count = sum(self.error_window)
        error_rate = (
            error_count / float(state.processed_requests)
            if state.processed_requests > 0
            else 0.0
        )
        exceeded_min_processed = state.processed_requests >= self.window_size
        exceeded_error_rate = error_rate >= self.max_error_rate

        return SchedulerUpdateAction(
            request_queuing=(
                "stop" if exceeded_min_processed and exceeded_error_rate else "continue"
            ),
            request_processing=(
                "stop_all"
                if exceeded_min_processed and exceeded_error_rate
                else "continue"
            ),
            metadata={
                "max_error_rate": self.max_error_rate,
                "window_size": self.window_size,
                "error_count": error_count,
                "processed_count": state.processed_requests,
                "current_window_size": len(self.error_window),
                "current_error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
            },
        )


@ConstraintsInitializerFactory.register("max_error_rate")
class MaxErrorRateConstraintInitializer(
    _MaxErrorRateBase, CallableConstraintInitializer
):
    """Factory for creating MaxErrorRateConstraint instances from configuration."""

    def __call__(self, **_kwargs) -> CallableConstraint:
        """
        Create a MaxErrorRateConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxErrorRateConstraint instance.
        """
        return MaxErrorRateConstraint(
            max_error_rate=self.max_error_rate,
            window_size=self.window_size,
        )


class _MaxGlobalErrorRateBase(StandardBaseModel):
    """Base configuration for global error rate constraints."""

    max_error_rate: Union[int, float] = Field(
        ge=0, le=1, description="Maximum error rate allowed (0.0 to 1.0)"
    )
    min_processed: Optional[Union[int, float]] = Field(
        default_factory=lambda: settings.constraint_max_error_rate_min_processed,
        gt=30,
        description=(
            "Minimum number of processed requests before applying error rate constraint"
        ),
    )


class MaxGlobalErrorRateConstraint(_MaxGlobalErrorRateBase, CallableConstraint):
    """
    Constraint that limits execution based on global error rate.

    Calculates error rate across all processed requests and stops execution
    when the rate exceeds the threshold, but only after a minimum number of
    requests have been processed to ensure statistical significance.
    """

    def __call__(
        self, state: SchedulerState, _request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against global error rate.

        :param state: Current scheduler state with global request and error counts.
        :param _request_info: Individual request information (unused).
        :return: Action indicating whether to continue or stop operations.
        """
        exceeded_min_processed = state.processed_requests >= self.min_processed
        error_rate = (
            state.errored_requests / float(state.processed_requests)
            if state.processed_requests > 0
            else 0.0
        )
        exceeded_error_rate = error_rate >= self.max_error_rate
        should_stop = exceeded_min_processed and exceeded_error_rate

        return SchedulerUpdateAction(
            request_queuing="stop" if should_stop else "continue",
            request_processing="stop_all" if should_stop else "continue",
            metadata={
                "max_error_rate": self.max_error_rate,
                "min_processed": self.min_processed,
                "processed_requests": state.processed_requests,
                "errored_requests": state.errored_requests,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
            },
        )


@ConstraintsInitializerFactory.register("max_global_error_rate")
class MaxGlobalErrorRateConstraintInitializer(
    _MaxGlobalErrorRateBase, CallableConstraintInitializer
):
    """Factory for creating MaxGlobalErrorRateConstraint instances."""

    def __call__(self, **_kwargs) -> CallableConstraint:
        """
        Create a MaxGlobalErrorRateConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxGlobalErrorRateConstraint instance.
        """
        return MaxGlobalErrorRateConstraint(
            max_error_rate=self.max_error_rate,
            min_processed=self.min_processed,
        )
