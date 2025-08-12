"""
Constraint system for scheduler behavior control and request processing limits.

Provides flexible constraints for managing scheduler behavior with configurable
thresholds based on time, error rates, and request counts.

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
    Constraint: Function signature for constraint evaluation.
    ConstraintInitializer: Function signature for constraint factory.
"""

import time
from typing import Any, Optional, Protocol, Union, runtime_checkable

from pydantic import Field

from guidellm.objects import StandardBaseModel
from guidellm.scheduler.objects import (
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.utils import RegistryMixin

__all__ = [
    "Constraint",
    "ConstraintInitializer",
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


@runtime_checkable
class Constraint(Protocol):
    """Protocol for constraint evaluation functions."""

    def __call__(
        self, state: SchedulerState, request: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against scheduler state and request information.

        :param state: Current scheduler state with metrics and timing.
        :param request: Individual request information and metadata.
        :return: Action indicating whether to continue or stop operations.
        """


@runtime_checkable
class ConstraintInitializer(Protocol):
    """Protocol for constraint initializer factory functions."""

    def create_constraint(self, **kwargs) -> Constraint:
        """
        Create a constraint instance from configuration parameters.

        :param kwargs: Configuration parameters for constraint creation.
        :return: Configured constraint evaluation function.
        """


class ConstraintsInitializerFactory(RegistryMixin[ConstraintInitializer]):
    """Registry factory for creating and managing constraint initializers."""

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> ConstraintInitializer:
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

        initializer_class = cls.get_registered_object(key)

        # Handle simple scalar values by delegating to the initializer class
        if (
            len(args) == 1
            and not kwargs
            and hasattr(initializer_class, "from_simple_value")
        ):
            return initializer_class.from_simple_value(args[0])

        return initializer_class(*args, **kwargs)

    @classmethod
    def create_constraint(cls, key: str, *args, **kwargs) -> Constraint:
        """
        Create a constraint instance for the specified key.

        :param key: Registered constraint initializer key.
        :param kwargs: Keyword arguments for constraint creation.
        :return: Configured constraint function ready for evaluation.
        :raises ValueError: If the key is not registered in the factory.
        """
        return cls.create(key, *args, **kwargs).create_constraint()

    @classmethod
    def resolve(
        cls,
        initializers: dict[
            str,
            Union[Any, dict[str, Any], Constraint, ConstraintInitializer],
        ],
    ) -> dict[str, Constraint]:
        """
        Resolve mixed constraint specifications to callable constraints.

        :param initializers: Dictionary mapping constraint keys to specifications.
        :return: Dictionary mapping constraint keys to callable functions.
        :raises ValueError: If any key is not registered in the factory.
        """
        constraints = {}

        for key, val in initializers.items():
            if isinstance(val, Constraint):
                constraints[key] = val
            elif isinstance(val, ConstraintInitializer):
                constraints[key] = val.create_constraint()
            elif isinstance(val, dict):
                constraints[key] = cls.create_constraint(key, **val)
            else:
                constraints[key] = cls.create_constraint(key, val)

        return constraints

    @classmethod
    def resolve_constraints(
        cls,
        constraints: dict[str, Union[Any, dict[str, Any], Constraint]],
    ) -> dict[str, Constraint]:
        """
        Resolve constraints from mixed constraint specifications.

        :param constraints: Dictionary mapping constraint keys to specifications.
        :return: Dictionary mapping constraint keys to callable functions.
        :raises ValueError: If any constraint key is not registered.
        """
        resolved_constraints = {}

        for key, val in constraints.items():
            if isinstance(val, Constraint):
                resolved_constraints[key] = val
            elif isinstance(val, dict):
                resolved_constraints[key] = cls.create_constraint(key, **val)
            else:
                resolved_constraints[key] = cls.create_constraint(key, val)

        return resolved_constraints


class _MaxNumberBase(StandardBaseModel):
    max_num: Union[int, float] = Field(
        gt=0, description="Maximum number of requests allowed"
    )


class MaxNumberConstraint(_MaxNumberBase):
    """Constraint that limits execution based on maximum request counts."""

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
            progress={
                "remaining_fraction": max(
                    0.0, 1.0 - state.processed_requests / float(self.max_num)
                ),
                "remaining_requests": max(0, self.max_num - state.processed_requests),
            },
        )


@ConstraintsInitializerFactory.register("max_number")
class MaxNumberConstraintInitializer(_MaxNumberBase):
    """Factory for creating MaxNumberConstraint instances."""

    @classmethod
    def from_simple_value(
        cls, value: Union[int, float]
    ) -> "MaxNumberConstraintInitializer":
        """
        Create a MaxNumberConstraintInitializer from a simple scalar value.

        :param value: Maximum number of requests allowed.
        :return: Configured MaxNumberConstraintInitializer instance.
        """
        return cls(max_num=value)

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a MaxNumberConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxNumberConstraint instance.
        """
        return MaxNumberConstraint(
            max_num=self.max_num,
        )


class _MaxDurationBase(StandardBaseModel):
    max_duration: Union[int, float] = Field(
        gt=0, description="Maximum duration in seconds"
    )


class MaxDurationConstraint(_MaxDurationBase):
    """Constraint that limits execution based on maximum time duration."""

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
            progress={
                "remaining_fraction": max(
                    0.0, 1.0 - elapsed / float(self.max_duration)
                ),
                "remaining_duration": max(0.0, self.max_duration - elapsed),
            },
        )


@ConstraintsInitializerFactory.register("max_duration")
class MaxDurationConstraintInitializer(_MaxDurationBase):
    """Factory for creating MaxDurationConstraint instances."""

    @classmethod
    def from_simple_value(
        cls, value: Union[int, float]
    ) -> "MaxDurationConstraintInitializer":
        """
        Create a MaxDurationConstraintInitializer from a simple scalar value.

        :param value: Maximum duration in seconds.
        :return: Configured MaxDurationConstraintInitializer instance.
        """
        return cls(max_duration=value)

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a MaxDurationConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxDurationConstraint instance.
        """
        return MaxDurationConstraint(
            max_duration=self.max_duration,
        )


class _MaxErrorsBase(StandardBaseModel):
    max_errors: Union[int, float] = Field(
        gt=0, description="Maximum number of errors allowed"
    )


class MaxErrorsConstraint(_MaxErrorsBase):
    """Constraint that limits execution based on absolute error count."""

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
class MaxErrorsConstraintInitializer(_MaxErrorsBase):
    """Factory for creating MaxErrorsConstraint instances."""

    @classmethod
    def from_simple_value(
        cls, value: Union[int, float]
    ) -> "MaxErrorsConstraintInitializer":
        """
        Create a MaxErrorsConstraintInitializer from a simple scalar value.

        :param value: Maximum number of errors allowed.
        :return: Configured MaxErrorsConstraintInitializer instance.
        """
        return cls(max_errors=value)

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a MaxErrorsConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxErrorsConstraint instance.
        """
        return MaxErrorsConstraint(
            max_errors=self.max_errors,
        )


class _MaxErrorRateBase(StandardBaseModel):
    max_error_rate: Union[int, float] = Field(
        gt=0, le=1, description="Maximum error rate allowed (0.0 to 1.0)"
    )
    window_size: Union[int, float] = Field(
        default=50,
        gt=0,
        description="Size of sliding window for calculating error rate",
    )


class MaxErrorRateConstraint(_MaxErrorRateBase):
    """Constraint that limits execution based on sliding window error rate."""

    error_window: list[bool] = Field(
        default_factory=list,
        description="Sliding window tracking error status of recent requests",
    )

    def __call__(
        self, state: SchedulerState, request_info: ScheduledRequestInfo
    ) -> SchedulerUpdateAction:
        """
        Evaluate constraint against sliding window error rate.

        :param state: Current scheduler state with request counts.
        :param request_info: Individual request with completion status.
        :return: Action indicating whether to continue or stop operations.
        """
        if request_info.status in ["completed", "errored", "cancelled"]:
            self.error_window.append(request_info.status == "errored")
            if len(self.error_window) > self.window_size:
                self.error_window.pop(0)

        error_count = sum(self.error_window)
        window_requests = len(self.error_window)
        error_rate = (
            error_count / float(window_requests) if window_requests > 0 else 0.0
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
class MaxErrorRateConstraintInitializer(_MaxErrorRateBase):
    """Factory for creating MaxErrorRateConstraint instances."""

    @classmethod
    def from_simple_value(
        cls, value: Union[int, float]
    ) -> "MaxErrorRateConstraintInitializer":
        """
        Create a MaxErrorRateConstraintInitializer from a simple scalar value.

        :param value: Maximum error rate allowed (0.0 to 1.0).
        :return: Configured MaxErrorRateConstraintInitializer instance.
        """
        return cls(max_error_rate=value)

    def create_constraint(self, **_kwargs) -> Constraint:
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
    max_error_rate: Union[int, float] = Field(
        gt=0, le=1, description="Maximum error rate allowed (0.0 to 1.0)"
    )
    min_processed: Optional[Union[int, float]] = Field(
        default=50,
        gt=30,
        description=(
            "Minimum number of processed requests before applying error rate constraint"
        ),
    )


class MaxGlobalErrorRateConstraint(_MaxGlobalErrorRateBase):
    """Constraint that limits execution based on global error rate."""

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
class MaxGlobalErrorRateConstraintInitializer(_MaxGlobalErrorRateBase):
    """Factory for creating MaxGlobalErrorRateConstraint instances."""

    @classmethod
    def from_simple_value(
        cls, value: Union[int, float]
    ) -> "MaxGlobalErrorRateConstraintInitializer":
        """
        Create a MaxGlobalErrorRateConstraintInitializer from a simple scalar value.

        :param value: Maximum error rate allowed (0.0 to 1.0).
        :return: Configured MaxGlobalErrorRateConstraintInitializer instance.
        """
        return cls(max_error_rate=value)

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a MaxGlobalErrorRateConstraint instance.

        :param _kwargs: Additional keyword arguments (unused).
        :return: Configured MaxGlobalErrorRateConstraint instance.
        """
        return MaxGlobalErrorRateConstraint(
            max_error_rate=self.max_error_rate,
            min_processed=self.min_processed,
        )
