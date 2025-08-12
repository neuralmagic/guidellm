import inspect
import random
import time
from typing import Protocol

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    Constraint,
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    MaxDurationConstraint,
    MaxDurationConstraintInitializer,
    MaxErrorRateConstraint,
    MaxErrorRateConstraintInitializer,
    MaxErrorsConstraint,
    MaxErrorsConstraintInitializer,
    MaxGlobalErrorRateConstraint,
    MaxGlobalErrorRateConstraintInitializer,
    MaxNumberConstraint,
    MaxNumberConstraintInitializer,
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
)


class TestConstraint:
    """Test the Constraint protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test that Constraint is a protocol and runtime checkable."""
        assert issubclass(Constraint, Protocol)
        assert hasattr(Constraint, "_is_protocol")
        assert Constraint._is_protocol is True
        assert hasattr(Constraint, "_is_runtime_protocol")
        assert Constraint._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signature(self):
        """Test that the Constraint protocol has the correct method signature."""
        call_method = Constraint.__call__
        sig = inspect.signature(call_method)

        expected_params = ["self", "state", "request"]
        assert list(sig.parameters.keys()) == expected_params

        params = sig.parameters
        assert "state" in params
        assert "request" in params

    @pytest.mark.smoke
    def test_runtime_is_constraint(self):
        """Test that Constraint can be checked at runtime using isinstance."""

        class ValidConstraint:
            def __call__(
                self,
                state: SchedulerState,
                request: ScheduledRequestInfo,
            ) -> SchedulerUpdateAction:
                return SchedulerUpdateAction()

        valid_instance = ValidConstraint()
        assert isinstance(valid_instance, Constraint)

        class InvalidConstraint:
            pass

        invalid_instance = InvalidConstraint()
        assert not isinstance(invalid_instance, Constraint)

    @pytest.mark.smoke
    def test_runtime_is_not_intializer(self):
        """
        Test that a class not implementing the ConstraintInitializer
        protocol is not recognized as such.
        """

        class ValidConstraint:
            def __call__(
                self,
                state: SchedulerState,
                request: ScheduledRequestInfo,
            ) -> SchedulerUpdateAction:
                return SchedulerUpdateAction()

        not_initializer_instance = ValidConstraint()
        assert not isinstance(not_initializer_instance, ConstraintInitializer)


class TestConstraintInitializer:
    """Test the ConstraintInitializer protocol."""

    @pytest.mark.smoke
    def test_is_protocol(self):
        """Test that ConstraintInitializer is a protocol and runtime checkable."""
        assert issubclass(ConstraintInitializer, Protocol)
        assert hasattr(ConstraintInitializer, "_is_protocol")
        assert ConstraintInitializer._is_protocol is True
        assert hasattr(ConstraintInitializer, "_is_runtime_protocol")
        assert ConstraintInitializer._is_runtime_protocol is True

    @pytest.mark.smoke
    def test_protocol_method_signature(self):
        """Test that ConstraintInitializer protocol has correct method signature."""
        create_constraint_method = ConstraintInitializer.create_constraint
        sig = inspect.signature(create_constraint_method)

        expected_params = ["self", "kwargs"]
        assert list(sig.parameters.keys()) == expected_params
        kwargs_param = sig.parameters["kwargs"]
        assert kwargs_param.kind == kwargs_param.VAR_KEYWORD

    @pytest.mark.smoke
    def test_runtime_is_initializer(self):
        """Test that ConstraintInitializer can be checked at runtime."""

        class ValidInitializer:
            def create_constraint(self, **kwargs) -> Constraint:
                class SimpleConstraint:
                    def __call__(
                        self,
                        state: SchedulerState,
                        request: ScheduledRequestInfo,
                    ) -> SchedulerUpdateAction:
                        return SchedulerUpdateAction()

                return SimpleConstraint()

        valid_instance = ValidInitializer()
        assert isinstance(valid_instance, ConstraintInitializer)

    @pytest.mark.smoke
    def test_runtime_is_not_constraint(self):
        """
        Test that a class not implementing the Constraint protocol
        is not recognized as such.
        """

        class ValidInitializer:
            def create_constraint(self, **kwargs) -> Constraint:
                class SimpleConstraint:
                    def __call__(
                        self,
                        state: SchedulerState,
                        request: ScheduledRequestInfo,
                    ) -> SchedulerUpdateAction:
                        return SchedulerUpdateAction()

                return SimpleConstraint()

        not_constraint_instance = ValidInitializer()
        assert not isinstance(not_constraint_instance, Constraint)


class TestMaxNumberConstraint:
    """Test the MaxNumberConstraint implementation."""

    @pytest.fixture(params=[{"max_num": 100}, {"max_num": 50.5}, {"max_num": 1}])
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxNumberConstraint(**constructor_args)

        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxNumberConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.sanity
    def test_is_not_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxNumberConstraint does not satisfy
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert not isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxNumberConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxNumberConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxNumberConstraint()
        with pytest.raises(ValidationError):
            MaxNumberConstraint(max_num=-1)
        with pytest.raises(ValidationError):
            MaxNumberConstraint(max_num=0)
        with pytest.raises(ValidationError):
            MaxNumberConstraint(max_num="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions and progress"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        for num_requests in range(0, int(constructor_args["max_num"]) * 2 + 1, 1):
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=num_requests,
                processed_requests=num_requests // 2,
            )
            request = ScheduledRequestInfo(
                request_id=f"test-{num_requests}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)
            created_exceeded = num_requests >= constructor_args["max_num"]
            processed_exceeded = num_requests // 2 >= constructor_args["max_num"]
            expected_queuing = "stop" if created_exceeded else "continue"
            expected_processing = "stop_local" if processed_exceeded else "continue"
            assert action.request_queuing == expected_queuing
            assert action.request_processing == expected_processing
            assert isinstance(action.metadata, dict)
            assert action.metadata == {
                "max_number": constructor_args["max_num"],
                "create_exceeded": created_exceeded,
                "processed_exceeded": processed_exceeded,
                "created_requests": state.created_requests,
                "processed_requests": state.processed_requests,
            }
            assert isinstance(action.progress, dict)
            processed_requests = num_requests // 2
            remaining_fraction = max(
                0.0, 1.0 - processed_requests / constructor_args["max_num"]
            )
            remaining_requests = max(
                0.0, constructor_args["max_num"] - processed_requests
            )
            assert action.progress["remaining_fraction"] == pytest.approx(
                remaining_fraction
            )
            assert action.progress["remaining_requests"] == remaining_requests

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxNumberConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxNumberConstraint.model_validate(data)
        assert reconstructed.max_num == instance.max_num

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxNumberConstraintInitializer:
    """Test the MaxNumberConstraintInitializer implementation."""

    @pytest.fixture(params=[{"max_num": 100}, {"max_num": 50.5}, {"max_num": 1}])
    def valid_instances(self, request):
        """Provide valid instances of MaxNumberConstraintInitializer."""
        params = request.param
        instance = MaxNumberConstraintInitializer(**params)
        return instance, params

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self):
        """Test that MaxNumberConstraintInitializer satisfies the protocol."""
        initializer = MaxNumberConstraintInitializer(max_num=100)
        assert isinstance(initializer, ConstraintInitializer)

    @pytest.mark.smoke
    def test_is_not_constraint_protocol(self):
        """
        Test that MaxNumberConstraintInitializer does not satisfy
        the constraint protocol.
        """
        initializer = MaxNumberConstraintInitializer(max_num=100)
        assert not isinstance(initializer, Constraint)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that the initializer can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that the initializer rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxNumberConstraintInitializer()
        with pytest.raises(ValidationError):
            MaxNumberConstraintInitializer(max_num=-1)
        with pytest.raises(ValidationError):
            MaxNumberConstraintInitializer(max_num=0)
        with pytest.raises(ValidationError):
            MaxNumberConstraintInitializer(max_num="invalid")

    def test_constraint_initialization_functionality(self, valid_instances):
        """Test that the constraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxNumberConstraint)
        assert constraint.max_num == constructor_args["max_num"]

    def test_marshalling(self, valid_instances):
        """
        Test that MaxNumberConstraintInitializer can be
        serialized and deserialized.
        """
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxNumberConstraintInitializer.model_validate(data)
        assert reconstructed.max_num == instance.max_num

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxDurationConstraint:
    """Test the MaxDurationConstraint implementation."""

    @pytest.fixture(
        params=[{"max_duration": 2.0}, {"max_duration": 1}, {"max_duration": 0.5}]
    )
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxDurationConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxDurationConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.sanity
    def test_is_not_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxDurationConstraint does not satisfy
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert not isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxDurationConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxDurationConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxDurationConstraint()
        with pytest.raises(ValidationError):
            MaxDurationConstraint(max_duration=-1)
        with pytest.raises(ValidationError):
            MaxDurationConstraint(max_duration=0)
        with pytest.raises(ValidationError):
            MaxDurationConstraint(max_duration="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions and progress through a time loop"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        max_duration = constructor_args["max_duration"]
        sleep_interval = max_duration * 0.05
        target_duration = max_duration * 1.5

        elapsed = 0.0
        step = 0

        while elapsed <= target_duration:
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=step + 1,
                processed_requests=step,
            )
            request = ScheduledRequestInfo(
                request_id=f"test-{step}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)

            duration_exceeded = elapsed >= max_duration

            if not duration_exceeded:
                assert action.request_queuing == "continue"
                assert action.request_processing == "continue"
            else:
                assert action.request_queuing == "stop"
                assert action.request_processing == "stop_local"
            assert isinstance(action.metadata, dict)
            assert action.metadata["max_duration"] == max_duration
            assert action.metadata["elapsed_time"] == pytest.approx(elapsed, abs=0.01)
            assert action.metadata["duration_exceeded"] == duration_exceeded
            assert action.metadata["start_time"] == start_time
            assert isinstance(action.progress, dict)
            expected_remaining_fraction = max(0.0, 1.0 - elapsed / max_duration)
            expected_remaining_duration = max(0.0, max_duration - elapsed)
            assert action.progress["remaining_fraction"] == pytest.approx(
                expected_remaining_fraction, abs=0.1
            )
            assert action.progress["remaining_duration"] == pytest.approx(
                expected_remaining_duration, abs=0.1
            )
            time.sleep(sleep_interval)
            elapsed = time.time() - start_time
            step += 1

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxDurationConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxDurationConstraint.model_validate(data)
        assert reconstructed.max_duration == instance.max_duration

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxDurationConstraintInitializer:
    """Test the MaxDurationConstraintInitializer implementation."""

    @pytest.fixture(
        params=[{"max_duration": 30.0}, {"max_duration": 60}, {"max_duration": 0.5}]
    )
    def valid_instances(self, request):
        """Provide valid instances of MaxDurationConstraintInitializer."""
        params = request.param
        instance = MaxDurationConstraintInitializer(**params)
        return instance, params

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self):
        """Test that MaxDurationConstraintInitializer satisfies the protocol."""
        initializer = MaxDurationConstraintInitializer(max_duration=30.0)
        assert isinstance(initializer, ConstraintInitializer)

    @pytest.mark.smoke
    def test_is_not_constraint_protocol(self):
        """
        Test that MaxDurationConstraintInitializer does not satisfy
        the constraint protocol.
        """
        initializer = MaxDurationConstraintInitializer(max_duration=30.0)
        assert not isinstance(initializer, Constraint)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that the initializer can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that the initializer rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxDurationConstraintInitializer()
        with pytest.raises(ValidationError):
            MaxDurationConstraintInitializer(max_duration=0)
        with pytest.raises(ValidationError):
            MaxDurationConstraintInitializer(max_duration=-1)
        with pytest.raises(ValidationError):
            MaxDurationConstraintInitializer(max_duration="invalid")

    def test_constraint_initialization_functionality(self, valid_instances):
        """Test that the constraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxDurationConstraint)
        assert constraint.max_duration == constructor_args["max_duration"]

    def test_marshalling(self, valid_instances):
        """
        Test that MaxDurationConstraintInitializer can be
        serialized and deserialized.
        """
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxDurationConstraintInitializer.model_validate(data)
        assert reconstructed.max_duration == instance.max_duration

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxErrorsConstraint:
    """Test the MaxErrorsConstraint implementation."""

    @pytest.fixture(params=[{"max_errors": 10}, {"max_errors": 5.5}, {"max_errors": 1}])
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxErrorsConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxErrorsConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.sanity
    def test_is_not_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxErrorsConstraint does not satisfy
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert not isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxErrorsConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxErrorsConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxErrorsConstraint()
        with pytest.raises(ValidationError):
            MaxErrorsConstraint(max_errors=-1)
        with pytest.raises(ValidationError):
            MaxErrorsConstraint(max_errors=0)
        with pytest.raises(ValidationError):
            MaxErrorsConstraint(max_errors="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        for num_errors in range(int(constructor_args["max_errors"] * 2)):
            created_requests = (num_errors + 1) * 2
            processed_requests = num_errors + 1
            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=created_requests,
                processed_requests=processed_requests,
                errored_requests=num_errors,
            )
            request = ScheduledRequestInfo(
                request_id=f"test-{num_errors}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )
            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)
            errors_exceeded = num_errors >= constructor_args["max_errors"]
            if not errors_exceeded:
                assert action.request_queuing == "continue"
                assert action.request_processing == "continue"
            else:
                assert action.request_queuing == "stop"
                assert action.request_processing == "stop_all"

            assert isinstance(action.metadata, dict)
            assert action.metadata == {
                "max_errors": constructor_args["max_errors"],
                "errors_exceeded": errors_exceeded,
                "current_errors": num_errors,
            }
            assert action.progress == {}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxErrorsConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxErrorsConstraint.model_validate(data)
        assert reconstructed.max_errors == instance.max_errors

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxErrorsConstraintInitializer:
    """Test the MaxErrorsConstraintInitializer implementation."""

    @pytest.fixture(params=[{"max_errors": 10}, {"max_errors": 5.5}, {"max_errors": 1}])
    def valid_instances(self, request):
        """Provide valid instances of MaxErrorsConstraintInitializer."""
        params = request.param
        instance = MaxErrorsConstraintInitializer(**params)
        return instance, params

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self):
        """Test that MaxErrorsConstraintInitializer satisfies the protocol."""
        initializer = MaxErrorsConstraintInitializer(max_errors=10)
        assert isinstance(initializer, ConstraintInitializer)

    @pytest.mark.smoke
    def test_is_not_constraint_protocol(self):
        """
        Test that MaxErrorsConstraintInitializer does not satisfy
        the constraint protocol.
        """
        initializer = MaxErrorsConstraintInitializer(max_errors=10)
        assert not isinstance(initializer, Constraint)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that the initializer can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that the initializer rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxErrorsConstraintInitializer()
        with pytest.raises(ValidationError):
            MaxErrorsConstraintInitializer(max_errors=-1)
        with pytest.raises(ValidationError):
            MaxErrorsConstraintInitializer(max_errors=0)
        with pytest.raises(ValidationError):
            MaxErrorsConstraintInitializer(max_errors="invalid")

    def test_constraint_initialization_functionality(self, valid_instances):
        """Test that the constraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxErrorsConstraint)
        assert constraint.max_errors == constructor_args["max_errors"]

    def test_marshalling(self, valid_instances):
        """
        Test that MaxErrorsConstraintInitializer can be
        serialized and deserialized.
        """
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxErrorsConstraintInitializer.model_validate(data)
        assert reconstructed.max_errors == instance.max_errors

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxErrorRateConstraint:
    """Test the MaxErrorRateConstraint implementation."""

    @pytest.fixture(
        params=[
            {"max_error_rate": 0.1, "window_size": 40},
            {"max_error_rate": 0.5, "window_size": 50},
            {"max_error_rate": 0.05, "window_size": 55},
        ]
    )
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxErrorRateConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxErrorRateConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.sanity
    def test_is_not_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxErrorRateConstraint does not satisfy
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert not isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that MaxErrorRateConstraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxErrorRateConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint()
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=0)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=-1)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=1.5)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate=0.5, window_size=0)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraint(max_error_rate="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions with sliding window behavior"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        max_error_rate = constructor_args["max_error_rate"]
        window_size = constructor_args["window_size"]
        safety_factor = 1.5
        total_errors = 0
        error_window = []

        for request_num in range(window_size * 2):
            error_probability = max_error_rate * safety_factor

            if random.random() < error_probability:
                total_errors += 1
                status = "errored"
                error_window.append(1)
            else:
                status = "completed"
                error_window.append(0)
            error_window = (
                error_window[-window_size:]
                if len(error_window) > window_size
                else error_window
            )

            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=request_num + 1,
                processed_requests=request_num + 1,
            )
            request = ScheduledRequestInfo(
                request_id=f"test-{request_num}",
                status=status,
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)
            error_count = sum(instance.error_window)
            processed_requests = state.processed_requests
            exceeded_min_processed = processed_requests >= window_size
            current_error_rate = (
                error_count / float(min(processed_requests, window_size))
                if processed_requests > 0
                else 0.0
            )
            exceeded_error_rate = current_error_rate >= max_error_rate
            should_stop = exceeded_min_processed and exceeded_error_rate
            expected_queuing = "stop" if should_stop else "continue"
            expected_processing = "stop_all" if should_stop else "continue"

            assert action.request_queuing == expected_queuing
            assert action.request_processing == expected_processing
            assert isinstance(action.metadata, dict)
            assert action.metadata["max_error_rate"] == max_error_rate
            assert action.metadata["window_size"] == window_size
            assert action.metadata["error_count"] == error_count
            assert action.metadata["current_error_rate"] == current_error_rate
            assert action.metadata["exceeded_error_rate"] == exceeded_error_rate
            assert action.progress == {}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxErrorRateConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxErrorRateConstraint.model_validate(data)
        assert reconstructed.max_error_rate == instance.max_error_rate
        assert reconstructed.window_size == instance.window_size

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxErrorRateConstraintInitializer:
    """Test the MaxErrorRateConstraintInitializer implementation."""

    @pytest.fixture(
        params=[
            {"max_error_rate": 0.1, "window_size": 10},
            {"max_error_rate": 0.5, "window_size": 20},
            {"max_error_rate": 0.05, "window_size": 5},
        ]
    )
    def valid_instances(self, request):
        """Provide valid instances of MaxErrorRateConstraintInitializer."""
        params = request.param
        instance = MaxErrorRateConstraintInitializer(**params)
        return instance, params

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self):
        """Test that MaxErrorRateConstraintInitializer satisfies the protocol."""
        initializer = MaxErrorRateConstraintInitializer(
            max_error_rate=0.1, window_size=10
        )
        assert isinstance(initializer, ConstraintInitializer)

    @pytest.mark.smoke
    def test_is_not_constraint_protocol(self):
        """
        Test that MaxErrorRateConstraintInitializer does not satisfy
        the constraint protocol.
        """
        initializer = MaxErrorRateConstraintInitializer(
            max_error_rate=0.1, window_size=10
        )
        assert not isinstance(initializer, Constraint)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that the initializer can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that the initializer rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxErrorRateConstraintInitializer()
        with pytest.raises(ValidationError):
            MaxErrorRateConstraintInitializer(max_error_rate=0)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraintInitializer(max_error_rate=-1)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraintInitializer(max_error_rate=1.5)
        with pytest.raises(ValidationError):
            MaxErrorRateConstraintInitializer(max_error_rate=0.5, window_size=0)

    def test_constraint_initialization_functionality(self, valid_instances):
        """Test that the constraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxErrorRateConstraint)
        assert constraint.max_error_rate == constructor_args["max_error_rate"]
        assert constraint.window_size == constructor_args["window_size"]

    def test_marshalling(self, valid_instances):
        """
        Test that MaxErrorRateConstraintInitializer can be
        serialized and deserialized.
        """
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxErrorRateConstraintInitializer.model_validate(data)
        assert reconstructed.max_error_rate == instance.max_error_rate
        assert reconstructed.window_size == instance.window_size

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxGlobalErrorRateConstraint:
    """Test the MaxGlobalErrorRateConstraint implementation."""

    @pytest.fixture(
        params=[
            {"max_error_rate": 0.1, "min_processed": 50},
            {"max_error_rate": 0.2, "min_processed": 100},
            {"max_error_rate": 0.05, "min_processed": 31},
        ]
    )
    def valid_instances(self, request):
        constructor_args = request.param
        instance = MaxGlobalErrorRateConstraint(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_is_constraint_protocol(self, valid_instances):
        """Test that MaxGlobalErrorRateConstraint satisfies the Constraint protocol."""
        constraint, _ = valid_instances
        assert isinstance(constraint, Constraint)

    @pytest.mark.sanity
    def test_is_not_constraint_initializer_protocol(self, valid_instances):
        """
        Test that MaxGlobalErrorRateConstraint does not satisfy
        the ConstraintInitializer protocol.
        """
        constraint, _ = valid_instances
        assert not isinstance(constraint, ConstraintInitializer)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """
        Test that MaxGlobalErrorRateConstraint can be initialized
        with valid parameters.
        """
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that MaxGlobalErrorRateConstraint rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint()
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=0)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=-1)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=1.5)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate=0.5, min_processed=30)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraint(max_error_rate="invalid")

    @pytest.mark.smoke
    def test_constraint_functionality(self, valid_instances):
        """Test constraint returns correct actions based on global error rate"""
        instance, constructor_args = valid_instances
        start_time = time.time()

        max_error_rate = constructor_args["max_error_rate"]
        min_processed = constructor_args["min_processed"]
        safety_factor = 1.5
        total_requests = min_processed * 2
        total_errors = 0

        for request_num in range(total_requests):
            error_probability = max_error_rate * safety_factor

            if random.random() < error_probability:
                total_errors += 1
                status = "errored"
            else:
                status = "completed"

            processed_requests = request_num + 1

            state = SchedulerState(
                node_id=0,
                num_processes=1,
                start_time=start_time,
                created_requests=processed_requests + 10,
                processed_requests=processed_requests,
                errored_requests=total_errors,
            )
            request = ScheduledRequestInfo(
                request_id=f"test-{request_num}",
                status=status,
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=start_time,
            )

            action = instance(state, request)
            assert isinstance(action, SchedulerUpdateAction)

            exceeded_min_processed = processed_requests >= min_processed
            error_rate = (
                total_errors / float(processed_requests)
                if processed_requests > 0
                else 0.0
            )
            exceeded_error_rate = error_rate >= max_error_rate
            should_stop = exceeded_min_processed and exceeded_error_rate

            expected_queuing = "stop" if should_stop else "continue"
            expected_processing = "stop_all" if should_stop else "continue"

            assert action.request_queuing == expected_queuing
            assert action.request_processing == expected_processing

            assert isinstance(action.metadata, dict)
            assert action.metadata == {
                "max_error_rate": max_error_rate,
                "min_processed": min_processed,
                "processed_requests": processed_requests,
                "errored_requests": total_errors,
                "error_rate": error_rate,
                "exceeded_min_processed": exceeded_min_processed,
                "exceeded_error_rate": exceeded_error_rate,
            }

            # Error constraints don't provide progress information
            assert action.progress == {}

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test that MaxGlobalErrorRateConstraint can be serialized and deserialized."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxGlobalErrorRateConstraint.model_validate(data)
        assert reconstructed.max_error_rate == instance.max_error_rate
        assert reconstructed.min_processed == instance.min_processed

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestMaxGlobalErrorRateConstraintInitializer:
    """Test the MaxGlobalErrorRateConstraintInitializer implementation."""

    @pytest.fixture(
        params=[
            {"max_error_rate": 0.1, "min_processed": 50},
            {"max_error_rate": 0.2, "min_processed": 100},
            {"max_error_rate": 0.05, "min_processed": 31},
        ]
    )
    def valid_instances(self, request):
        """Provide valid instances of MaxGlobalErrorRateConstraintInitializer."""
        params = request.param
        instance = MaxGlobalErrorRateConstraintInitializer(**params)
        return instance, params

    @pytest.mark.smoke
    def test_is_constraint_initializer_protocol(self):
        """Test that MaxGlobalErrorRateConstraintInitializer satisfies the protocol."""
        initializer = MaxGlobalErrorRateConstraintInitializer(
            max_error_rate=0.1, min_processed=50
        )
        assert isinstance(initializer, ConstraintInitializer)

    @pytest.mark.smoke
    def test_is_not_constraint_protocol(self):
        """
        Test that MaxGlobalErrorRateConstraintInitializer does not satisfy
        the constraint protocol.
        """
        initializer = MaxGlobalErrorRateConstraintInitializer(
            max_error_rate=0.1, min_processed=50
        )
        assert not isinstance(initializer, Constraint)

    @pytest.mark.smoke
    def test_initialization_valid(self, valid_instances):
        """Test that the initializer can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        for key, value in constructor_args.items():
            assert hasattr(instance, key)
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    def test_initialization_invalid(self):
        """Test that the initializer rejects invalid parameters."""
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraintInitializer()
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraintInitializer(max_error_rate=0)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraintInitializer(max_error_rate=-1)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraintInitializer(max_error_rate=1.5)
        with pytest.raises(ValidationError):
            MaxGlobalErrorRateConstraintInitializer(
                max_error_rate=0.5, min_processed=30
            )

    def test_constraint_initialization_functionality(self, valid_instances):
        """Test that the constraint can be initialized with valid parameters."""
        instance, constructor_args = valid_instances

        constraint = instance.create_constraint()
        assert isinstance(constraint, MaxGlobalErrorRateConstraint)
        assert constraint.max_error_rate == constructor_args["max_error_rate"]
        assert constraint.min_processed == constructor_args["min_processed"]

    def test_marshalling(self, valid_instances):
        """
        Test that MaxGlobalErrorRateConstraintInitializer can be
        serialized and deserialized.
        """
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = MaxGlobalErrorRateConstraintInitializer.model_validate(data)
        assert reconstructed.max_error_rate == instance.max_error_rate
        assert reconstructed.min_processed == instance.min_processed

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestConstraintsInitializerFactory:
    """Test the ConstraintsInitializerFactory implementation."""

    EXPECTED_REGISTERED_KEYS = {
        "max_number": MaxNumberConstraintInitializer,
        "max_duration": MaxDurationConstraintInitializer,
        "max_errors": MaxErrorsConstraintInitializer,
        "max_error_rate": MaxErrorRateConstraintInitializer,
        "max_global_error_rate": MaxGlobalErrorRateConstraintInitializer,
    }

    @pytest.mark.smoke
    def test_registered_constraint_keys(self):
        """Test that all expected constraint keys are registered and no others."""
        registered_keys = set(ConstraintsInitializerFactory.registered_objects().keys())
        expected_keys = set(self.EXPECTED_REGISTERED_KEYS.keys())

        assert registered_keys == expected_keys, (
            f"Registered keys {registered_keys} do not match "
            f"expected keys {expected_keys}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("key", "expected_class"),
        [
            ("max_number", MaxNumberConstraintInitializer),
            ("max_duration", MaxDurationConstraintInitializer),
            ("max_errors", MaxErrorsConstraintInitializer),
            ("max_error_rate", MaxErrorRateConstraintInitializer),
            ("max_global_error_rate", MaxGlobalErrorRateConstraintInitializer),
        ],
    )
    def test_registered_constraint_classes(self, key, expected_class):
        """Test that each registered key maps to the expected initializer class."""
        assert ConstraintsInitializerFactory.is_registered(key)
        registered_class = ConstraintsInitializerFactory.get_registered_object(key)
        assert registered_class == expected_class

    @pytest.mark.sanity
    def test_unregistered_key_fails(self):
        """Test that unregistered keys raise ValueError."""
        unregistered_key = "nonexistent_constraint"
        assert not ConstraintsInitializerFactory.is_registered(unregistered_key)

        with pytest.raises(
            ValueError, match=f"Unknown constraint initializer key: {unregistered_key}"
        ):
            ConstraintsInitializerFactory.create(unregistered_key)

        with pytest.raises(
            ValueError, match=f"Unknown constraint initializer key: {unregistered_key}"
        ):
            ConstraintsInitializerFactory.create_constraint(unregistered_key)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("key", "init_args", "expected_constraint_class"),
        [
            ("max_number", {"max_num": 100}, MaxNumberConstraint),
            ("max_duration", {"max_duration": 30.0}, MaxDurationConstraint),
            ("max_errors", {"max_errors": 5}, MaxErrorsConstraint),
            (
                "max_error_rate",
                {"max_error_rate": 0.1, "window_size": 50},
                MaxErrorRateConstraint,
            ),
            (
                "max_global_error_rate",
                {"max_error_rate": 0.05, "min_processed": 100},
                MaxGlobalErrorRateConstraint,
            ),
        ],
    )
    def test_create_initializer(self, key, init_args, expected_constraint_class):
        """Test that create method returns properly configured initializers."""
        initializer = ConstraintsInitializerFactory.create(key, **init_args)

        assert isinstance(initializer, ConstraintInitializer)
        assert isinstance(initializer, self.EXPECTED_REGISTERED_KEYS[key])

        for attr_name, attr_value in init_args.items():
            assert hasattr(initializer, attr_name)
            assert getattr(initializer, attr_name) == attr_value

        constraint = initializer.create_constraint()
        assert isinstance(constraint, Constraint)
        assert isinstance(constraint, expected_constraint_class)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("key", "init_args", "expected_constraint_class"),
        [
            ("max_number", {"max_num": 100}, MaxNumberConstraint),
            ("max_duration", {"max_duration": 30.0}, MaxDurationConstraint),
            ("max_errors", {"max_errors": 5}, MaxErrorsConstraint),
            (
                "max_error_rate",
                {"max_error_rate": 0.1, "window_size": 50},
                MaxErrorRateConstraint,
            ),
            (
                "max_global_error_rate",
                {"max_error_rate": 0.05, "min_processed": 100},
                MaxGlobalErrorRateConstraint,
            ),
        ],
    )
    def test_create_constraint_direct(self, key, init_args, expected_constraint_class):
        """Test that create_constraint method returns configured constraints."""
        constraint = ConstraintsInitializerFactory.create_constraint(key, **init_args)

        assert isinstance(constraint, Constraint)
        assert isinstance(constraint, expected_constraint_class)

        for attr_name, attr_value in init_args.items():
            assert hasattr(constraint, attr_name)
            assert getattr(constraint, attr_name) == attr_value

    @pytest.mark.smoke
    def test_resolve_with_constraint_instances(self):
        """Test resolve method with pre-instantiated Constraint objects."""
        max_num_constraint = MaxNumberConstraint(max_num=50)
        max_duration_constraint = MaxDurationConstraint(max_duration=60.0)

        initializers = {
            "max_number": max_num_constraint,
            "max_duration": max_duration_constraint,
        }

        resolved = ConstraintsInitializerFactory.resolve(initializers)

        assert len(resolved) == 2
        assert resolved["max_number"] is max_num_constraint
        assert resolved["max_duration"] is max_duration_constraint
        assert all(isinstance(c, Constraint) for c in resolved.values())

    @pytest.mark.smoke
    def test_resolve_with_initializer_instances(self):
        """Test resolve method with pre-instantiated ConstraintInitializer objects."""
        max_num_initializer = MaxNumberConstraintInitializer(max_num=75)
        max_errors_initializer = MaxErrorsConstraintInitializer(max_errors=10)

        initializers = {
            "max_number": max_num_initializer,
            "max_errors": max_errors_initializer,
        }

        resolved = ConstraintsInitializerFactory.resolve(initializers)

        assert len(resolved) == 2
        assert isinstance(resolved["max_number"], MaxNumberConstraint)
        assert isinstance(resolved["max_errors"], MaxErrorsConstraint)
        assert resolved["max_number"].max_num == 75
        assert resolved["max_errors"].max_errors == 10
        assert all(isinstance(c, Constraint) for c in resolved.values())

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("input_spec", "expected_values"),
        [
            (
                {
                    "max_number": {"max_num": 200},
                    "max_duration": {"max_duration": 45.0},
                    "max_errors": {"max_errors": 3},
                },
                {
                    "max_number": ("max_num", 200),
                    "max_duration": ("max_duration", 45.0),
                    "max_errors": ("max_errors", 3),
                },
            ),
            (
                {
                    "max_error_rate": {"max_error_rate": 0.15, "window_size": 100},
                    "max_global_error_rate": {
                        "max_error_rate": 0.08,
                        "min_processed": 50,
                    },
                },
                {
                    "max_error_rate": ("max_error_rate", 0.15),
                    "max_global_error_rate": ("max_error_rate", 0.08),
                },
            ),
        ],
    )
    def test_resolve_with_dict_configs(self, input_spec, expected_values):
        """Test resolve method with dictionary configurations."""
        resolved = ConstraintsInitializerFactory.resolve(input_spec)

        assert len(resolved) == len(input_spec)
        assert all(isinstance(c, Constraint) for c in resolved.values())

        for key, (attr_name, attr_value) in expected_values.items():
            assert key in resolved
            constraint = resolved[key]
            assert hasattr(constraint, attr_name)
            assert getattr(constraint, attr_name) == attr_value

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("input_spec", "expected_values"),
        [
            (
                {"max_number": 150},
                {"max_number": ("max_num", 150)},
            ),
            (
                {"max_duration": 90.0},
                {"max_duration": ("max_duration", 90.0)},
            ),
            (
                {"max_errors": 8},
                {"max_errors": ("max_errors", 8)},
            ),
            (
                {"max_error_rate": 0.15},
                {"max_error_rate": ("max_error_rate", 0.15)},
            ),
            (
                {"max_global_error_rate": 0.05},
                {"max_global_error_rate": ("max_error_rate", 0.05)},
            ),
        ],
    )
    def test_resolve_with_simple_values(self, input_spec, expected_values):
        """Test that resolve method now supports simple scalar values."""
        resolved = ConstraintsInitializerFactory.resolve(input_spec)

        assert len(resolved) == len(input_spec)
        assert all(isinstance(c, Constraint) for c in resolved.values())

        for key, (attr_name, attr_value) in expected_values.items():
            assert key in resolved
            constraint = resolved[key]
            assert hasattr(constraint, attr_name)
            assert getattr(constraint, attr_name) == attr_value

    @pytest.mark.smoke
    def test_resolve_mixed_types(self):
        """Test resolve method with mixed constraint types including simple values."""
        max_num_constraint = MaxNumberConstraint(max_num=25)
        max_duration_initializer = MaxDurationConstraintInitializer(max_duration=120.0)

        mixed_spec = {
            "max_number": max_num_constraint,
            "max_duration": max_duration_initializer,
            "max_errors": {"max_errors": 15},
            "max_error_rate": 0.08,
            "max_global_error_rate": {"max_error_rate": 0.12},
        }

        resolved = ConstraintsInitializerFactory.resolve(mixed_spec)

        assert len(resolved) == 5
        assert all(isinstance(c, Constraint) for c in resolved.values())
        assert resolved["max_number"] is max_num_constraint
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert isinstance(resolved["max_errors"], MaxErrorsConstraint)
        assert isinstance(resolved["max_error_rate"], MaxErrorRateConstraint)
        assert isinstance(
            resolved["max_global_error_rate"], MaxGlobalErrorRateConstraint
        )

        assert resolved["max_error_rate"].max_error_rate == 0.08

    @pytest.mark.sanity
    def test_resolve_constraints_method_bug_fixed(self):
        """Test resolve_constraints method now works correctly after bug fix.

        Note: Previously resolve_constraints had a bug where the parameter name
        'constraints' was shadowed by a local variable, causing it to
        always return an empty dictionary. This bug has been fixed.
        """
        max_num_constraint = MaxNumberConstraint(max_num=80)

        constraints_spec = {
            "max_number": max_num_constraint,
            "max_duration": {"max_duration": 300.0},
        }

        resolved = ConstraintsInitializerFactory.resolve_constraints(constraints_spec)

        assert len(resolved) == 2
        assert resolved["max_number"] is max_num_constraint
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert resolved["max_duration"].max_duration == 300.0

    @pytest.mark.sanity
    def test_resolve_with_invalid_key(self):
        """Test that resolve raises ValueError for unregistered keys."""
        invalid_spec = {
            "max_number": {"max_num": 100},
            "invalid_constraint": {"some_param": 42},
        }

        with pytest.raises(
            ValueError, match="Unknown constraint initializer key: invalid_constraint"
        ):
            ConstraintsInitializerFactory.resolve(invalid_spec)

    @pytest.mark.sanity
    def test_resolve_constraints_with_invalid_key_now_raises(self):
        """Test that resolve_constraints now properly validates keys after bug fix.

        Note: Previously due to the variable shadowing bug in resolve_constraints,
        it didn't actually process the input and therefore didn't validate keys,
        always returning an empty dictionary. Now it properly validates.
        """
        invalid_spec = {
            "max_duration": {"max_duration": 60.0},
            "nonexistent_key": {"param": "value"},
        }

        with pytest.raises(ValueError, match="Unknown constraint initializer key"):
            ConstraintsInitializerFactory.resolve_constraints(invalid_spec)

    @pytest.mark.smoke
    def test_functional_constraint_creation(self):
        """Test that created constraints are functionally correct."""
        constraint = ConstraintsInitializerFactory.create_constraint(
            "max_number", max_num=10
        )
        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=5,
            processed_requests=5,
        )
        request = ScheduledRequestInfo(
            request_id="test-request",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"

        state_exceeded = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=15,
            processed_requests=15,
        )
        action_exceeded = constraint(state_exceeded, request)
        assert action_exceeded.request_queuing == "stop"
        assert action_exceeded.request_processing == "stop_local"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("initializer_class", "key", "value", "expected_attr"),
        [
            (MaxNumberConstraintInitializer, "max_number", 100, "max_num"),
            (MaxDurationConstraintInitializer, "max_duration", 45.0, "max_duration"),
            (MaxErrorsConstraintInitializer, "max_errors", 5, "max_errors"),
            (
                MaxErrorRateConstraintInitializer,
                "max_error_rate",
                0.1,
                "max_error_rate",
            ),
            (
                MaxGlobalErrorRateConstraintInitializer,
                "max_global_error_rate",
                0.05,
                "max_error_rate",
            ),
        ],
    )
    def test_from_simple_value_class_method(
        self, initializer_class, key, value, expected_attr
    ):
        """Test that each initializer class properly handles from_simple_value."""
        initializer = initializer_class.from_simple_value(value)
        assert hasattr(initializer, expected_attr)
        assert getattr(initializer, expected_attr) == value

        constraint = initializer.create_constraint()
        assert hasattr(constraint, expected_attr)
        assert getattr(constraint, expected_attr) == value

        factory_result = ConstraintsInitializerFactory.resolve({key: value})
        assert key in factory_result
        factory_constraint = factory_result[key]
        assert hasattr(factory_constraint, expected_attr)
        assert getattr(factory_constraint, expected_attr) == value
