from __future__ import annotations

import inspect
import typing
from abc import ABC
from collections.abc import AsyncIterator
from typing import Any, Optional, TypeVar, Union

import pytest
from pydantic import ValidationError
from typing_extensions import TypeAliasType

from guidellm.scheduler import (
    BackendInterface,
    BackendT,
    MeasuredRequestTimings,
    MeasuredRequestTimingsT,
    MultiTurnRequestT,
    RequestSchedulerTimings,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
    SchedulerUpdateAction,
    SchedulerUpdateActionProgress,
)
from guidellm.utils import StandardBaseModel


def test_request_t():
    """Validate that RequestT is a TypeVar usable for generics and isn't bound."""
    assert isinstance(RequestT, TypeVar)
    assert RequestT.__name__ == "RequestT"
    assert RequestT.__bound__ is None
    assert RequestT.__constraints__ == ()


def test_response_t():
    """Validate that ResponseT is a TypeVar usable for generics and isn't bound."""
    assert isinstance(ResponseT, TypeVar)
    assert ResponseT.__name__ == "ResponseT"
    assert ResponseT.__bound__ is None
    assert ResponseT.__constraints__ == ()


def test_request_timings_t():
    """Validate MeasuredRequestTimingsT is a TypeVar bound to MeasuredRequestTimings."""
    assert isinstance(MeasuredRequestTimingsT, TypeVar)
    assert MeasuredRequestTimingsT.__name__ == "MeasuredRequestTimingsT"
    assert MeasuredRequestTimingsT.__bound__ == MeasuredRequestTimings
    assert MeasuredRequestTimingsT.__constraints__ == ()


def test_backend_t():
    """Validate that BackendT is a TypeVar bound to BackendInterface."""
    assert isinstance(BackendT, TypeVar)
    assert BackendT.__name__ == "BackendT"
    assert BackendT.__bound__.__name__ == "BackendInterface"
    assert BackendT.__constraints__ == ()


def test_multi_turn_request_t():
    """Validate MultiTurnRequestT is a TypeAliasType for multi-turn requests."""
    assert isinstance(MultiTurnRequestT, TypeAliasType)
    assert MultiTurnRequestT.__name__ == "MultiTurnRequestT"

    value = MultiTurnRequestT.__value__
    assert hasattr(value, "__origin__")
    assert value.__origin__ is Union

    type_params = getattr(MultiTurnRequestT, "__type_params__", ())
    assert len(type_params) == 1
    assert type_params[0].__name__ == "RequestT"


class TestBackendInterface:
    """Test the BackendInterface abstract base class."""

    @pytest.mark.smoke
    def test_is_abstract_base_class(self):
        """Test that BackendInterface is an ABC and cannot be instantiated directly."""
        assert issubclass(BackendInterface, ABC)
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BackendInterface()

    @pytest.mark.smoke
    def test_abstract_methods_defined(self):
        """Test that all expected abstract methods are defined."""
        expected_methods = {
            "info",
            "process_startup",
            "validate",
            "process_shutdown",
            "resolve",
        }
        expected_properties = {
            "processes_limit",
            "requests_limit",
        }

        for method_name in expected_methods:
            assert hasattr(BackendInterface, method_name)
            method = getattr(BackendInterface, method_name)
            assert inspect.isfunction(method) or inspect.ismethod(method)

        for prop_name in expected_properties:
            assert hasattr(BackendInterface, prop_name)
            prop = getattr(BackendInterface, prop_name)
            assert hasattr(prop, "__get__")

    @pytest.mark.smoke
    def test_generic_type_parameters(self):
        """Test that BackendInterface has the correct generic type parameters."""
        orig_bases = BackendInterface.__orig_bases__
        abc_base = None
        generic_base = None

        for base in orig_bases:
            if hasattr(base, "__origin__"):
                if base.__origin__ is typing.Generic:
                    generic_base = base
            elif base.__name__ == "ABC":
                abc_base = base

        assert abc_base is not None, "Should inherit from ABC"
        assert generic_base is not None, "Should inherit from Generic"

        if hasattr(generic_base, "__args__"):
            type_params = generic_base.__args__
            assert len(type_params) == 3, "Should have 3 type parameters"
            param_names = [param.__name__ for param in type_params]
            expected_names = ["RequestT", "MeasuredRequestTimingsT", "ResponseT"]
            assert param_names == expected_names

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that a concrete implementation must implement all abstract methods."""

        class PartialBackend(BackendInterface):
            @property
            def processes_limit(self):
                return 1

            @property
            def requests_limit(self):
                return 10

            def info(self):
                return {}

            async def process_startup(self):
                pass

            # Missing: validate, process_shutdown, resolve

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PartialBackend()

    @pytest.mark.smoke
    def test_implementation_construction(self):
        """Test that a complete concrete implementation can be instantiated."""

        class ConcreteBackend(BackendInterface[str, MeasuredRequestTimings, str]):
            @property
            def processes_limit(self) -> int | None:
                return 4

            @property
            def requests_limit(self) -> int | None:
                return 100

            def info(self) -> dict[str, Any]:
                return {"model": "test", "version": "1.0"}

            async def process_startup(self) -> None:
                pass

            async def validate(self) -> None:
                pass

            async def process_shutdown(self) -> None:
                pass

            async def resolve(
                self,
                request: str,
                request_info: ScheduledRequestInfo[MeasuredRequestTimings],
                history: list[tuple[str, str]] | None = None,
            ) -> AsyncIterator[
                tuple[str, ScheduledRequestInfo[MeasuredRequestTimings]]
            ]:
                yield f"Response to: {request}", request_info

        backend = ConcreteBackend()
        assert isinstance(backend, BackendInterface)
        assert isinstance(backend, ConcreteBackend)
        assert backend.processes_limit == 4
        assert backend.requests_limit == 100
        info = backend.info()
        assert info == {"model": "test", "version": "1.0"}

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_implementation_async_methods(self):
        """Test that async methods work correctly in concrete implementation."""

        class AsyncBackend(BackendInterface[dict, MeasuredRequestTimings, dict]):
            def __init__(self):
                self.startup_called = False
                self.validate_called = False
                self.shutdown_called = False

            @property
            def processes_limit(self) -> int | None:
                return None  # Unlimited

            @property
            def requests_limit(self) -> int | None:
                return None  # Unlimited

            def info(self) -> dict[str, Any]:
                return {"backend": "async_test"}

            async def process_startup(self) -> None:
                self.startup_called = True

            async def validate(self) -> None:
                self.validate_called = True

            async def process_shutdown(self) -> None:
                self.shutdown_called = True

            async def resolve(
                self,
                request: dict,
                request_info: ScheduledRequestInfo[MeasuredRequestTimings],
                history: list[tuple[dict, dict]] | None = None,
            ) -> AsyncIterator[
                tuple[dict, ScheduledRequestInfo[MeasuredRequestTimings]]
            ]:
                response = {"result": request.get("input", ""), "status": "success"}
                yield response, request_info

        backend = AsyncBackend()
        await backend.process_startup()
        assert backend.startup_called

        await backend.validate()
        assert backend.validate_called

        await backend.process_shutdown()
        assert backend.shutdown_called

        request = {"input": "test_request"}
        request_info = ScheduledRequestInfo(
            request_id="test-123",
            status="queued",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
        )
        results = []
        async for response, updated_info in backend.resolve(request, request_info):
            results.append((response, updated_info))

        assert len(results) == 1
        response, updated_info = results[0]
        assert response == {"result": "test_request", "status": "success"}
        assert updated_info == request_info

    @pytest.mark.smoke
    def test_method_signatures(self):
        """Test that abstract methods have the expected signatures."""
        info_sig = inspect.signature(BackendInterface.info)
        assert len(info_sig.parameters) == 1
        assert list(info_sig.parameters.keys()) == ["self"]

        startup_sig = inspect.signature(BackendInterface.process_startup)
        assert len(startup_sig.parameters) == 1  # Only self
        assert list(startup_sig.parameters.keys()) == ["self"]

        validate_sig = inspect.signature(BackendInterface.validate)
        assert len(validate_sig.parameters) == 1  # Only self
        assert list(validate_sig.parameters.keys()) == ["self"]

        shutdown_sig = inspect.signature(BackendInterface.process_shutdown)
        assert len(shutdown_sig.parameters) == 1  # Only self
        assert list(shutdown_sig.parameters.keys()) == ["self"]

        resolve_sig = inspect.signature(BackendInterface.resolve)
        expected_params = ["self", "request", "request_info", "history"]
        assert list(resolve_sig.parameters.keys()) == expected_params

        history_param = resolve_sig.parameters["history"]
        assert history_param.default is None


class TestRequestSchedulerTimings:
    """Test the RequestSchedulerTimings model class."""

    CHECK_KEYS = [
        "targeted_start",
        "queued",
        "dequeued",
        "resolve_start",
        "resolve_end",
        "finalized",
    ]

    @pytest.fixture(
        params=[
            {},
            {
                "targeted_start": None,
                "queued": None,
                "dequeued": None,
                "resolve_start": None,
                "resolve_end": None,
                "finalized": None,
            },
            {
                "targeted_start": 1000.0,
                "queued": 200.0,
                "dequeued": 800.0,
                "resolve_start": 1000.5,
                "resolve_end": 1100.0,
                "finalized": 1100.5,
            },
            {
                "queued": 200.0,
                "resolve_start": 1000.5,
                "resolve_end": 1100.0,
            },
            {
                "targeted_start": 0.0,
                "queued": 0.0,
                "dequeued": 0.0,
                "resolve_start": 0.0,
                "resolve_end": 0.0,
                "finalized": 0.0,
            },
        ],
        ids=[
            "default_empty",
            "all_none_explicit",
            "complete_sequence",
            "partial_data",
            "zero_timestamps",
        ],
    )
    def valid_instances(self, request):
        """Creates various valid configurations of RequestSchedulerTimings."""
        constructor_args = request.param
        instance = RequestSchedulerTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test RequestSchedulerTimings inheritance and type relationships."""
        assert issubclass(RequestSchedulerTimings, StandardBaseModel)
        assert hasattr(RequestSchedulerTimings, "model_dump")
        assert hasattr(RequestSchedulerTimings, "model_validate")

        # Check all expected fields are defined
        fields = RequestSchedulerTimings.model_fields
        for key in self.CHECK_KEYS:
            assert key in fields
            field_info = fields[key]
            assert field_info.annotation in (Union[float, None], Optional[float])
            assert field_info.default is None

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, RequestSchedulerTimings)
        for key in self.CHECK_KEYS:
            assert hasattr(instance, key)

        # Validate that the instance attributes match the constructor args
        for field, expected_value in constructor_args.items():
            assert getattr(instance, field) == expected_value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("targeted_start", "invalid_string"),
            ("queued", "invalid_string"),
            ("dequeued", [1, 2, 3]),
            ("resolve_start", {"key": "value"}),
            ("resolve_end", [1, 2, 3]),
            ("finalized", object()),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            RequestSchedulerTimings(**kwargs)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data = instance.model_dump()
        assert isinstance(data, dict)
        assert all(key in data for key in self.CHECK_KEYS)

        # Test model_validate
        reconstructed = RequestSchedulerTimings.model_validate(data)
        assert isinstance(reconstructed, RequestSchedulerTimings)

        # Validate that all fields match between original and reconstructed instances
        for field in self.CHECK_KEYS:
            assert getattr(reconstructed, field) == getattr(instance, field)

        # Validate that the reconstructed instance matches original constructor args
        for field, expected_value in constructor_args.items():
            assert getattr(reconstructed, field) == expected_value


class TestRequestTimings:
    """Test the MeasuredRequestTimings model class."""

    CHECK_KEYS = [
        "request_start",
        "request_end",
    ]

    @pytest.fixture(
        params=[
            {},
            {
                "request_start": None,
                "request_end": None,
            },
            {
                "request_start": 1000.0,
                "request_end": 1100.0,
            },
            {
                "request_start": 1000.0,
            },
            {
                "request_start": 0.0,
                "request_end": 0.0,
            },
        ],
        ids=[
            "default_empty",
            "all_none_explicit",
            "complete_sequence",
            "partial_data",
            "zero_timestamps",
        ],
    )
    def valid_instances(self, request):
        """Creates various valid configurations of MeasuredRequestTimings."""
        constructor_args = request.param
        instance = MeasuredRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test MeasuredRequestTimings inheritance and type relationships."""
        assert issubclass(MeasuredRequestTimings, StandardBaseModel)
        assert hasattr(MeasuredRequestTimings, "model_dump")
        assert hasattr(MeasuredRequestTimings, "model_validate")

        # Check all expected fields are defined
        fields = MeasuredRequestTimings.model_fields
        for key in self.CHECK_KEYS:
            assert key in fields
            field_info = fields[key]
            assert field_info.annotation in (Union[float, None], Optional[float])
            assert field_info.default is None

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, MeasuredRequestTimings)
        for key in self.CHECK_KEYS:
            assert hasattr(instance, key)

        # Validate that the instance attributes match the constructor args
        for field, expected_value in constructor_args.items():
            assert getattr(instance, field) == expected_value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("request_start", "invalid_string"),
            ("request_end", [1, 2, 3]),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            MeasuredRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data = instance.model_dump()
        assert isinstance(data, dict)
        assert all(key in data for key in self.CHECK_KEYS)

        # Test model_validate
        reconstructed = MeasuredRequestTimings.model_validate(data)
        assert isinstance(reconstructed, MeasuredRequestTimings)

        # Validate that all fields match between original and reconstructed instances
        for field in self.CHECK_KEYS:
            assert getattr(reconstructed, field) == getattr(instance, field)

        # Validate that the reconstructed instance matches original constructor args
        for field, expected_value in constructor_args.items():
            assert getattr(reconstructed, field) == expected_value


class TestScheduledRequestInfo:
    CHECK_KEYS = [
        "request_id",
        "status",
        "error",
        "scheduler_node_id",
        "scheduler_process_id",
        "scheduler_start_time",
        "scheduler_timings",
        "request_timings",
    ]

    @pytest.fixture(
        params=[
            # Minimal required configuration
            {
                "request_id": "test-req-123",
                "status": "queued",
                "scheduler_node_id": 1,
                "scheduler_process_id": 0,
                "scheduler_start_time": 1000.0,
            },
            # Complete configuration with all fields
            {
                "request_id": "test-req-456",
                "status": "completed",
                "error": None,
                "scheduler_node_id": 2,
                "scheduler_process_id": 1,
                "scheduler_start_time": 2000.0,
                "scheduler_timings": {
                    "targeted_start": 1900.0,
                    "queued": 1950.0,
                    "dequeued": 2000.0,
                    "resolve_start": 2050.0,
                    "resolve_end": 2100.0,
                    "finalized": 2150.0,
                },
                "request_timings": {
                    "request_start": 2060.0,
                    "request_end": 2110.0,
                },
            },
            # Error state configuration
            {
                "request_id": "test-req-error",
                "status": "errored",
                "error": "Connection timeout",
                "scheduler_node_id": 0,
                "scheduler_process_id": 0,
                "scheduler_start_time": 3000.0,
            },
            # Different status values
            {
                "request_id": "test-req-pending",
                "status": "pending",
                "scheduler_node_id": 1,
                "scheduler_process_id": 2,
                "scheduler_start_time": 4000.0,
            },
            {
                "request_id": "test-req-in-progress",
                "status": "in_progress",
                "scheduler_node_id": 2,
                "scheduler_process_id": 1,
                "scheduler_start_time": 5000.0,
            },
        ],
        ids=[
            "minimal_required",
            "complete_configuration",
            "error_state",
            "pending_status",
            "in_progress_status",
        ],
    )
    def valid_instances(self, request):
        """Creates various valid configurations of ScheduledRequestInfo.

        Returns:
            tuple: (instance, constructor_args) where instance is the constructed
                   ScheduledRequestInfo and constructor_args are the kwargs used.
        """
        constructor_args = request.param.copy()

        # Handle nested objects
        if "scheduler_timings" in constructor_args:
            constructor_args["scheduler_timings"] = RequestSchedulerTimings(
                **constructor_args["scheduler_timings"]
            )
        if "request_timings" in constructor_args:
            constructor_args["request_timings"] = MeasuredRequestTimings(
                **constructor_args["request_timings"]
            )

        instance = ScheduledRequestInfo(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ScheduledRequestInfo inheritance and type relationships."""
        assert issubclass(ScheduledRequestInfo, StandardBaseModel)
        assert issubclass(ScheduledRequestInfo, typing.Generic)
        assert hasattr(ScheduledRequestInfo, "model_dump")
        assert hasattr(ScheduledRequestInfo, "model_validate")

        # Check computed properties
        assert hasattr(ScheduledRequestInfo, "started_at")
        assert hasattr(ScheduledRequestInfo, "completed_at")
        assert isinstance(ScheduledRequestInfo.started_at, property)
        assert isinstance(ScheduledRequestInfo.completed_at, property)

        # Check that it's properly generic
        orig_bases = getattr(ScheduledRequestInfo, "__orig_bases__", ())
        generic_base = next(
            (
                base
                for base in orig_bases
                if hasattr(base, "__origin__") and base.__origin__ is typing.Generic
            ),
            None,
        )
        assert generic_base is not None

        # Check required fields
        fields = ScheduledRequestInfo.model_fields
        for key in self.CHECK_KEYS:
            assert key in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ScheduledRequestInfo)
        for key in self.CHECK_KEYS:
            assert hasattr(instance, key)

        # Validate that the instance attributes match the constructor args
        for field, expected_value in constructor_args.items():
            if field in ["scheduler_timings", "request_timings"]:
                actual_value = getattr(instance, field)
                if expected_value is None:
                    assert actual_value is None or (
                        field == "scheduler_timings"
                        and isinstance(actual_value, RequestSchedulerTimings)
                    )
                else:
                    assert isinstance(actual_value, type(expected_value))
            else:
                assert getattr(instance, field) == expected_value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("request_id", None),  # Required field
            ("request_id", 123),  # Wrong type
            ("status", "invalid_status"),  # Invalid literal
            ("scheduler_node_id", "not_an_int"),
            ("scheduler_process_id", -1.5),
            ("scheduler_start_time", "not_a_float"),
            ("error", 123),  # Should be string or None
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        # Start with valid base config
        base_kwargs = {
            "request_id": "test-req",
            "status": "queued",
            "scheduler_node_id": 1,
            "scheduler_process_id": 0,
            "scheduler_start_time": 1000.0,
        }
        base_kwargs[field] = value
        with pytest.raises(ValidationError):
            ScheduledRequestInfo(**base_kwargs)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data = instance.model_dump()
        assert isinstance(data, dict)
        assert all(key in data for key in self.CHECK_KEYS)

        # Test model_validate
        reconstructed = ScheduledRequestInfo.model_validate(data)
        assert isinstance(reconstructed, ScheduledRequestInfo)

        # Validate that all fields match between original and reconstructed instances
        for field in self.CHECK_KEYS:
            original_value = getattr(instance, field)
            reconstructed_value = getattr(reconstructed, field)

            if field in ["scheduler_timings", "request_timings"]:
                if original_value is not None and reconstructed_value is not None:
                    assert (
                        original_value.model_dump() == reconstructed_value.model_dump()
                    )
                else:
                    assert original_value is None or isinstance(
                        original_value,
                        (RequestSchedulerTimings, MeasuredRequestTimings),
                    )
                    assert reconstructed_value is None or isinstance(
                        reconstructed_value,
                        (RequestSchedulerTimings, MeasuredRequestTimings),
                    )
            else:
                assert original_value == reconstructed_value

    @pytest.mark.smoke
    def test_started_at_property(self):
        """Test the started_at property logic."""
        # Test with request_timings.request_start (should take precedence)
        instance = ScheduledRequestInfo(
            request_id="test-req",
            status="completed",
            scheduler_node_id=1,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
            scheduler_timings=RequestSchedulerTimings(resolve_start=2000.0),
            request_timings=MeasuredRequestTimings(request_start=2100.0),
        )
        assert instance.started_at == 2100.0

        # Test with only scheduler_timings.resolve_start
        instance = ScheduledRequestInfo(
            request_id="test-req",
            status="completed",
            scheduler_node_id=1,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
            scheduler_timings=RequestSchedulerTimings(resolve_start=2000.0),
        )
        assert instance.started_at == 2000.0

        # Test with no timing info
        instance = ScheduledRequestInfo(
            request_id="test-req",
            status="queued",
            scheduler_node_id=1,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
        )
        assert instance.started_at is None

    @pytest.mark.smoke
    def test_completed_at_property(self):
        """Test the completed_at property logic."""
        # Test with request_timings.request_end (should take precedence)
        instance = ScheduledRequestInfo(
            request_id="test-req",
            status="completed",
            scheduler_node_id=1,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
            scheduler_timings=RequestSchedulerTimings(resolve_end=2000.0),
            request_timings=MeasuredRequestTimings(request_end=2100.0),
        )
        assert instance.completed_at == 2100.0

        # Test with only scheduler_timings.resolve_end
        instance = ScheduledRequestInfo(
            request_id="test-req",
            status="completed",
            scheduler_node_id=1,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
            scheduler_timings=RequestSchedulerTimings(resolve_end=2000.0),
        )
        assert instance.completed_at == 2000.0

        # Test with no timing info
        instance = ScheduledRequestInfo(
            request_id="test-req",
            status="queued",
            scheduler_node_id=1,
            scheduler_process_id=0,
            scheduler_start_time=1000.0,
        )
        assert instance.completed_at is None


class TestSchedulerState:
    CHECK_KEYS = [
        "node_id",
        "num_processes",
        "start_time",
        "end_time",
        "end_queuing_time",
        "end_queuing_constraints",
        "end_processing_time",
        "end_processing_constraints",
        "scheduler_constraints",
        "remaining_fraction",
        "remaining_requests",
        "remaining_duration",
        "created_requests",
        "queued_requests",
        "pending_requests",
        "processing_requests",
        "processed_requests",
        "successful_requests",
        "errored_requests",
        "cancelled_requests",
    ]

    @pytest.fixture(
        params=[
            # Minimal required configuration
            {
                "node_id": 0,
                "num_processes": 1,
                "start_time": 1000.0,
            },
            # Complete configuration with all fields
            {
                "node_id": 1,
                "num_processes": 4,
                "start_time": 2000.0,
                "end_time": 3000.0,
                "end_queuing_time": 2500.0,
                "end_queuing_constraints": {
                    "time_limit": SchedulerUpdateAction(
                        request_queuing="stop", metadata={"max_duration": 1500}
                    )
                },
                "end_processing_time": 2800.0,
                "end_processing_constraints": {
                    "request_limit": SchedulerUpdateAction(
                        request_processing="stop_all", metadata={"max_requests": 1000}
                    )
                },
                "scheduler_constraints": {
                    "rate_limit": SchedulerUpdateAction(metadata={"max_rps": 100})
                },
                "remaining_fraction": 0.25,
                "remaining_requests": 50,
                "remaining_duration": 300.0,
                "created_requests": 200,
                "queued_requests": 180,
                "pending_requests": 20,
                "processing_requests": 10,
                "processed_requests": 150,
                "successful_requests": 140,
                "errored_requests": 8,
                "cancelled_requests": 2,
            },
            # Partial configuration with some stats
            {
                "node_id": 2,
                "num_processes": 2,
                "start_time": 4000.0,
                "created_requests": 50,
                "processed_requests": 30,
                "successful_requests": 28,
                "errored_requests": 2,
            },
            # Edge case: zero values
            {
                "node_id": 0,
                "num_processes": 1,
                "start_time": 0.0,
                "created_requests": 0,
                "processed_requests": 0,
                "successful_requests": 0,
            },
        ],
        ids=[
            "minimal_required",
            "complete_configuration",
            "partial_stats",
            "zero_values",
        ],
    )
    def valid_instances(self, request):
        """Creates various valid configurations of SchedulerState.

        Returns:
            tuple: (instance, constructor_args) where instance is the constructed
                   SchedulerState and constructor_args are the kwargs used.
        """
        constructor_args = request.param
        instance = SchedulerState(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SchedulerState inheritance and type relationships."""
        assert issubclass(SchedulerState, StandardBaseModel)
        assert hasattr(SchedulerState, "model_dump")
        assert hasattr(SchedulerState, "model_validate")

        # Check all expected fields are defined
        fields = SchedulerState.model_fields
        for key in self.CHECK_KEYS:
            assert key in fields

        # Check field defaults for key counters
        counter_fields = [
            "created_requests",
            "queued_requests",
            "pending_requests",
            "processing_requests",
            "processed_requests",
            "successful_requests",
            "errored_requests",
            "cancelled_requests",
        ]
        for field in counter_fields:
            field_info = fields[field]
            assert field_info.default == 0

        # Check that start_time has a default factory
        start_time_field = fields["start_time"]
        assert start_time_field.default_factory is not None

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SchedulerState)
        for key in self.CHECK_KEYS:
            assert hasattr(instance, key)

        # Validate that the instance attributes match the constructor args
        for field, expected_value in constructor_args.items():
            assert getattr(instance, field) == expected_value

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("node_id", "not_an_int"),
            ("start_time", "not_a_float"),
            ("end_time", [1, 2, 3]),
            ("remaining_fraction", "not_a_float"),
            ("created_requests", "not_an_int"),
            ("end_queuing_constraints", "not_a_dict"),
            ("scheduler_constraints", ["not", "a", "dict"]),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        # Start with valid base config
        base_kwargs = {
            "node_id": 0,
            "num_processes": 1,
            "start_time": 1000.0,
        }
        base_kwargs[field] = value
        with pytest.raises(ValidationError):
            SchedulerState(**base_kwargs)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data = instance.model_dump()
        assert isinstance(data, dict)
        assert all(key in data for key in self.CHECK_KEYS)

        # Test model_validate
        reconstructed = SchedulerState.model_validate(data)
        assert isinstance(reconstructed, SchedulerState)

        # Validate that all fields match between original and reconstructed instances
        for field in self.CHECK_KEYS:
            assert getattr(reconstructed, field) == getattr(instance, field)

        # Validate that the reconstructed instance matches original constructor args
        for field, expected_value in constructor_args.items():
            assert getattr(reconstructed, field) == expected_value


class TestSchedulerUpdateAction:
    CHECK_KEYS = [
        "request_queuing",
        "request_processing",
        "metadata",
        "progress",
    ]

    @pytest.fixture(
        params=[
            # Default configuration
            {},
            # All explicit default values
            {
                "request_queuing": "continue",
                "request_processing": "continue",
                "metadata": {},
                "progress": {},
            },
            # Stop queuing configuration
            {
                "request_queuing": "stop",
                "request_processing": "continue",
                "metadata": {"reason": "rate_limit_exceeded"},
            },
            # Stop local processing configuration
            {
                "request_queuing": "continue",
                "request_processing": "stop_local",
                "metadata": {"node_id": 1, "reason": "resource_exhausted"},
            },
            # Stop all processing configuration
            {
                "request_queuing": "stop",
                "request_processing": "stop_all",
                "metadata": {
                    "emergency_stop": True,
                    "reason": "critical_error",
                    "error_details": {"code": 500, "message": "Internal server error"},
                },
            },
            # Complex metadata configuration
            {
                "request_queuing": "continue",
                "request_processing": "continue",
                "metadata": {
                    "stats": {"processed": 100, "pending": 50},
                    "constraints": {"max_rps": 10, "max_concurrent": 20},
                    "config": {"batch_size": 32, "timeout": 30.0},
                },
            },
            # Progress with remaining_fraction only
            {
                "request_queuing": "continue",
                "request_processing": "continue",
                "progress": {"remaining_fraction": 0.75},
            },
            # Progress with remaining_requests only
            {
                "request_queuing": "continue",
                "request_processing": "continue",
                "progress": {"remaining_requests": 250.0},
            },
            # Progress with remaining_duration only
            {
                "request_queuing": "continue",
                "request_processing": "continue",
                "progress": {"remaining_duration": 120.5},
            },
            # Complete progress configuration
            {
                "request_queuing": "stop",
                "request_processing": "stop_all",
                "metadata": {"shutdown_reason": "completion"},
                "progress": {
                    "remaining_fraction": 0.0,
                    "remaining_requests": 0.0,
                    "remaining_duration": 0.0,
                },
            },
            # Partial progress configuration
            {
                "request_queuing": "continue",
                "request_processing": "continue",
                "metadata": {"checkpoint": "mid_benchmark"},
                "progress": {
                    "remaining_fraction": 0.45,
                    "remaining_duration": 180.0,
                },
            },
        ],
        ids=[
            "default_empty",
            "explicit_defaults",
            "stop_queuing",
            "stop_local_processing",
            "stop_all_processing",
            "complex_metadata",
            "progress_fraction_only",
            "progress_requests_only",
            "progress_duration_only",
            "complete_progress",
            "partial_progress",
        ],
    )
    def valid_instances(self, request):
        """Creates various valid configurations of SchedulerUpdateAction.

        Returns:
            tuple: (instance, constructor_args) where instance is the constructed
                   SchedulerUpdateAction and constructor_args are the kwargs used.
        """
        constructor_args = request.param
        instance = SchedulerUpdateAction(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SchedulerUpdateAction inheritance and type relationships."""
        assert issubclass(SchedulerUpdateAction, StandardBaseModel)
        assert hasattr(SchedulerUpdateAction, "model_dump")
        assert hasattr(SchedulerUpdateAction, "model_validate")

        # Check all expected fields are defined
        fields = SchedulerUpdateAction.model_fields
        for key in self.CHECK_KEYS:
            assert key in fields

        # Check field defaults
        assert fields["request_queuing"].default == "continue"
        assert fields["request_processing"].default == "continue"
        metadata_field = fields["metadata"]
        assert metadata_field.default_factory is not None
        progress_field = fields["progress"]
        assert progress_field.default_factory is not None

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SchedulerUpdateAction)
        for key in self.CHECK_KEYS:
            assert hasattr(instance, key)

        # Validate that the instance attributes match the constructor args or defaults
        for field in self.CHECK_KEYS:
            if field in constructor_args:
                assert getattr(instance, field) == constructor_args[field]
            elif field in ["request_queuing", "request_processing"]:
                assert getattr(instance, field) == "continue"
            elif field in ["metadata", "progress"]:
                assert getattr(instance, field) == {}

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("request_queuing", "invalid_action"),
            ("request_queuing", 123),
            ("request_processing", "invalid_action"),
            ("request_processing", ["stop"]),
            ("metadata", "not_a_dict"),
            ("metadata", [{"key": "value"}]),
            ("progress", "not_a_dict"),
            ("progress", [{"remaining_fraction": 0.5}]),
            ("progress", {"remaining_fraction": "not_a_float"}),
            ("progress", {"remaining_requests": "not_a_float"}),
            ("progress", {"remaining_duration": "not_a_float"}),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            SchedulerUpdateAction(**kwargs)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        # Test model_dump
        data = instance.model_dump()
        assert isinstance(data, dict)
        assert all(key in data for key in self.CHECK_KEYS)

        # Test model_validate
        reconstructed = SchedulerUpdateAction.model_validate(data)
        assert isinstance(reconstructed, SchedulerUpdateAction)

        # Validate that all fields match between original and reconstructed instances
        for field in self.CHECK_KEYS:
            assert getattr(reconstructed, field) == getattr(instance, field)

        # Validate that the reconstructed instance matches expected values
        for field in self.CHECK_KEYS:
            if field in constructor_args:
                assert getattr(reconstructed, field) == constructor_args[field]
            elif field in ["request_queuing", "request_processing"]:
                assert getattr(reconstructed, field) == "continue"
            elif field in ["metadata", "progress"]:
                assert getattr(reconstructed, field) == {}

    @pytest.mark.smoke
    def test_progress_field_behavior(self):
        """Test the progress field specific behavior and validation."""
        # Test empty progress (default)
        instance = SchedulerUpdateAction()
        assert instance.progress == {}
        assert isinstance(instance.progress, dict)

        # Test progress with all valid fields
        progress_data = {
            "remaining_fraction": 0.75,
            "remaining_requests": 100.0,
            "remaining_duration": 30.5,
        }
        instance = SchedulerUpdateAction(progress=progress_data)
        assert instance.progress == progress_data

        # Test progress with partial fields (TypedDict allows partial)
        partial_progress = {"remaining_fraction": 0.25}
        instance = SchedulerUpdateAction(progress=partial_progress)
        assert instance.progress == partial_progress

        # Test progress with zero values
        zero_progress = {
            "remaining_fraction": 0.0,
            "remaining_requests": 0.0,
            "remaining_duration": 0.0,
        }
        instance = SchedulerUpdateAction(progress=zero_progress)
        assert instance.progress == zero_progress

        # Test that progress field persists through marshalling
        data = instance.model_dump()
        assert "progress" in data
        assert data["progress"] == zero_progress

        reconstructed = SchedulerUpdateAction.model_validate(data)
        assert reconstructed.progress == zero_progress

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "progress_value",
        [
            {"remaining_fraction": 0.0},
            {"remaining_fraction": 1.0},
            {"remaining_requests": 0.0},
            {"remaining_requests": 1000.0},
            {"remaining_duration": 0.0},
            {"remaining_duration": 3600.0},
            {"remaining_fraction": 0.5, "remaining_requests": 50.0},
            {"remaining_requests": 25.0, "remaining_duration": 120.0},
            {"remaining_fraction": 0.33, "remaining_duration": 45.0},
        ],
    )
    def test_progress_valid_combinations(self, progress_value):
        """Test various valid combinations of progress field values."""
        instance = SchedulerUpdateAction(progress=progress_value)
        assert instance.progress == progress_value

        # Verify marshalling works correctly
        data = instance.model_dump()
        reconstructed = SchedulerUpdateAction.model_validate(data)
        assert reconstructed.progress == progress_value

    @pytest.mark.smoke
    def test_scheduler_update_action_progress_typeddict(self):
        """Test the SchedulerUpdateActionProgress TypedDict behavior."""
        # Test that SchedulerUpdateActionProgress is a proper TypedDict
        # Verify it's a TypedDict (has the special attributes)
        assert hasattr(SchedulerUpdateActionProgress, "__annotations__")
        assert hasattr(SchedulerUpdateActionProgress, "__total__")
        assert hasattr(SchedulerUpdateActionProgress, "__required_keys__")
        assert hasattr(SchedulerUpdateActionProgress, "__optional_keys__")

        # Check that all keys are optional (total=False)
        expected_keys = {
            "remaining_fraction",
            "remaining_requests",
            "remaining_duration",
        }
        actual_keys = set(SchedulerUpdateActionProgress.__annotations__.keys())
        assert actual_keys == expected_keys
        assert SchedulerUpdateActionProgress.__total__ is False
        assert SchedulerUpdateActionProgress.__required_keys__ == frozenset()
        assert SchedulerUpdateActionProgress.__optional_keys__ == expected_keys

        # Test that type annotations are correct
        annotations = SchedulerUpdateActionProgress.__annotations__
        assert "remaining_fraction" in annotations
        assert "remaining_requests" in annotations
        assert "remaining_duration" in annotations

        # Test creation of valid TypedDict instances
        valid_progress_1: SchedulerUpdateActionProgress = {}
        valid_progress_2: SchedulerUpdateActionProgress = {"remaining_fraction": 0.5}
        valid_progress_3: SchedulerUpdateActionProgress = {
            "remaining_fraction": 0.25,
            "remaining_requests": 100.0,
            "remaining_duration": 60.0,
        }

        # All should be valid dict instances
        assert isinstance(valid_progress_1, dict)
        assert isinstance(valid_progress_2, dict)
        assert isinstance(valid_progress_3, dict)
