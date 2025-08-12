import inspect
import time
from abc import ABC
from typing import Generic
from unittest.mock import patch

import pytest

from guidellm.scheduler import (
    Environment,
    MaxNumberConstraint,
    NonDistributedEnvironment,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SynchronousStrategy,
)


class TestEnvironment:
    @pytest.mark.smoke
    def test_is_abstract_base_class(self):
        """Test that Environment is an abstract base class."""
        assert issubclass(Environment, ABC)
        assert inspect.isabstract(Environment)

    @pytest.mark.smoke
    def test_abstract_methods_defined(self):
        """Test that the required abstract methods are defined."""
        abstract_methods = Environment.__abstractmethods__
        expected_methods = {
            "sync_run_params",
            "sync_run_start",
            "update_run_iteration",
            "sync_run_error",
            "sync_run_end",
        }
        assert abstract_methods == expected_methods

    @pytest.mark.smoke
    def test_generic_type_parameters(self):
        """Test that Environment is generic with correct type parameters."""
        assert issubclass(Environment, Generic)
        # Environment should be Generic[RequestT, ResponseT]
        orig_bases = getattr(Environment, "__orig_bases__", ())
        assert len(orig_bases) > 0
        generic_base = next(
            (
                base
                for base in orig_bases
                if hasattr(base, "__origin__") and base.__origin__ is Generic
            ),
            None,
        )
        assert generic_base is not None
        type_args = getattr(generic_base, "__args__", ())
        assert RequestT in type_args
        assert ResponseT in type_args

    @pytest.mark.smoke
    def test_invalid_implementation(self):
        """Test that invalid implementations raise TypeError."""

        class InvalidImplementation(Environment):
            pass

        with pytest.raises(TypeError):
            InvalidImplementation()

    @pytest.mark.smoke
    def test_implementation_construction(self):
        """Test that concrete implementations can be constructed."""

        class TestEnvironment(Environment):
            async def sync_run_params(self, requests, strategy, constraints):
                return requests, strategy, constraints

            async def sync_run_start(self):
                return 0.0

            async def update_run_iteration(self, response, request, request_info):
                pass

            async def sync_run_error(self, err):
                pass

            async def sync_run_end(self):
                yield

        env = TestEnvironment()
        assert isinstance(env, Environment)

    @pytest.mark.smoke
    def test_method_signatures(self):
        """Test that method signatures match expected interface."""
        info_sig = inspect.signature(Environment.info)
        assert len(info_sig.parameters) == 1
        assert "self" in info_sig.parameters

        params_sig = inspect.signature(Environment.sync_run_params)
        assert len(params_sig.parameters) == 4
        param_names = list(params_sig.parameters.keys())
        assert param_names == ["self", "requests", "strategy", "constraints"]

        start_sig = inspect.signature(Environment.sync_run_start)
        assert len(start_sig.parameters) == 1
        assert "self" in start_sig.parameters

        update_sig = inspect.signature(Environment.update_run_iteration)
        assert len(update_sig.parameters) == 4
        param_names = list(update_sig.parameters.keys())
        assert param_names == ["self", "response", "request", "request_info"]

        error_sig = inspect.signature(Environment.sync_run_error)
        assert len(error_sig.parameters) == 2
        param_names = list(error_sig.parameters.keys())
        assert param_names == ["self", "err"]

        end_sig = inspect.signature(Environment.sync_run_end)
        assert len(end_sig.parameters) == 1
        assert "self" in end_sig.parameters


class TestNonDistributedEnvironment:
    @pytest.mark.smoke
    def test_initialization(self):
        """Test basic initialization of NonDistributedEnvironment."""
        env = NonDistributedEnvironment()
        assert env.run_err is None
        assert isinstance(env, Environment)

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test that initialization doesn't accept invalid arguments."""
        with pytest.raises(TypeError):
            NonDistributedEnvironment("invalid_arg")

    @pytest.mark.smoke
    def test_inheritance_and_typing(self):
        """Test inheritance and type relationships."""
        env = NonDistributedEnvironment()

        # Should inherit from Environment
        assert isinstance(env, Environment)
        assert issubclass(NonDistributedEnvironment, Environment)

        # Should implement all required methods
        required_methods = [
            "sync_run_params",
            "sync_run_start",
            "update_run_iteration",
            "sync_run_error",
            "sync_run_end",
        ]

        for method_name in required_methods:
            assert hasattr(env, method_name)
            assert callable(getattr(env, method_name))

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("requests", "strategy", "constraints", "error_to_inject"),
        [
            (
                ["request1", "request2"],
                SynchronousStrategy(),
                {"max_requests": MaxNumberConstraint(max_num=10)},
                None,
            ),
            (
                [],
                SynchronousStrategy(),
                {},
                None,
            ),
            (
                ["single_request"],
                SynchronousStrategy(),
                {"max_requests": MaxNumberConstraint(max_num=1)},
                RuntimeError("Test error"),
            ),
            (
                range(5),
                SynchronousStrategy(),
                {"max_requests": MaxNumberConstraint(max_num=5)},
                ValueError("Connection failed"),
            ),
        ],
        ids=[
            "normal_execution",
            "empty_requests",
            "with_error",
            "multiple_requests_with_error",
        ],
    )
    async def test_lifecycle(self, requests, strategy, constraints, error_to_inject):
        """Test the complete lifecycle of environment methods."""
        env = NonDistributedEnvironment()

        (
            returned_requests,
            returned_strategy,
            returned_constraints,
        ) = await env.sync_run_params(requests, strategy, constraints)
        assert returned_requests is requests
        assert returned_strategy is strategy
        assert returned_constraints is constraints

        with (
            patch("time.time", return_value=1000.0),
            patch("guidellm.scheduler.environment.settings") as mock_settings,
        ):
            mock_settings.scheduler_start_delay_non_distributed = 2.5
            start_time = await env.sync_run_start()
            assert start_time == 1002.5

        mock_response = "mock_response"
        mock_request = "mock_request"
        mock_request_info = ScheduledRequestInfo(
            request_id="test-123",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=time.time(),
        )

        await env.update_run_iteration(mock_response, mock_request, mock_request_info)
        await env.update_run_iteration(None, mock_request, mock_request_info)
        await env.update_run_iteration(mock_response, None, mock_request_info)

        if error_to_inject:
            await env.sync_run_error(error_to_inject)
            assert env.run_err is error_to_inject

        if error_to_inject:
            with pytest.raises(type(error_to_inject)) as exc_info:
                async for _ in env.sync_run_end():
                    pass
            assert str(exc_info.value) == str(error_to_inject)
        else:
            results = []
            async for result in env.sync_run_end():
                results.append(result)
            assert results == []

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_sync_run_start_uses_config(self):
        """Test that sync_run_start uses configuration value."""
        env = NonDistributedEnvironment()

        with (
            patch("time.time", return_value=500.0),
            patch("guidellm.scheduler.environment.settings") as mock_settings,
        ):
            # Test different delay values
            mock_settings.scheduler_start_delay_non_distributed = 0.0
            start_time = await env.sync_run_start()
            assert start_time == 500.0

            mock_settings.scheduler_start_delay_non_distributed = 1.5
            start_time = await env.sync_run_start()
            assert start_time == 501.5

            mock_settings.scheduler_start_delay_non_distributed = 10.0
            start_time = await env.sync_run_start()
            assert start_time == 510.0
