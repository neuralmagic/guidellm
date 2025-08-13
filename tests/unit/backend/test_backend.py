"""
Unit tests for the Backend base class and registry functionality.

### WRITTEN BY AI ###
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from guidellm.backend.backend import Backend
from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
)
from guidellm.scheduler import ScheduledRequestInfo


class TestBackend:
    """Test cases for Backend base class."""

    @pytest.mark.smoke
    def test_backend_default_properties(self):
        """Test Backend default property implementations.

        ### WRITTEN BY AI ###
        """

        class TestBackend(Backend):
            def info(self) -> dict[str, Any]:
                return {"test": "info"}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self) -> str:
                return "test-model"

        backend = TestBackend("openai_http")
        assert backend.processes_limit is None
        assert backend.requests_limit is None
        assert backend.type_ == "openai_http"

    @pytest.mark.sanity
    def test_backend_initialization(self):
        """Test Backend initialization with type.

        ### WRITTEN BY AI ###
        """

        class TestBackend(Backend):
            def info(self) -> dict[str, Any]:
                return {"type": self.type_}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self) -> str:
                return "test-model"

        backend = TestBackend("openai_http")
        assert backend.type_ == "openai_http"
        assert backend.info() == {"type": "openai_http"}

    @pytest.mark.sanity
    def test_backend_registry_mixin(self):
        """Test that Backend inherits from RegistryMixin.

        ### WRITTEN BY AI ###
        """
        from guidellm.utils.registry import RegistryMixin

        assert issubclass(Backend, RegistryMixin)
        assert hasattr(Backend, "register")
        assert hasattr(Backend, "get_registered_object")
        assert hasattr(Backend, "create")

    @pytest.mark.sanity
    def test_backend_create_method(self):
        """Test Backend.create class method.

        ### WRITTEN BY AI ###
        """
        # Mock a registered backend
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        with patch.object(
            Backend, "get_registered_object", return_value=mock_backend_class
        ):
            result = Backend.create("openai_http", test_arg="value")

            Backend.get_registered_object.assert_called_once_with("openai_http")
            mock_backend_class.assert_called_once_with(test_arg="value")
            assert result == mock_backend_instance

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_backend_interface_compatibility(self):
        """Test that Backend is compatible with BackendInterface.

        ### WRITTEN BY AI ###
        """
        from guidellm.scheduler import BackendInterface as SchedulerBackendInterface

        assert issubclass(Backend, SchedulerBackendInterface)

        # Test that Backend uses the correct generic types
        class TestBackend(Backend):
            def info(self) -> dict[str, Any]:
                return {}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                # Verify types match the interface
                assert isinstance(request, GenerationRequest)
                assert isinstance(request_info, ScheduledRequestInfo)
                yield request, request_info

            async def default_model(self) -> str:
                return "test-model"

        backend = TestBackend("openai_http")

        # Create test request and info
        request = GenerationRequest(content="test")
        request_info = ScheduledRequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=GenerationRequestTimings(),
        )

        # Test resolve method
        async for response, info in backend.resolve(request, request_info):
            assert response == request
            assert info == request_info

    @pytest.mark.regression
    def test_backend_register_process(self):
        """Test that Backend docstring examples are valid.

        ### WRITTEN BY AI ###
        """

        # Test that the pattern shown in docstring works
        class MyBackend(Backend):
            def __init__(self, api_key: str):
                super().__init__("mock_backend")  # type: ignore [arg-type]
                self.api_key = api_key

            def info(self) -> dict[str, Any]:
                return {"api_key": "***"}

            async def process_startup(self):
                self.client = Mock()  # Simulate API client

            async def process_shutdown(self):
                self.client = None  # type: ignore[assignment]

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self) -> str:
                return "my-model"

        # Register the backend
        Backend.register("my_backend")(MyBackend)

        # Create instance
        backend = Backend.create("my_backend", api_key="secret")
        assert isinstance(backend, MyBackend)
        assert backend.api_key == "secret"
        assert backend.type_ == "mock_backend"


class TestBackendRegistry:
    """Test cases for Backend registry functionality."""

    @pytest.mark.smoke
    def test_openai_backend_registered(self):
        """Test that OpenAI HTTP backend is registered.

        ### WRITTEN BY AI ###
        """
        from guidellm.backend.openai import OpenAIHTTPBackend

        # OpenAI backend should be registered
        backend = Backend.create("openai_http", target="http://test")
        assert isinstance(backend, OpenAIHTTPBackend)
        assert backend.type_ == "openai_http"

    @pytest.mark.smoke
    def test_backend_create_invalid_type(self):
        """Test Backend.create with invalid type.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError):
            Backend.create("invalid_type")

    @pytest.mark.sanity
    def test_backend_registry_functionality(self):
        """Test that backend registry functions work.

        ### WRITTEN BY AI ###
        """
        from guidellm.backend.openai import OpenAIHTTPBackend

        # Test that we can get registered backends
        openai_class = Backend.get_registered_object("openai_http")
        assert openai_class == OpenAIHTTPBackend

        # Test creating with kwargs
        backend = Backend.create(
            "openai_http", target="http://localhost:8000", model="gpt-4"
        )
        assert backend.target == "http://localhost:8000"
        assert backend.model == "gpt-4"

    @pytest.mark.regression
    def test_backend_registration_decorator(self):
        """Test that backend registration decorator works.

        ### WRITTEN BY AI ###
        """

        # Create a test backend class
        @Backend.register("test_backend")
        class TestBackend(Backend):
            def __init__(self, test_param="default"):
                super().__init__("test_backend")  # type: ignore
                self._test_param = test_param

            def info(self):
                return {"test_param": self._test_param}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self):
                return "test-model"

        # Test that it's registered and can be created
        backend = Backend.create("test_backend", test_param="custom")
        assert isinstance(backend, TestBackend)
        assert backend.info() == {"test_param": "custom"}
