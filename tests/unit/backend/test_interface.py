"""
Unit tests for the BackendInterface abstract class.

### WRITTEN BY AI ###
"""

from typing import Any, Optional

import pytest

from guidellm.backend.interface import BackendInterface


class TestBackendInterface:
    """Test cases for BackendInterface abstract class."""

    @pytest.mark.sanity
    def test_backend_interface_properties_are_abstract(self):
        """Test that required properties are abstract.

        ### WRITTEN BY AI ###
        """

        # Create a partial implementation to verify abstract nature
        class PartialBackend(BackendInterface):
            # Missing required properties/methods
            pass

        with pytest.raises(TypeError):
            PartialBackend()

    @pytest.mark.sanity
    def test_minimal_concrete_implementation(self):
        """Test that a minimal concrete implementation can be created.

        ### WRITTEN BY AI ###
        """

        class MinimalBackend(BackendInterface):
            @property
            def processes_limit(self) -> Optional[int]:
                return None

            @property
            def requests_limit(self) -> Optional[int]:
                return None

            def info(self) -> dict[str, Any]:
                return {}

            async def process_startup(self) -> None:
                pass

            async def validate(self) -> None:
                pass

            async def process_shutdown(self) -> None:
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

        # Should be able to instantiate
        backend = MinimalBackend()
        assert backend is not None
        assert isinstance(backend, BackendInterface)

    @pytest.mark.regression
    def test_backend_interface_method_signatures(self):
        """Test that BackendInterface methods have correct signatures.

        ### WRITTEN BY AI ###
        """
        import inspect

        # Check resolve method signature
        resolve_sig = inspect.signature(BackendInterface.resolve)
        params = list(resolve_sig.parameters.keys())

        expected_params = ["self", "request", "request_info", "history"]
        assert params == expected_params

        # Check that history has default value
        history_param = resolve_sig.parameters["history"]
        assert history_param.default is None
