import pytest
from unittest.mock import MagicMock, patch
from typing import Iterator
from guidellm.backend import Backend, BackendTypes, GenerativeResponse
from guidellm.core.request import BenchmarkRequest
from guidellm.core.result import BenchmarkResult


@Backend.register_backend(BackendTypes.TEST)
class TestBackend(Backend):
    def __init__(self, api_key: str = "dummy_key", model: str = "test-model"):
        self.api_key = api_key
        self.model = model

    def make_request(self, request: BenchmarkRequest) -> Iterator[GenerativeResponse]:
        # Mock implementation for making a request to the backend
        yield GenerativeResponse(type="token_iter", add_token="Token1")
        yield GenerativeResponse(type="token_iter", add_token="Token2")
        yield GenerativeResponse(
            type="final",
            output="Token1Token2",
            prompt_tokens=["Test"],
            output_tokens=["Token1", "Token2"],
        )


@pytest.mark.unit
def test_register_backend():
    assert Backend._registry[BackendTypes.TEST] == TestBackend


@pytest.mark.unit
def test_create_backend():
    backend = Backend.create_backend(
        BackendTypes.TEST, api_key="dummy_key", model="test-model"
    )
    assert isinstance(backend, TestBackend)


@pytest.mark.unit
def test_backend_submit():
    class MockBackend(Backend):
        def make_request(
            self, request: BenchmarkRequest
        ) -> Iterator[GenerativeResponse]:
            yield GenerativeResponse(type="token_iter", add_token="Token1")
            yield GenerativeResponse(type="token_iter", add_token="Token2")
            yield GenerativeResponse(
                type="final",
                output="Token1Token2",
                prompt_tokens=["Test"],
                output_tokens=["Token1", "Token2"],
            )

    request = BenchmarkRequest(prompt="Test prompt")
    backend = MockBackend()
    result = backend.submit(request)

    assert result.prompt == "Test prompt"
    assert result.output == "Token1Token2"
    assert result.prompt_token_count == 1  # Mock prompt_tokens length
    assert result.output_token_count == 2  # Mock output_tokens length
    assert result.first_token_time is not None
    assert result.end_time is not None


@pytest.mark.unit
def test_backend_make_request_not_implemented():
    class MockBackend(Backend):
        pass

    request = BenchmarkRequest(prompt="Test prompt")
    backend = MockBackend()
    with pytest.raises(NotImplementedError):
        list(backend.make_request(request))


@pytest.mark.end_to_end
def test_backend_full_workflow():
    class MockBackend(Backend):
        def make_request(
            self, request: BenchmarkRequest
        ) -> Iterator[GenerativeResponse]:
            yield GenerativeResponse(type="token_iter", add_token="Token1")
            yield GenerativeResponse(type="token_iter", add_token="Token2")
            yield GenerativeResponse(
                type="final",
                output="Token1Token2",
                prompt_tokens=["Test"],
                output_tokens=["Token1", "Token2"],
            )

    request = BenchmarkRequest(prompt="Test prompt")
    backend = MockBackend()
    result = backend.submit(request)

    assert result.prompt == "Test prompt"
    assert result.output == "Token1Token2"
    assert result.prompt_token_count == 1  # Mock prompt_tokens length
    assert result.output_token_count == 2  # Mock output_tokens length
    assert result.first_token_time is not None
    assert result.end_time is not None
    assert result.start_time is not None
