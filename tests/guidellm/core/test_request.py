import pytest
from guidellm import BenchmarkRequest


@pytest.mark.unit
def test_benchmark_request_initialization():
    prompt = "Generate a summary of the following text."
    params = {"max_tokens": 50}
    request = BenchmarkRequest(prompt, params)
    assert request.prompt == prompt
    assert request.params == params


@pytest.mark.unit
def test_benchmark_request_default_params():
    prompt = "Generate a summary of the following text."
    request = BenchmarkRequest(prompt)
    assert request.prompt == prompt
    assert request.params == {}


@pytest.mark.unit
def test_benchmark_request_str():
    prompt = "Generate a summary of the following text."
    params = {"max_tokens": 50}
    request = BenchmarkRequest(prompt, params)
    expected_str = f"BenchmarkRequest(prompt={prompt}, params={params})"
    assert str(request) == expected_str


@pytest.mark.unit
def test_benchmark_request_repr():
    prompt = "Generate a summary of the following text."
    params = {"max_tokens": 50}
    request = BenchmarkRequest(prompt, params)
    expected_repr = f"BenchmarkRequest(prompt={prompt}, params={params})"
    assert repr(request) == expected_repr


@pytest.mark.end_to_end
def test_benchmark_request_full_workflow():
    prompt = "Generate a summary of the following text."
    params = {"max_tokens": 50}
    request = BenchmarkRequest(prompt, params)

    # Check initialization
    assert request.prompt == prompt
    assert request.params == params

    # Check string representation
    expected_str = f"BenchmarkRequest(prompt={prompt}, params={params})"
    assert str(request) == expected_str

    # Check unambiguous representation
    expected_repr = f"BenchmarkRequest(prompt={prompt}, params={params})"
    assert repr(request) == expected_repr
