import pytest

from guidellm.core import (
    TextGenerationBenchmark,
    TextGenerationBenchmarkReport,
    TextGenerationError,
    TextGenerationRequest,
    TextGenerationResult,
)


@pytest.mark.smoke
def test_text_generation_result_initialization():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request)
    assert result.request == request
    assert result.prompt == ""
    assert result.output == ""


@pytest.mark.sanity
def test_text_generation_result_start():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request)
    prompt = "Once upon a time"
    result.start(prompt)
    assert result.prompt == prompt
    assert result.start_time is not None


@pytest.mark.regression
def test_text_generation_result_repr():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request)

    assert repr(result) == (
        f"TextGenerationResult("
        f"request_id={request.id}, "
        f"prompt='', "
        f"output='', "
        f"start_time=None, "
        f"end_time=None, "
        f"first_token_time=None, "
        f"decode_times=Distribution(mean=0.00, median=0.00, min=0.0, max=0.0, count=0))"
    )


@pytest.mark.sanity
def test_text_generation_result_end():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request)
    result.end()
    assert result.output == ""
    assert result.end_time is not None


@pytest.mark.smoke
def test_text_generation_error_initialization():
    request = TextGenerationRequest(prompt="Generate a story")
    error = Exception("Test error")
    result = TextGenerationError(request, error)
    assert result.request == request
    assert result.error == error


@pytest.mark.regression
def test_text_generation_error_repr():
    request = TextGenerationRequest(prompt="Generate a story")
    error = Exception("Test error")
    result = TextGenerationError(request, error)
    assert repr(result) == f"TextGenerationError(request={request}, error={error})"


@pytest.mark.smoke
def test_text_generation_benchmark_initialization():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    assert benchmark.mode == "test"
    assert benchmark.rate == 1.0
    assert benchmark.request_count == 0
    assert benchmark.error_count == 0


@pytest.mark.sanity
def test_text_generation_benchmark_started():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    assert len(benchmark._concurrencies) == 1


@pytest.mark.regression
def test_text_generation_benchmark_completed_with_result():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request)
    benchmark.request_completed(result)
    assert benchmark.request_count == 1
    assert benchmark.error_count == 0


@pytest.mark.regression
def test_text_generation_benchmark_completed_with_error():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    request = TextGenerationRequest(prompt="Generate a story")
    error = TextGenerationError(request, Exception("Test error"))
    benchmark.request_completed(error)
    assert benchmark.request_count == 0
    assert benchmark.error_count == 1


@pytest.mark.smoke
def test_text_generation_benchmark_report_initialization():
    report = TextGenerationBenchmarkReport()
    assert len(report.benchmarks) == 0
    assert len(report.args) == 0


@pytest.mark.sanity
def test_text_generation_benchmark_report_add_benchmark():
    report = TextGenerationBenchmarkReport()
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    report.add_benchmark(benchmark)
    assert len(report.benchmarks) == 1
