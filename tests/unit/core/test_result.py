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
    result = TextGenerationResult(request=request)
    assert result.request == request
    assert result.prompt == ""
    assert result.output == ""


@pytest.mark.sanity
def test_text_generation_result_start():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    prompt = "Once upon a time"
    result.start(prompt)
    assert result.prompt == prompt
    assert result.start_time is not None


@pytest.mark.sanity
def test_text_generation_result_end():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    result.end("The end")
    assert result.output == "The end"
    assert result.end_time is not None


@pytest.mark.smoke
def test_text_generation_error_initialization():
    request = TextGenerationRequest(prompt="Generate a story")
    error = Exception("Test error")
    result = TextGenerationError(request=request, error=error)
    assert result.request == request
    assert result.error == str(error)


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
    assert len(benchmark.concurrencies) == 1


@pytest.mark.regression
def test_text_generation_benchmark_completed_with_result():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    benchmark.request_completed(result)
    assert benchmark.request_count == 1
    assert benchmark.error_count == 0


@pytest.mark.regression
def test_text_generation_benchmark_completed_with_error():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    request = TextGenerationRequest(prompt="Generate a story")
    error = TextGenerationError(request=request, error=Exception("Test error"))
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
