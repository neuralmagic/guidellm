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
def test_text_generation_result_output_token():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    prompt = "Once upon a time"
    result.start(prompt)
    token = "the"
    result.output_token(token)

    assert result.last_time
    assert result.start_time
    assert result.output == f"{token} "
    assert result.last_time is not None and result.last_time > result.start_time


@pytest.mark.sanity
def test_text_generation_result_end():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    result.start("Once upon a time")
    result.end("The end")

    assert result.output == "The end"
    assert result.last_time
    assert result.start_time
    assert result.end_time is not None and result.end_time > result.start_time


@pytest.mark.regression
def test_text_generation_result_json():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    prompt = "Once upon a time"
    result.start(prompt)
    generated = "The end"
    result.end(generated)
    json_str = result.to_json()
    assert '"prompt":"Once upon a time"' in json_str
    assert '"output":"The end"' in json_str

    result_restored = TextGenerationResult.from_json(json_str)
    assert result.request == result_restored.request
    assert result_restored.prompt == prompt
    assert result_restored.output == generated

    json_str_restored = result_restored.to_json()
    assert json_str == json_str_restored


@pytest.mark.regression
def test_text_generation_result_yaml():
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    prompt = "Once upon a time"
    result.start(prompt)
    generated = "The end"
    result.end(generated)
    yaml_str = result.to_yaml()
    assert "prompt: Once upon a time" in yaml_str
    assert "output: The end" in yaml_str

    result_restored = TextGenerationResult.from_yaml(yaml_str)
    assert result.request == result_restored.request
    assert result_restored.prompt == prompt
    assert result_restored.output == generated

    yaml_str_restored = result_restored.to_yaml()
    assert yaml_str == yaml_str_restored


@pytest.mark.smoke
def test_text_generation_error_initialization():
    request = TextGenerationRequest(prompt="Generate a story")
    error = Exception("Test error")
    result = TextGenerationError(request=request, message=str(error))
    assert result.request == request
    assert str(result.message) == str(error)


@pytest.mark.regression
def test_text_generation_error_json():
    request = TextGenerationRequest(prompt="Generate a story")
    error = Exception("Test error")
    result = TextGenerationError(request=request, message=str(error))
    json_str = result.to_json()

    result_restored = TextGenerationError.from_json(json_str)

    assert result.message == "Test error"
    assert result.request == result_restored.request
    assert str(result_restored.message) == str(error)

    json_str_restored = result_restored.to_json()
    assert json_str == json_str_restored


@pytest.mark.regression
def test_text_generation_error_yaml():
    request = TextGenerationRequest(prompt="Generate a story")
    error = Exception("Test error")
    result = TextGenerationError(request=request, message=str(error))
    yaml_str = result.to_yaml()

    result_restored = TextGenerationError.from_yaml(yaml_str)

    assert result.message == "Test error"
    assert result.request == result_restored.request
    assert str(result_restored.message) == str(error)

    yaml_str_restored = result_restored.to_yaml()
    assert yaml_str == yaml_str_restored


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
    error = TextGenerationError(request=request, message=str(Exception("Test error")))
    benchmark.request_completed(error)
    assert benchmark.request_count == 0
    assert benchmark.error_count == 1


@pytest.mark.regression
def test_text_generation_benchmark_json():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    benchmark.request_completed(result)
    json_str = benchmark.to_json()
    assert '"mode":"test"' in json_str
    assert '"rate":1.0' in json_str

    benchmark_restored = TextGenerationBenchmark.from_json(json_str)
    assert benchmark.mode == benchmark_restored.mode
    assert benchmark.rate == benchmark_restored.rate
    assert benchmark.request_count == benchmark_restored.request_count
    assert benchmark.error_count == benchmark_restored.error_count

    json_str_restored = benchmark_restored.to_json()
    assert json_str == json_str_restored


@pytest.mark.regression
def test_text_generation_benchmark_yaml():
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    benchmark.request_started()
    request = TextGenerationRequest(prompt="Generate a story")
    result = TextGenerationResult(request=request)
    benchmark.request_completed(result)
    yaml_str = benchmark.to_yaml()
    assert "mode: test" in yaml_str
    assert "rate: 1.0" in yaml_str

    benchmark_restored = TextGenerationBenchmark.from_yaml(yaml_str)
    assert benchmark.mode == benchmark_restored.mode
    assert benchmark.rate == benchmark_restored.rate
    assert benchmark.request_count == benchmark_restored.request_count
    assert benchmark.error_count == benchmark_restored.error_count

    yaml_str_restored = benchmark_restored.to_yaml()
    assert yaml_str == yaml_str_restored


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


@pytest.mark.regression
def test_text_generation_benchmark_report_json():
    report = TextGenerationBenchmarkReport()
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    report.add_benchmark(benchmark)
    json_str = report.to_json()
    assert '"benchmarks":' in json_str
    assert '"args":[]' in json_str

    report_restored = TextGenerationBenchmarkReport.from_json(json_str)
    assert len(report.benchmarks) == len(report_restored.benchmarks)
    assert len(report.args) == len(report_restored.args)

    json_str_restored = report_restored.to_json()
    assert json_str == json_str_restored


@pytest.mark.regression
def test_text_generation_benchmark_report_yaml():
    report = TextGenerationBenchmarkReport()
    benchmark = TextGenerationBenchmark(mode="test", rate=1.0)
    report.add_benchmark(benchmark)
    yaml_str = report.to_yaml()
    assert "benchmarks:" in yaml_str
    assert "args: []" in yaml_str

    report_restored = TextGenerationBenchmarkReport.from_yaml(yaml_str)
    assert len(report.benchmarks) == len(report_restored.benchmarks)
    assert len(report.args) == len(report_restored.args)

    yaml_str_restored = report_restored.to_yaml()
    assert yaml_str == yaml_str_restored
