import time

import pytest

from guidellm.core import (
    RequestConcurrencyMeasurement,
    TextGenerationBenchmark,
    TextGenerationBenchmarkReport,
    TextGenerationError,
    TextGenerationRequest,
    TextGenerationResult,
)


def create_sample_request():
    return TextGenerationRequest(prompt="Hello, world!")


def create_sample_result():
    start_time = time.time()

    return TextGenerationResult(
        request=create_sample_request(),
        prompt_token_count=4,
        output="Generated text",
        output_token_count=3,
        start_time=start_time,
        end_time=start_time + 1.5,
        first_token_time=start_time + 0.5,
        last_token_time=start_time + 1.4,
    )


@pytest.mark.smoke()
def test_text_generation_result_default_initialization():
    result = TextGenerationResult(request=create_sample_request())
    assert result.request.prompt == "Hello, world!"
    assert result.prompt_token_count is None
    assert result.output == ""
    assert result.output_token_count is None
    assert result.start_time is None
    assert result.end_time is None
    assert result.first_token_time is None
    assert result.last_token_time is None


@pytest.mark.smoke()
def test_text_generation_result_initialization():
    result = create_sample_result()
    assert result.request.prompt == "Hello, world!"
    assert result.prompt_token_count == 4
    assert result.output == "Generated text"
    assert result.output_token_count == 3
    assert result.start_time >= 0.0
    assert result.end_time == result.start_time + 1.5
    assert result.first_token_time == result.start_time + 0.5
    assert result.last_token_time == result.start_time + 1.4

    # computed fields
    assert result.request_latency == 1.5
    assert result.time_to_first_token == 0.5 * 1000
    assert result.inter_token_latency == pytest.approx((1.4 - 0.5) * 1000 / 2)
    assert result.output_tokens_per_second == pytest.approx(2 / (1.4 - 0.5))


@pytest.mark.smoke()
def test_text_generation_result_marshalling():
    result = create_sample_result()
    serialized = result.model_dump()
    deserialized = TextGenerationResult.model_validate(serialized)

    for key, value in vars(result).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke()
def test_text_generation_error_initialization():
    error = TextGenerationError(
        request=create_sample_request(), message="Error message"
    )
    assert error.request.prompt == "Hello, world!"
    assert error.message == "Error message"


@pytest.mark.smoke()
def test_text_generation_error_marshalling():
    error = TextGenerationError(
        request=create_sample_request(), message="Error message"
    )
    serialized = error.model_dump()
    deserialized = TextGenerationError.model_validate(serialized)

    for key, value in vars(error).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke()
def test_request_concurrency_measurement_initialization():
    start_time = time.time()
    measurement = RequestConcurrencyMeasurement(
        time=start_time,
        completed=8,
        errored=2,
        processing=3,
    )
    assert measurement.time == start_time
    assert measurement.completed == 8
    assert measurement.errored == 2
    assert measurement.processing == 3


@pytest.mark.smoke()
def test_request_concurrency_measurement_marshalling():
    start_time = time.time()
    measurement = RequestConcurrencyMeasurement(
        time=start_time,
        completed=8,
        errored=2,
        processing=3,
    )
    serialized = measurement.model_dump()
    deserialized = RequestConcurrencyMeasurement.model_validate(serialized)

    for key, value in vars(measurement).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke()
def test_text_generation_benchmark_default_initialization():
    benchmark = TextGenerationBenchmark(mode="asynchronous")
    assert benchmark.mode == "asynchronous"
    assert benchmark.rate is None
    assert benchmark.results == []
    assert benchmark.errors == []
    assert benchmark.concurrencies == []

    # computed
    assert benchmark.request_count == 0
    assert benchmark.error_count == 0
    assert benchmark.total_count == 0
    assert benchmark.start_time is None
    assert benchmark.end_time is None
    assert benchmark.duration == 0.0
    assert benchmark.completed_request_rate == 0.0
    assert benchmark.request_latency_distribution is not None
    assert benchmark.request_latency == 0.0
    assert benchmark.request_latency_percentiles == {}
    assert benchmark.ttft_distribution is not None
    assert benchmark.time_to_first_token == 0.0
    assert benchmark.time_to_first_token_percentiles == {}
    assert benchmark.itl_distribution is not None
    assert benchmark.inter_token_latency == 0.0
    assert benchmark.inter_token_latency_percentiles == {}
    assert benchmark.output_token_throughput == 0.0
    assert benchmark.prompt_token_distribution is not None
    assert benchmark.prompt_token == 0.0
    assert benchmark.prompt_token_percentiles == {}
    assert benchmark.output_token_distribution is not None
    assert benchmark.output_token == 0.0
    assert benchmark.output_token_percentiles == {}


@pytest.mark.smoke()
def test_text_generation_benchmark_initialization():
    benchmark = TextGenerationBenchmark(mode="asynchronous", rate=10)
    assert benchmark.mode == "asynchronous"
    assert benchmark.rate == 10

    for _ in range(5):
        benchmark.request_started()
        benchmark.request_completed(create_sample_result())
        time.sleep(1.5)

    for _ in range(2):
        benchmark.request_started()
        benchmark.request_completed(
            TextGenerationError(
                request=create_sample_request(), message="Error message"
            )
        )

    def _test_percentiles(percentiles, value=None):
        assert len(percentiles) == 7
        assert list(percentiles.keys()) == ["1", "5", "10", "50", "90", "95", "99"]

        if value is None:
            assert all(per >= 0.0 for per in percentiles.values())
        else:
            assert all(per == pytest.approx(value) for per in percentiles.values())

    assert len(benchmark.results) == 5
    assert len(benchmark.errors) == 2
    assert len(benchmark.concurrencies) == 14
    assert benchmark.request_count == 5
    assert benchmark.error_count == 2
    assert benchmark.total_count == 7
    assert benchmark.start_time == pytest.approx(time.time() - 1.5 * 5, abs=0.01)
    assert benchmark.end_time == pytest.approx(time.time(), abs=0.01)
    assert benchmark.duration == benchmark.end_time - benchmark.start_time  # type: ignore
    assert benchmark.completed_request_rate == pytest.approx(5 / benchmark.duration)
    assert benchmark.request_latency_distribution is not None
    assert benchmark.request_latency == pytest.approx(1.5)
    _test_percentiles(benchmark.request_latency_percentiles, 1.5)
    assert benchmark.ttft_distribution is not None
    assert benchmark.time_to_first_token == pytest.approx(500)
    _test_percentiles(benchmark.time_to_first_token_percentiles, 500)
    assert benchmark.itl_distribution is not None
    assert benchmark.inter_token_latency == pytest.approx(450)
    _test_percentiles(benchmark.inter_token_latency_percentiles, 450)
    assert benchmark.output_token_throughput == pytest.approx(3.0 / 1.5, abs=0.01)
    assert benchmark.prompt_token_distribution is not None
    assert benchmark.prompt_token == pytest.approx(4.0)
    _test_percentiles(benchmark.prompt_token_percentiles, 4.0)
    assert benchmark.output_token_distribution is not None
    assert benchmark.output_token == pytest.approx(3.0)
    _test_percentiles(benchmark.output_token_percentiles, 3.0)


@pytest.mark.smoke()
def test_text_generation_benchmark_marshalling():
    benchmark = TextGenerationBenchmark(mode="asynchronous", rate=10)
    for _ in range(5):
        benchmark.request_started()
        benchmark.request_completed(create_sample_result())

    for _ in range(2):
        benchmark.request_started()
        benchmark.request_completed(
            TextGenerationError(
                request=create_sample_request(), message="Error message"
            )
        )

    serialized = benchmark.model_dump()
    deserialized = TextGenerationBenchmark.model_validate(serialized)

    for key, value in vars(benchmark).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke()
def test_text_generation_benchmark_report_initialization():
    report = TextGenerationBenchmarkReport(
        benchmarks=[
            TextGenerationBenchmark(mode="asynchronous", rate=10),
            TextGenerationBenchmark(mode="asynchronous", rate=20),
        ],
        args={
            "backend_type": "http",
            "target": "http://example.com",
            "model": "test-model",
        },
    )
    assert len(report.benchmarks) == 2
    assert report.args == {
        "backend_type": "http",
        "target": "http://example.com",
        "model": "test-model",
    }


@pytest.mark.smoke()
def test_text_generation_benchmark_report_marshalling():
    report = TextGenerationBenchmarkReport(
        benchmarks=[
            TextGenerationBenchmark(mode="asynchronous", rate=10),
            TextGenerationBenchmark(mode="asynchronous", rate=20),
        ],
        args={
            "backend_type": "http",
            "target": "http://example.com",
            "model": "test-model",
        },
    )
    serialized = report.model_dump()
    deserialized = TextGenerationBenchmarkReport.model_validate(serialized)

    for key, value in vars(report).items():
        assert getattr(deserialized, key) == value
