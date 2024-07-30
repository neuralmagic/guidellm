import os
import tempfile

import pytest

from guidellm.core import (
    Distribution,
    GuidanceReport,
    TextGenerationBenchmark,
    TextGenerationBenchmarkReport,
    TextGenerationError,
    TextGenerationRequest,
    TextGenerationResult,
)


@pytest.fixture
def sample_benchmark_report() -> TextGenerationBenchmarkReport:
    sample_request = TextGenerationRequest(prompt="sample prompt")
    sample_distribution = Distribution()
    sample_result = TextGenerationResult(
        request=sample_request,
        prompt="sample prompt",
        prompt_word_count=2,
        prompt_token_count=2,
        output="sample output",
        output_word_count=2,
        output_token_count=2,
        last_time=None,
        first_token_set=False,
        start_time=None,
        end_time=None,
        first_token_time=None,
        decode_times=sample_distribution,
    )
    sample_error = TextGenerationError(request=sample_request, message="sample error")
    sample_benchmark = TextGenerationBenchmark(
        mode="async",
        rate=1.0,
        results=[sample_result],
        errors=[sample_error],
        concurrencies=[],
    )
    return TextGenerationBenchmarkReport(
        benchmarks=[sample_benchmark], args=[{"arg1": "value1"}]
    )


def compare_guidance_reports(report1: GuidanceReport, report2: GuidanceReport) -> bool:
    return report1 == report2


@pytest.mark.smoke
def test_guidance_report_initialization():
    report = GuidanceReport()
    assert report.benchmarks == []


@pytest.mark.smoke
def test_guidance_report_initialization_with_params(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    assert report.benchmarks == [sample_benchmark_report]


@pytest.mark.smoke
def test_guidance_report_file(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "report.yaml")
        report.save_file(file_path)
        loaded_report = GuidanceReport.load_file(file_path)
        assert compare_guidance_reports(report, loaded_report)


@pytest.mark.regression
def test_guidance_report_json(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    json_str = report.to_json()
    loaded_report = GuidanceReport.from_json(json_str)
    assert compare_guidance_reports(report, loaded_report)


@pytest.mark.regression
def test_guidance_report_yaml(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    yaml_str = report.to_yaml()
    loaded_report = GuidanceReport.from_yaml(yaml_str)
    assert compare_guidance_reports(report, loaded_report)
