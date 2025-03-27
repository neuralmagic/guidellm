import tempfile
from pathlib import Path

import pytest

from guidellm.core import (
    GuidanceReport,
    TextGenerationBenchmark,
    TextGenerationBenchmarkReport,
    TextGenerationRequest,
    TextGenerationResult,
)


@pytest.fixture()
def sample_benchmark_report() -> TextGenerationBenchmarkReport:
    sample_request = TextGenerationRequest(prompt="sample prompt")
    sample_result = TextGenerationResult(
        request=sample_request,
        prompt_token_count=2,
        output="sample output",
        output_token_count=2,
        start_time=None,
        end_time=None,
        first_token_time=None,
        last_token_time=None,
    )
    sample_benchmark = TextGenerationBenchmark(
        mode="asynchronous",
        rate=1.0,
        results=[sample_result],
        errors=[],
        concurrencies=[],
    )
    return TextGenerationBenchmarkReport(
        benchmarks=[sample_benchmark], args={"arg1": "value1"}
    )


def compare_guidance_reports(report1: GuidanceReport, report2: GuidanceReport) -> bool:
    return report1.benchmarks == report2.benchmarks


@pytest.mark.smoke()
def test_guidance_report_initialization():
    report = GuidanceReport()
    assert report.benchmarks == []


@pytest.mark.smoke()
def test_guidance_report_initialization_with_params(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    assert report.benchmarks == [sample_benchmark_report]


@pytest.mark.sanity()
def test_guidance_report_print(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    report.print()  # This will output to the console


@pytest.mark.sanity()
def test_guidance_report_json(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    json_str = report.to_json()
    loaded_report = GuidanceReport.from_json(json_str)
    assert compare_guidance_reports(report, loaded_report)


@pytest.mark.sanity()
def test_guidance_report_yaml(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    yaml_str = report.to_yaml()
    loaded_report = GuidanceReport.from_yaml(yaml_str)
    assert compare_guidance_reports(report, loaded_report)


@pytest.mark.sanity()
def test_guidance_report_save_load_file(sample_benchmark_report):
    report = GuidanceReport(benchmarks=[sample_benchmark_report])
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "report.yaml"
        report.save_file(file_path)
        loaded_report = GuidanceReport.load_file(file_path)
        assert compare_guidance_reports(report, loaded_report)


@pytest.mark.regression()
def test_empty_guidance_report():
    report = GuidanceReport()
    assert len(report.benchmarks) == 0
    report.print()  # Ensure it doesn't raise error with no benchmarks


@pytest.mark.regression()
def test_compare_guidance_reports(sample_benchmark_report):
    report1 = GuidanceReport(benchmarks=[sample_benchmark_report])
    report2 = GuidanceReport(benchmarks=[sample_benchmark_report])
    assert compare_guidance_reports(report1, report2)


@pytest.mark.regression()
def test_compare_guidance_reports_inequality(sample_benchmark_report):
    report1 = GuidanceReport(benchmarks=[sample_benchmark_report])
    report2 = GuidanceReport(benchmarks=[])
    assert not compare_guidance_reports(report1, report2)
