import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import create_autospec, patch

import pytest
from click.testing import CliRunner

from guidellm import generate_benchmark_report
from guidellm.backend import Backend
from guidellm.core import TextGenerationBenchmarkReport, TextGenerationResult
from guidellm.executor import Executor, ExecutorResult, Profile, ProfileGenerationMode
from guidellm.main import generate_benchmark_report_cli
from guidellm.request import (
    EmulatedRequestGenerator,
    FileRequestGenerator,
    TransformersDatasetRequestGenerator,
)
from guidellm.scheduler import SchedulerResult
from guidellm.utils.progress import BenchmarkReportProgress


@pytest.fixture()
def mock_benchmark_report():
    with patch("guidellm.main.GuidanceReport") as mock_benchmark_report:

        def _mock_const(*args, **kwargs):
            instance = create_autospec(BenchmarkReportProgress, instance=True)
            instance.args = args
            instance.kwargs = kwargs
            instance.benchmarks = []
            instance.save_file = lambda output_path: None
            instance.print = lambda *args, **kwargs: None

            return instance

        mock_benchmark_report.side_effect = _mock_const
        yield mock_benchmark_report


@pytest.fixture()
def mock_benchmark_report_progress():
    with patch(
        "guidellm.main.BenchmarkReportProgress"
    ) as mock_benchmark_report_progress:

        def _mock_const(*args, **kwargs):
            instance = create_autospec(BenchmarkReportProgress, instance=True)
            instance.args = args
            instance.kwargs = kwargs

            return instance

        mock_benchmark_report_progress.side_effect = _mock_const
        yield mock_benchmark_report_progress


@pytest.fixture()
def mock_backend():
    with patch("guidellm.main.Backend.create") as mock_create:

        def _mock_create(*args, **kwargs):
            backend = create_autospec(Backend, instance=True)
            backend.args = args
            backend.kwargs = kwargs
            return backend

        mock_create.side_effect = _mock_create
        yield mock_create


@pytest.fixture()
def mock_request_generator_emulated():
    with patch(
        "guidellm.main.EmulatedRequestGenerator"
    ) as mock_request_generator_class:

        def _mock_const(*args, **kwargs):
            request_generator = create_autospec(EmulatedRequestGenerator, instance=True)
            request_generator.args = args
            request_generator.kwargs = kwargs
            return request_generator

        mock_request_generator_class.side_effect = _mock_const
        yield mock_request_generator_class


@pytest.fixture()
def mock_request_generator_file():
    with patch("guidellm.main.FileRequestGenerator") as mock_request_generator_class:

        def _mock_const(*args, **kwargs):
            request_generator = create_autospec(FileRequestGenerator, instance=True)
            request_generator.args = args
            request_generator.kwargs = kwargs
            return request_generator

        mock_request_generator_class.side_effect = _mock_const
        yield mock_request_generator_class


@pytest.fixture()
def mock_request_generator_transformers():
    with patch(
        "guidellm.main.TransformersDatasetRequestGenerator"
    ) as mock_request_generator_class:

        def _mock_const(*args, **kwargs):
            request_generator = create_autospec(
                TransformersDatasetRequestGenerator, instance=True
            )
            request_generator.args = args
            request_generator.kwargs = kwargs
            return request_generator

        mock_request_generator_class.side_effect = _mock_const
        yield mock_request_generator_class


@pytest.fixture()
def mock_executor():
    with patch("guidellm.main.Executor") as mock_executor_class:

        def _mock_const(*args, **kwargs):
            executor = create_autospec(Executor, instance=True)
            executor.args = args
            executor.kwargs = kwargs

            async def _mock_executor_run():
                generation_modes: List[ProfileGenerationMode]
                generation_rates: List[Optional[float]]
                completed_rates: List[float]

                if kwargs["mode"] == "sweep":
                    num_benchmarks = 12
                    generation_modes = [  # type: ignore
                        "synchronous",
                        "throughput",
                    ] + ["constant"] * 10
                    generation_rates = [None, None] + [ind + 1.0 for ind in range(10)]
                    completed_rates = [1.0, 10.0] + [ind + 1.0 for ind in range(10)]
                elif kwargs["rate"] is not None and isinstance(kwargs["rate"], list):
                    num_benchmarks = len(kwargs["rate"])
                    generation_modes = [kwargs["mode"]] * num_benchmarks
                    generation_rates = kwargs["rate"]
                    completed_rates = kwargs["rate"]
                else:
                    num_benchmarks = 1
                    generation_modes = [kwargs["mode"]]
                    generation_rates = [kwargs["rate"]]
                    completed_rates = [1.0]

                report = create_autospec(TextGenerationBenchmarkReport, instance=True)
                report.args = {
                    "backend": "backend",
                    "request_generator": "request_generator",
                    "mode": kwargs["mode"],
                    "rate": kwargs["rate"],
                    "max_number": kwargs["max_number"],
                    "max_duration": kwargs["max_duration"],
                }
                yield ExecutorResult(
                    completed=False,
                    count_total=num_benchmarks,
                    count_completed=0,
                    generation_modes=generation_modes,
                    report=report,
                    scheduler_result=None,
                    current_index=None,
                    current_profile=None,
                )
                for bench in range(num_benchmarks):
                    benchmark = create_autospec(
                        TextGenerationBenchmarkReport, instance=True
                    )
                    benchmark.start_time = 0
                    benchmark.end_time = 1
                    benchmark.completed_request_rate = completed_rates[bench]
                    profile = Profile(
                        load_gen_mode=generation_modes[bench],  # type: ignore
                        load_gen_rate=generation_rates[bench],
                    )
                    yield ExecutorResult(
                        completed=False,
                        count_total=num_benchmarks,
                        count_completed=bench,
                        generation_modes=generation_modes,
                        report=report,
                        scheduler_result=SchedulerResult(
                            completed=False,
                            count_total=10,
                            count_completed=0,
                            benchmark=benchmark,
                            current_result=None,
                        ),
                        current_index=bench,
                        current_profile=profile,
                    )
                    for ind in range(10):
                        yield ExecutorResult(
                            completed=False,
                            count_total=num_benchmarks,
                            count_completed=bench,
                            generation_modes=generation_modes,
                            report=report,
                            scheduler_result=SchedulerResult(
                                completed=False,
                                count_total=10,
                                count_completed=ind + 1,
                                benchmark=benchmark,
                                current_result=create_autospec(TextGenerationResult),
                            ),
                            current_index=bench,
                            current_profile=profile,
                        )
                    yield ExecutorResult(
                        completed=False,
                        count_total=num_benchmarks,
                        count_completed=bench + 1,
                        generation_modes=generation_modes,
                        report=report,
                    )
                yield ExecutorResult(
                    completed=True,
                    count_total=num_benchmarks,
                    count_completed=num_benchmarks,
                    generation_modes=generation_modes,
                    report=report,
                )

            executor.run.side_effect = _mock_executor_run
            return executor

        mock_executor_class.side_effect = _mock_const
        yield mock_executor_class


@pytest.mark.smoke()
def test_generate_benchmark_report_invoke_smoke(
    mock_backend, mock_request_generator_emulated, mock_executor
):
    report = generate_benchmark_report(
        target="http://localhost:8000/v1",
        backend="openai_server",
        model=None,
        data=None,
        data_type="emulated",
        tokenizer=None,
        rate_type="sweep",
        rate=None,
        max_seconds=10,
        max_requests=10,
        output_path="benchmark_report.json",
        cont_refresh_table=False,
    )
    assert report is not None


@pytest.mark.smoke()
def test_generate_benchmark_report_cli_smoke(
    mock_backend, mock_request_generator_emulated, mock_executor
):
    runner = CliRunner()
    result = runner.invoke(
        generate_benchmark_report_cli,
        [
            "--target",
            "http://localhost:8000/v1",
            "--backend",
            "openai_server",
            "--data-type",
            "emulated",
            "--data",
            "prompt_tokens=512",
            "--rate-type",
            "sweep",
            "--max-seconds",
            "10",
            "--max-requests",
            "10",
            "--output-path",
            "benchmark_report.json",
        ],
    )

    if result.stdout:
        print(result.stdout)  # noqa: T201

    assert result.exit_code == 0
    assert "Benchmarks" in result.output
    assert "Generating report..." in result.output
    assert "Benchmark Report 1" in result.output


@pytest.mark.smoke()
def test_generate_benchmark_report_emulated_with_dataset_requests(
    mock_backend, mock_request_generator_emulated, mock_executor
):
    with pytest.raises(ValueError, match="Cannot use 'dataset' for emulated data"):
        generate_benchmark_report(
            target="http://localhost:8000/v1",
            backend="openai_server",
            model=None,
            data_type="emulated",
            data=None,
            tokenizer=None,
            rate_type="sweep",
            rate=None,
            max_seconds=10,
            max_requests="dataset",
            output_path="benchmark_report.json",
            cont_refresh_table=False,
        )


@pytest.mark.smoke()
def test_generate_benchmark_report_cli_emulated_with_dataset_requests(
    mock_backend, mock_request_generator_emulated, mock_executor
):
    runner = CliRunner()
    with pytest.raises(ValueError, match="Cannot use 'dataset' for emulated data"):
        runner.invoke(
            generate_benchmark_report_cli,
            [
                "--target",
                "http://localhost:8000/v1",
                "--backend",
                "openai_server",
                "--data-type",
                "emulated",
                "--data",
                "prompt_tokens=512",
                "--rate-type",
                "sweep",
                "--max-seconds",
                "10",
                "--max-requests",
                "dataset",
                "--output-path",
                "benchmark_report.json",
            ],
            catch_exceptions=False,
        )


@pytest.mark.sanity()
@pytest.mark.parametrize(("rate_type", "rate"), [("constant", 1.0), ("sweep", 1.0)])
@pytest.mark.parametrize(
    ("file_extension", "file_content", "expected_results"),
    [
        ("txt", "Test prompt 1", 1),
        ("txt", "Test prompt 1\nTest prompt 2\nTest prompt 3\n", 3),
    ],
)
def test_generate_benchmark_report_openai_limited_by_file_dataset(
    mocker,
    mock_auto_tokenizer,
    mock_benchmark_report,
    mock_benchmark_report_progress,
    rate_type,
    rate,
    file_extension,
    file_content,
    expected_results,
):
    """
    Mock only a few functions to get the proper report result
    from the ``Backend.make_request``.

    Notes:
        All the results are collected in the `benchmark.errors``,
        since the most of the responses are mocked and can't be processed.
        But the ordering of the results is still the same for both collections.

        ``mock_benchmark_report`` and ``mock_benchmark_report_progress``
        are used for preventing working with IO bound tasks.
    """

    mocker.patch("guidellm.backend.openai.AsyncOpenAI")
    mocker.patch("openai.AsyncOpenAI")
    mocker.patch("guidellm.backend.openai.OpenAIBackend.test_connection")
    mocker.patch("guidellm.backend.openai.OpenAIBackend.available_models")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / f"example.{file_extension}"
        file_path.write_text(file_content)

        # Run the benchmark report generation
        report = generate_benchmark_report(
            target="http://localhost:8000/v1",
            backend="openai_server",
            model=None,
            data=str(file_path),
            data_type="file",
            tokenizer=None,
            rate_type=rate_type,
            rate=rate,
            max_seconds=None,
            max_requests="dataset",
            output_path="benchmark_report.json",
            cont_refresh_table=False,
        )

        assert report is not None
        assert len(report.benchmarks) == 1
        assert len(report.benchmarks[0].benchmarks[0].errors) == expected_results

        file_lines: List[str] = [line for line in file_content.split("\n") if line]
        output_prompts = [
            text_generation.request.prompt
            for text_generation in report.benchmarks[0].benchmarks[0].errors
        ]

        assert output_prompts == file_lines
