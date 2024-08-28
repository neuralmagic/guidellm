import asyncio
from typing import Literal, Optional, get_args

import click
from loguru import logger

from guidellm.backend import Backend, BackendEnginePublic
from guidellm.core import GuidanceReport, TextGenerationBenchmarkReport
from guidellm.executor import Executor, ProfileGenerationMode
from guidellm.request import (
    EmulatedRequestGenerator,
    FileRequestGenerator,
    TransformersDatasetRequestGenerator,
)
from guidellm.request.base import RequestGenerator
from guidellm.utils import BenchmarkReportProgress

__all__ = ["generate_benchmark_report"]


@click.command()
@click.option(
    "--target",
    type=str,
    help=(
        "The target path or url for the backend to evaluate. "
        "Ex: 'http://localhost:8000/v1'"
    ),
)
@click.option(
    "--backend",
    type=click.Choice(get_args(BackendEnginePublic)),
    default="openai_server",
    help=(
        "The backend to use for benchmarking. "
        "The default is OpenAI Server enabling compatiability with any server that "
        "follows the OpenAI spec including vLLM"
    ),
)
@click.option(
    "--model",
    type=str,
    default=None,
    help=(
        "The Model to use for benchmarking. If not provided, it will use "
        "the first available model provided the backend supports listing models."
    ),
)
@click.option(
    "--data",
    type=str,
    default=None,
    help=(
        "The data source to use for benchmarking. Depending on the data-type, "
        "it should be a path to a file, a dataset name, "
        "or a configuration for emulated data."
    ),
)
@click.option(
    "--data-type",
    type=click.Choice(["emulated", "file", "transformers"]),
    default="emulated",
    help="The type of data given for benchmarking",
)
@click.option(
    "--tokenizer",
    type=str,
    default=None,
    help=(
        "The tokenizer to use for calculating the number of prompt tokens. "
        "If not provided, will use a Llama 3.1 tokenizer."
    ),
)
@click.option(
    "--rate-type",
    type=click.Choice(get_args(ProfileGenerationMode)),
    default="sweep",
    help=(
        "The type of request rate to use for benchmarking. "
        "The default is sweep, which will run a synchronous and throughput "
        "rate-type and then interfill with constant rates between the two values"
    ),
)
@click.option(
    "--rate",
    type=float,
    default=None,
    help="The request rate to use for constant and poisson rate types",
    multiple=True,
)
@click.option(
    "--max-seconds",
    type=int,
    default=120,
    help=(
        "The maximum number of seconds for each benchmark run. "
        "Either max-seconds, max-requests, or both must be set. "
        "The default is 120 seconds. "
        "Note, this is the maximum time for each rate supplied, not the total time. "
        "This value should be large enough to allow for "
        "the server's performance to stabilize."
    ),
)
@click.option(
    "--max-requests",
    type=int,
    default=None,
    help=(
        "The maximum number of requests for each benchmark run. "
        "Either max-seconds, max-requests, or both must be set. "
        "Note, this is the maximum number of requests for each rate supplied, "
        "not the total number of requests. "
        "This value should be large enough to allow for "
        "the server's performance to stabilize."
    ),
)
@click.option(
    "--output-path",
    type=str,
    default="guidance_report.json",
    help=(
        "The output path to save the output report to. "
        "The default is guidance_report.json."
    ),
)
@click.option(
    "--disable-continuous-refresh",
    is_flag=True,
    default=False,
    help=(
        "Disable continual refreshing of the output table in the CLI "
        "until the user exits. "
    ),
)
def generate_benchmark_report_cli(
    target: str,
    backend: BackendEnginePublic,
    model: Optional[str],
    data: Optional[str],
    data_type: Literal["emulated", "file", "transformers"],
    tokenizer: Optional[str],
    rate_type: ProfileGenerationMode,
    rate: Optional[float],
    max_seconds: Optional[int],
    max_requests: Optional[int],
    output_path: str,
    disable_continuous_refresh: bool,
):
    """
    Generate a benchmark report for a specified backend and dataset.
    """
    generate_benchmark_report(
        target=target,
        backend=backend,
        model=model,
        data=data,
        data_type=data_type,
        tokenizer=tokenizer,
        rate_type=rate_type,
        rate=rate,
        max_seconds=max_seconds,
        max_requests=max_requests,
        output_path=output_path,
        cont_refresh_table=not disable_continuous_refresh,
    )


def generate_benchmark_report(
    target: str,
    backend: BackendEnginePublic,
    model: Optional[str],
    data: Optional[str],
    data_type: Literal["emulated", "file", "transformers"],
    tokenizer: Optional[str],
    rate_type: ProfileGenerationMode,
    rate: Optional[float],
    max_seconds: Optional[int],
    max_requests: Optional[int],
    output_path: str,
    cont_refresh_table: bool = True,
) -> GuidanceReport:
    """
    Generate a benchmark report for a specified backend and dataset.

    :param target: The target URL or path for the backend to evaluate.
    :param backend: The backend type to use for benchmarking.
    :param model: The model to benchmark;
        defaults to the first available if not specified.
    :param data: The data source for benchmarking,
        which may be a path, dataset name, or config.
    :param data_type: The type of data to use,
        such as 'emulated', 'file', or 'transformers'.
    :param tokenizer: The tokenizer to use for token counting,
        defaulting to Llama 3.1 if not provided.
    :param rate_type: The rate type for requests during benchmarking.
    :param rate: The specific request rate for constant and poisson rate types.
    :param max_seconds: Maximum duration for each benchmark run in seconds.
    :param max_requests: Maximum number of requests per benchmark run.
    :param output_path: Path to save the output report file.
    :param cont_refresh_table: Continually refresh the table in the CLI
        until the user exits.
    """
    logger.info(
        "Generating benchmark report with target: {}, backend: {}", target, backend
    )

    # Create backend
    backend_inst = Backend.create(
        backend_type=backend,
        target=target,
        model=model,
    )

    request_generator: RequestGenerator

    # Create request generator
    if data_type == "emulated":
        request_generator = EmulatedRequestGenerator(config=data, tokenizer=tokenizer)
    elif data_type == "file":
        request_generator = FileRequestGenerator(path=data, tokenizer=tokenizer)
    elif data_type == "transformers":
        request_generator = TransformersDatasetRequestGenerator(
            dataset=data, tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # Create executor
    executor = Executor(
        backend=backend_inst,
        request_generator=request_generator,
        mode=rate_type,
        rate=rate if rate_type in ("constant", "poisson") else None,
        max_number=max_requests,
        max_duration=max_seconds,
    )

    # Run executor
    logger.debug(
        "Running executor with args: {}",
        {
            "backend": backend,
            "request_generator": request_generator,
            "mode": rate_type,
            "rate": rate,
            "max_number": max_requests,
            "max_duration": max_seconds,
        },
    )
    report = asyncio.run(_run_executor_for_result(executor))

    # Save and print report
    guidance_report = GuidanceReport()
    guidance_report.benchmarks.append(report)
    guidance_report.save_file(output_path)
    guidance_report.print(output_path, continual_refresh=cont_refresh_table)

    return guidance_report


async def _run_executor_for_result(executor: Executor) -> TextGenerationBenchmarkReport:
    report = None
    progress = BenchmarkReportProgress()
    started = False

    async for result in executor.run():
        if not started:
            progress.start(result.generation_modes)  # type: ignore  # noqa: PGH003
            started = True

        if result.current_index is not None:
            description = f"{result.current_profile.load_gen_mode}"  # type: ignore  # noqa: PGH003
            if result.current_profile.load_gen_mode in ("constant", "poisson"):  # type: ignore  # noqa: PGH003
                description += f"@{result.current_profile.load_gen_rate:.2f} req/s"  # type: ignore  # noqa: PGH003

            progress.update_benchmark(
                index=result.current_index,
                description=description,
                completed=result.scheduler_result.completed,  # type: ignore  # noqa: PGH003
                completed_count=result.scheduler_result.count_completed,  # type: ignore  # noqa: PGH003
                completed_total=result.scheduler_result.count_total,  # type: ignore  # noqa: PGH003
                start_time=result.scheduler_result.benchmark.start_time,  # type: ignore  # noqa: PGH003
                req_per_sec=result.scheduler_result.benchmark.completed_request_rate,  # type: ignore  # noqa: PGH003
            )

        if result.completed:
            report = result.report
            break

    progress.finish()

    if not report:
        raise ValueError("No report generated by executor")

    return report


if __name__ == "__main__":
    generate_benchmark_report_cli()
