import asyncio
from typing import get_args

import click
from loguru import logger

from guidellm.backend import Backend
from guidellm.core import GuidanceReport, TextGenerationBenchmarkReport
from guidellm.executor import Executor, ProfileGenerationMode
from guidellm.logger import configure_logger
from guidellm.request import (
    EmulatedRequestGenerator,
    FileRequestGenerator,
    TransformersDatasetRequestGenerator,
)
from guidellm.request.base import RequestGenerator
from guidellm.utils import BenchmarkReportProgress


@click.command()
@click.option(
    "--target",
    type=str,
    default="http://localhost:8000",
    help="Target for benchmarking",
)
@click.option("--host", type=str, default=None, help="Host for benchmarking")
@click.option("--port", type=str, default=None, help="Port for benchmarking")
@click.option(
    "--backend",
    type=click.Choice(["openai_server"]),
    default="openai_server",
    help="Backend type for benchmarking",
)
@click.option("--model", type=str, default=None, help="Model to use for benchmarking")
@click.option("--task", type=str, default=None, help="Task to use for benchmarking")
@click.option(
    "--data", type=str, default=None, help="Data file or alias for benchmarking"
)
@click.option(
    "--data-type",
    type=click.Choice(["emulated", "file", "transformers"]),
    default="transformers",
    help="The type of data given for benchmarking",
)
@click.option(
    "--tokenizer", type=str, default=None, help="Tokenizer to use for benchmarking"
)
@click.option(
    "--rate-type",
    type=click.Choice(get_args(ProfileGenerationMode)),
    default="sweep",
    help="Type of rate generation for benchmarking",
)
@click.option(
    "--rate",
    type=float,
    default=None,
    help="Rate to use for constant and poisson rate types",
    multiple=True,
)
@click.option(
    "--max-seconds",
    type=int,
    default=120,
    help="Number of seconds to result each request rate at",
)
@click.option(
    "--max-requests",
    type=int,
    default=None,
    help="Number of requests to send for each rate",
)
@click.option(
    "--output-path",
    type=str,
    default="benchmark_report.json",
    help="Path to save report report to",
)
def main(
    target,
    host,
    port,
    backend,
    model,
    task,
    data,
    data_type,
    tokenizer,
    rate_type,
    rate,
    max_seconds,
    max_requests,
    output_path,
):
    # Create backend
    backend = Backend.create(
        backend_type=backend,
        target=target,
        host=host,
        port=port,
        model=model,
    )

    # Create request generator
    if not data and task:
        raise NotImplementedError(
            "Task-based request generation is not yet implemented"
        )

    if data_type == "emulated":
        request_generator: RequestGenerator = EmulatedRequestGenerator(
            config=data, tokenizer=tokenizer
        )
    elif data_type == "file":
        request_generator = FileRequestGenerator(path=data, tokenizer=tokenizer)
    elif data_type == "transformers":
        request_generator = TransformersDatasetRequestGenerator(
            dataset=data, tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    executor = Executor(
        backend=backend,
        request_generator=request_generator,
        mode=rate_type,
        rate=rate if rate_type in ("constant", "poisson") else None,
        max_number=max_requests,
        max_duration=max_seconds,
    )
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

    # Save or print results
    guidance_report = GuidanceReport()
    guidance_report.benchmarks.append(report)
    guidance_report.save_file(output_path)
    guidance_report.print(output_path, continual_refresh=True)


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
    # invoke logger setup on import with default values
    # enabling console logging with INFO and disabling file logging
    configure_logger()

    # entrypoint
    main()
