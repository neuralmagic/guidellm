import json

import click

from guidellm.backend import Backend
from guidellm.core import TextGenerationBenchmarkReport
from guidellm.executor import Executor
from guidellm.request import (
    EmulatedRequestGenerator,
    FileRequestGenerator,
    TransformersDatasetRequestGenerator,
)
from guidellm.request.base import RequestGenerator


@click.command()
@click.option(
    "--target",
    type=str,
    default="localhost:8000/completions",
    help="Target for benchmarking",
)
@click.option("--host", type=str, help="Host for benchmarking")
@click.option("--port", type=str, help="Port for benchmarking")
@click.option("--path", type=str, help="Path for benchmarking")
@click.option(
    "--backend", type=str, default="openai_server", help="Backend type for benchmarking"
)
@click.option("--model", type=str, default=None, help="Model to use for benchmarking")
@click.option("--task", type=str, default=None, help="Task to use for benchmarking")
@click.option("--data", type=str, help="Data file or alias for benchmarking")
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
    type=click.Choice(["sweep", "synchronous", "constant", "poisson"]),
    default="sweep",
    help="Type of rate generation for benchmarking",
)
@click.option(
    "--rate",
    type=float,
    default="1.0",
    help="Rate to use for constant and poisson rate types",
)
@click.option(
    "--num-seconds",
    type=int,
    default="120",
    help="Number of seconds to result each request rate at",
)
@click.option(
    "--num-requests",
    type=int,
    default=None,
    help="Number of requests to send for each rate",
)
def main(
    target,
    host,
    port,
    path,
    backend,
    model,
    task,
    data,
    data_type,
    tokenizer,
    rate_type,
    rate,
    num_seconds,
    num_requests,
):
    # Create backend
    Backend.create(
        backend_type=backend,
        target=target,
        host=host,
        port=port,
        path=path,
        model=model,
    )

    # Create request generator
    if not data and task:
        # TODO: Implement task-based request generation
        raise NotImplementedError(
            "Task-based request generation is not yet implemented"
        )

    if data_type == "emulated":
        request_generator: RequestGenerator = EmulatedRequestGenerator(config=data, tokenizer=tokenizer)
    elif data_type == "file":
        request_generator = FileRequestGenerator(file_path=data, tokenizer=tokenizer)
    elif data_type == "transformers":
        request_generator = TransformersDatasetRequestGenerator(
            dataset=data, tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # Create executor
    executor = Executor(
        request_generator=request_generator,
        backend=backend,
        profile_mode=rate_type,
        profile_args={"rate_type": rate_type, "rate": rate},
        max_requests=num_requests,
        max_duration=num_seconds,
    )
    report = executor.run()

    # Save or print results
    save_report(report, "benchmark_report.json")
    print_report(report)


def save_report(report: TextGenerationBenchmarkReport, filename: str):
    with open(filename, "w") as f:
        json.dump(report.to_dict(), f, indent=4)


def print_report(report: TextGenerationBenchmarkReport):
    for benchmark in report.benchmarks:
        print(f"Rate: {benchmark.request_rate}, Results: {benchmark.results}")


if __name__ == "__main__":
    main()
