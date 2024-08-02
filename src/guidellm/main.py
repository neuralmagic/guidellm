import click
from loguru import logger

from guidellm.backend import Backend
from guidellm.core import GuidanceReport
from guidellm.executor import (
    RATE_TYPE_TO_LOAD_GEN_MODE_MAPPER,
    RATE_TYPE_TO_PROFILE_MODE_MAPPER,
    Executor,
)
from guidellm.logger import configure_logger
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
    default="http://localhost:8000",
    help="Target for benchmarking",
)
@click.option("--host", type=str, default=None, help="Host for benchmarking")
@click.option("--port", type=str, default=None, help="Port for benchmarking")
@click.option(
    "--backend",
    type=click.Choice(["test", "openai_server"]),
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
    type=click.Choice(["sweep", "synchronous", "constant", "poisson"]),
    default="synchronous",
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
    help="Path to save benchmark report to",
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
    _backend = Backend.create(
        backend_type=backend,
        target=target,
        host=host,
        port=port,
        model=model,
    )

    # Create request generator
    if not data and task:
        # TODO: Implement task-based request generation
        raise NotImplementedError(
            "Task-based request generation is not yet implemented"
        )

    if data_type == "emulated":
        request_generator: RequestGenerator = EmulatedRequestGenerator(
            config=data, tokenizer=tokenizer
        )
    elif data_type == "file":
        request_generator = FileRequestGenerator(file_path=data, tokenizer=tokenizer)
    elif data_type == "transformers":
        request_generator = TransformersDatasetRequestGenerator(
            dataset=data, tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    profile_mode = RATE_TYPE_TO_PROFILE_MODE_MAPPER.get(rate_type)
    load_gen_mode = RATE_TYPE_TO_LOAD_GEN_MODE_MAPPER.get(rate_type)

    if not profile_mode or not load_gen_mode:
        raise ValueError("Invalid rate type")

    # Create executor
    executor = Executor(
        request_generator=request_generator,
        backend=_backend,
        profile_mode=profile_mode,
        profile_args={"load_gen_mode": load_gen_mode, "rates": rate},
        max_requests=max_requests,
        max_duration=max_seconds,
    )

    logger.debug("Running the executor")
    report = executor.run()

    # Save or print results
    guidance_report = GuidanceReport()
    guidance_report.benchmarks.append(report)
    guidance_report.save_file(output_path)

    print("Guidance Report Complete:")
    print(guidance_report)


if __name__ == "__main__":
    # invoke logger setup on import with default values
    # enabling console logging with INFO and disabling file logging
    configure_logger()

    # entrypoint
    main()
