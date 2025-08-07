from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend import (
    Backend,
    BackendType,
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.benchmark.output import (
    GenerativeBenchmarksConsole,
    GenerativeBenchmarksReport,
)
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.progress import GenerativeTextBenchmarkerProgressDisplay
from guidellm.benchmark.scenario import GenerativeTextScenario, Scenario
from guidellm.request import GenerativeRequestLoader
from guidellm.scheduler import StrategyType
from guidellm.scheduler.environment import NonDistributedEnvironment
from guidellm.benchmark.aggregator import (
    SchedulerStatsAggregator,
    GenerativeRequestsAggregator,
    GenerativeRequestsStatsProgressAggregator,
)


__all__ = [
    "benchmark_with_scenario",
    "benchmark_generative_text",
    "reimport_benchmarks_report",
]


async def benchmark_with_scenario(scenario: Scenario, **kwargs):
    """
    Run a benchmark using a scenario and specify any extra arguments
    """

    if isinstance(scenario, GenerativeTextScenario):
        return await benchmark_generative_text(**vars(scenario), **kwargs)
    else:
        raise ValueError(f"Unsupported Scenario type {type(scenario)}")


async def benchmark_generative_text(
    target: str,
    backend_type: BackendType,
    backend_args: Optional[dict[str, Any]],
    model: Optional[str],
    processor: Optional[Optional[Union[str, Path, PreTrainedTokenizerBase]]],
    processor_args: Optional[dict[str, Any]],
    data: Union[
        str,
        Path,
        Iterable[Union[str, dict[str, Any]]],
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
    ],
    data_args: Optional[dict[str, Any]],
    data_sampler: Optional[Literal["random"]],
    rate_type: Union[StrategyType, ProfileType],
    rate: Optional[Union[float, list[float]]],
    max_seconds: Optional[float],
    max_requests: Optional[int],
    max_errors: Optional[int],
    max_error_rate: Optional[float],
    max_global_error_rate: Optional[float],
    warmup_percent: Optional[float],
    cooldown_percent: Optional[float],
    output_path: Optional[Union[str, Path]],
    output_extras: Optional[dict[str, Any]],
    output_sampling: Optional[int],
    random_seed: int,
    show_progress: bool = True,
    show_progress_scheduler_stats: bool = False,
    output_console: bool = True,
) -> tuple[GenerativeBenchmarksReport, Optional[Path]]:
    console = GenerativeBenchmarksConsole(enabled=show_progress)
    backend = Backend.create(
        backend_type, target=target, model=model, **(backend_args or {})
    )

    if processor is None:
        processor = await backend.default_model()

    console.print_line("Creating request loader...")
    request_loader = GenerativeRequestLoader(
        data=data,
        data_args=data_args,
        processor=processor,
        processor_args=processor_args,
        shuffle=data_sampler == "random",
        random_seed=random_seed,
    )
    unique_requests = request_loader.num_unique_items(raise_err=False)
    console.print_line(
        f"Created loader with {unique_requests} unique requests from {data}.\n\n"
        if unique_requests > 0
        else f"Created loader with unknown number unique requests from {data}.\n\n"
    )

    constraints = {}
    if max_requests is not None:
        constraints["max_requests"] = max_requests
    if max_seconds is not None:
        constraints["max_seconds"] = max_seconds
    if max_errors is not None:
        constraints["max_errors"] = max_errors
    if max_error_rate is not None:
        constraints["max_error_rate"] = max_error_rate
    if max_global_error_rate is not None:
        constraints["max_global_error_rate"] = max_global_error_rate
    profile = Profile.create(
        rate_type=rate_type,
        rate=rate,
        random_seed=random_seed,
        constraints=constraints,
    )
    progress = (
        GenerativeTextBenchmarkerProgressDisplay(
            display_scheduler_stats=show_progress_scheduler_stats
        )
        if show_progress
        else None
    )
    report = GenerativeBenchmarksReport()

    async for aggregator_update, benchmark, strategy, scheduler_state in Benchmarker[
        GenerativeBenchmark,
        GenerationRequest,
        GenerationRequestTimings,
        GenerationResponse,
    ].run(
        requests=request_loader,
        backend=backend,
        profile=profile,
        environment=NonDistributedEnvironment(),
        benchmark_aggregators={
            "scheduler_stats": SchedulerStatsAggregator(),
            "requests_progress": GenerativeRequestsStatsProgressAggregator(),
            "requests": GenerativeRequestsAggregator(
                warmup_requests=(
                    int(max_requests * warmup_percent) if warmup_percent else None
                ),
                warmup_duration=(
                    max_seconds * warmup_percent if warmup_percent else None
                ),
                cooldown_requests=(
                    int(max_requests * cooldown_percent) if cooldown_percent else None
                ),
                cooldown_duration=(
                    max_seconds * cooldown_percent if cooldown_percent else None
                ),
            ),
        },
        benchmark_class=GenerativeBenchmark,
    ):
        if progress:
            progress.update(aggregator_update, benchmark, strategy, scheduler_state)

        if benchmark:
            report.benchmarks.append(benchmark)

    if output_console:
        console.benchmarks = report.benchmarks
        console.print_full_report()

    if output_path:
        console.print_line("\nSaving benchmarks report...")
        saved_path = report.save_file(output_path)
        console.print_line(f"Benchmarks report saved to {saved_path}")
    else:
        saved_path = None

    console.print_line("\nBenchmarking complete.")

    return report, saved_path


def reimport_benchmarks_report(file: Path, output_path: Optional[Path]) -> None:
    """
    The command-line entry point for re-importing and displaying an
    existing benchmarks report. Can also specify
    Assumes the file provided exists.
    """
    console = GenerativeBenchmarksConsole(enabled=True)
    report = GenerativeBenchmarksReport.load_file(file)
    console.benchmarks = report.benchmarks
    console.print_full_report()

    if output_path:
        console.print_line("\nSaving benchmarks report...")
        saved_path = report.save_file(output_path)
        console.print_line(f"Benchmarks report saved to {saved_path}")
