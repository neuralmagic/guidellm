from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import Field, validate_call
from transformers import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend import (
    Backend,
    BackendType,
)
from guidellm.benchmark.aggregator import (
    GenerativeRequestsAggregator,
    GenerativeRequestsStatsProgressAggregator,
    SchedulerStatsAggregator,
)
from guidellm.benchmark.benchmark import GenerativeBenchmark, GenerativeBenchmarksReport
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.output import (
    GenerativeBenchmarkerConsole,
)
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.progress import (
    BenchmarkerProgress,
    BenchmarkerProgressGroup,
    GenerativeConsoleBenchmarkerProgress,
)
from guidellm.benchmark.scenario import GenerativeTextScenario, Scenario
from guidellm.request import GenerativeRequestLoader
from guidellm.scheduler import StrategyType
from guidellm.scheduler.environment import NonDistributedEnvironment

__all__ = [
    "benchmark_generative_text",
    "benchmark_with_scenario",
    "reimport_benchmarks_report",
]


async def benchmark_with_scenario(scenario: Scenario, **kwargs):
    """
    Run a benchmark using a scenario and specify any extra arguments
    """

    if isinstance(scenario, GenerativeTextScenario):
        # Extract and handle special kwargs that need to be translated
        show_progress = kwargs.pop("show_progress", True)
        show_progress_scheduler_stats = kwargs.pop(
            "show_progress_scheduler_stats", False
        )

        # Convert show_progress to the progress parameter
        if show_progress:
            progress = [
                GenerativeConsoleBenchmarkerProgress(
                    enabled=True, display_scheduler_stats=show_progress_scheduler_stats
                )
            ]
        else:
            progress = None

        return await benchmark_generative_text(
            **vars(scenario), progress=progress, **kwargs
        )
    else:
        raise ValueError(f"Unsupported Scenario type {type(scenario)}")


@validate_call(config={"arbitrary_types_allowed": True})
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
    progress: Optional[list[BenchmarkerProgress]] = Field(
        default_factory=lambda: [GenerativeConsoleBenchmarkerProgress()]
    ),
    output_console: bool = True,
) -> tuple[GenerativeBenchmarksReport, Optional[Path]]:
    console = GenerativeBenchmarkerConsole()
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
    profile = Profile.create(
        rate_type=rate_type,
        rate=rate,
        random_seed=random_seed,
        constraints={
            key: val
            for key, val in {
                "max_number": max_requests,
                "max_duration": max_seconds,
                "max_errors": max_errors,
                "max_error_rate": max_error_rate,
                "max_global_error_rate": max_global_error_rate,
            }.items()
            if val is not None
        },
    )
    report = GenerativeBenchmarksReport()
    aggregators = {
        "scheduler_stats": SchedulerStatsAggregator(),
        "requests_progress": GenerativeRequestsStatsProgressAggregator(),
        "requests": GenerativeRequestsAggregator(
            warmup_requests=(
                int(max_requests * warmup_percent)
                if warmup_percent and max_requests
                else None
            ),
            warmup_duration=(
                max_seconds * warmup_percent if warmup_percent and max_seconds else None
            ),
            cooldown_requests=(
                int(max_requests * cooldown_percent)
                if cooldown_percent and max_requests
                else None
            ),
            cooldown_duration=(
                max_seconds * cooldown_percent
                if cooldown_percent and max_seconds
                else None
            ),
        ),
    }
    progress_group = BenchmarkerProgressGroup(
        instances=progress or [], enabled=progress is not None
    )

    async for (
        _aggregator_update,
        benchmark,
        _strategy,
        _scheduler_state,
    ) in progress_group(
        profile,
        Benchmarker().run(
            requests=request_loader,
            backend=backend,
            profile=profile,
            environment=NonDistributedEnvironment(),
            benchmark_aggregators=aggregators,
            benchmark_class=GenerativeBenchmark,
        ),
    ):
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
    console = GenerativeBenchmarkerConsole()
    report = GenerativeBenchmarksReport.load_file(file)
    console.benchmarks = report.benchmarks
    console.print_full_report()

    if output_path:
        console.print_line("\nSaving benchmarks report...")
        saved_path = report.save_file(output_path)
        console.print_line(f"Benchmarks report saved to {saved_path}")
