from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import validate_call
from rich.console import Console
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
from guidellm.benchmark.aggregator import (
    Aggregator,
    CompilableAggregator,
    GenerativeRequestsAggregator,
    GenerativeStatsProgressAggregator,
    SchedulerStatsAggregator,
    SerializableAggregator,
)
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.objects import GenerativeBenchmark, GenerativeBenchmarksReport
from guidellm.benchmark.output import (
    GenerativeBenchmarkerConsole,
    GenerativeBenchmarkerOutput,
)
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.progress import (
    BenchmarkerProgress,
    BenchmarkerProgressGroup,
)
from guidellm.benchmark.scenario import GenerativeTextScenario, Scenario
from guidellm.request import GenerativeRequestLoader
from guidellm.scheduler import (
    ConstraintInitializer,
    NonDistributedEnvironment,
    StrategyType,
)
from guidellm.utils import UNSET, Colors

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
        return await benchmark_generative_text(**vars(scenario), **kwargs)
    else:
        raise ValueError(f"Unsupported Scenario type {type(scenario)}")


@validate_call(config={"arbitrary_types_allowed": True})
async def benchmark_generative_text(
    target: str,
    data: (
        Iterable[str]
        | Iterable[dict[str, Any]]
        | Dataset
        | DatasetDict
        | IterableDataset
        | IterableDatasetDict
        | str
        | Path
    ),
    profile: StrategyType | ProfileType | Profile,
    rate: float | list[float] = UNSET,
    random_seed: int = 42,
    # Backend configuration
    backend: BackendType | Backend = "openai_http",
    backend_args: dict[str, Any] | None = None,
    model: str | None = None,
    # Data configuration
    processor: str | Path | PreTrainedTokenizerBase | None = None,
    processor_args: dict[str, Any] | None = None,
    data_args: dict[str, Any] | None = None,
    data_sampler: Literal["random"] | None = None,
    # Output configuration
    save_path: str | Path | None = UNSET,
    outputs: (
        dict[str, str | dict[str, Any] | GenerativeBenchmarkerOutput] | None
    ) = UNSET,
    # Updates configuration
    progress: list[BenchmarkerProgress] = UNSET,
    print_updates: bool = True,
    # Aggregators configuration
    aggregators: (
        dict[str, str | dict[str, Any] | Aggregator | CompilableAggregator]
    ) = UNSET,
    warmup: float | None = None,
    cooldown: float | None = None,
    request_samples: int | None = 20,
    # Constraints configuration
    max_seconds: int | float | None = None,
    max_requests: int | None = None,
    max_errors: int | None = None,
    max_error_rate: float | None = None,
    max_global_error_rate: float | None = None,
    **constraints: dict[str, ConstraintInitializer | Any],
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    console = Console(quiet=not print_updates)

    backend = (
        Backend.create(backend, target=target, model=model, **(backend_args or {}))
        if not isinstance(backend, Backend)
        else backend
    )
    console.print(
        f"[{Colors.SUCCESS}]Backend initialized:[/{Colors.SUCCESS}] "
        f"{backend.__class__.__name__}: {backend.info}"
    )

    if processor is None and model:
        processor = model
        console.print(
            f"[{Colors.INFO}]Processor autoset:[/{Colors.INFO}] "
            f"Using model '{model}' as processor"
        )
    elif processor is None:
        # create tmp backend to run on main process to get processor
        # future work: spawn separate process/API for processor interactions
        console.print(
            f"[{Colors.INFO}]Processor loading:[/{Colors.INFO}] "
            f"Retrieving default processor from backend {backend.__class__.__name__}"
        )
        tmp_backend = deepcopy(backend)
        processor = await tmp_backend.default_model()
        del tmp_backend
        console.print(
            f"[{Colors.SUCCESS}]Processor autoset:[/{Colors.SUCCESS}] "
            f"Using processor '{processor}' from backend"
        )

    console.print(
        f"[{Colors.INFO}]Request loader initializing:[/{Colors.INFO}] "
        f"data={data}, processor={processor}, sampler={data_sampler}, "
        f"seed={random_seed}"
    )
    request_loader = GenerativeRequestLoader(
        data=data,
        data_args=data_args,
        processor=processor,
        processor_args=processor_args,
        shuffle=data_sampler == "random",
        random_seed=random_seed,
    )
    unique_requests = request_loader.num_unique_items(raise_err=False)
    console.print(
        f"[{Colors.SUCCESS}]Request loader created:[/{Colors.SUCCESS}] "
        f"with {unique_requests} unique requests, {request_loader.info}"
    )

    for key, val in {
        "max_seconds": max_seconds,
        "max_requests": max_requests,
        "max_errors": max_errors,
        "max_error_rate": max_error_rate,
        "max_global_error_rate": max_global_error_rate,
    }.items():
        if val is not None:
            constraints[key] = val
    if not isinstance(profile, Profile):
        profile = Profile.create(
            rate_type=profile,
            rate=rate,
            random_seed=random_seed,
            constraints={**constraints},
        )
    elif constraints:
        raise ValueError(
            "Constraints must be empty or unset when providing a Profile instance. "
            f"Provided constraints: {constraints} ; provided profile: {profile}"
        )
    console.print(
        f"[{Colors.SUCCESS}]Profile created:[/{Colors.SUCCESS}] "
        f"{profile.__class__.__name__} {profile.info}"
    )

    aggregators = (
        {
            "scheduler_stats": SchedulerStatsAggregator(),
            "requests_progress": GenerativeStatsProgressAggregator(),
            "requests": GenerativeRequestsAggregator(
                request_samples=request_samples,
                warmup=warmup,
                cooldown=cooldown,
            ),
        }
        if aggregators == UNSET
        else SerializableAggregator.resolve(aggregators)
    )
    console.print(
        f"[{Colors.SUCCESS}]Aggregators created:[/{Colors.SUCCESS}] "
        f"{len(aggregators)} aggregators: {', '.join(aggregators.keys())}"
    )

    progress_group = BenchmarkerProgressGroup(
        instances=progress or [], enabled=progress is not None
    )
    report = GenerativeBenchmarksReport()
    console.print(f"[{Colors.INFO}]Starting benchmark run...[/{Colors.INFO}]\n\n\n")

    async for (
        _aggregator_update,
        benchmark,
        _strategy,
        _scheduler_state,
    ) in progress_group(
        profile,
        Benchmarker[
            GenerativeBenchmark,
            GenerationRequest,
            GenerationRequestTimings,
            GenerationResponse,
        ].run(
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

    finalized_outputs = {}
    if save_path is not None:
        save_path = report.save_file(save_path if save_path is not UNSET else None)
        finalized_outputs["report"] = save_path
        console.print(
            f"[{Colors.SUCCESS}]Report saved:[/{Colors.SUCCESS}] "
            f"Benchmarks report saved to {save_path}"
        )

    if outputs == UNSET:
        outputs = {
            "console": {"save_path": save_path},
            "csv": {"save_path": save_path},
            "html": {"save_path": save_path},
        }
    for key, output in GenerativeBenchmarkerOutput.resolve(outputs or {}).items():
        finalized_outputs[key] = await output.finalize(report)
        console.print(
            f"[{Colors.SUCCESS}]Output finalized:[/{Colors.SUCCESS}] "
            f"'{key}' ({output.__class__.__name__}) finalized with return value: "
            f"{finalized_outputs[key]}"
        )

    console.print(
        f"\n[{Colors.SUCCESS}]Benchmarking complete![/{Colors.SUCCESS}] "
        f"Generated {len(report.benchmarks)} benchmark(s) with "
        f"{len(finalized_outputs)} output(s)"
    )

    return report, finalized_outputs


def reimport_benchmarks_report(file: Path, output_path: Path | None) -> None:
    """
    The command-line entry point for re-importing and displaying an
    existing benchmarks report. Can also specify
    Assumes the file provided exists.
    """
    report = GenerativeBenchmarksReport.load_file(file)
    console_output = GenerativeBenchmarkerConsole()
    console_output.finalize(report)
    console = Console()

    if output_path:
        console.print(f"[{Colors.INFO}]Saving benchmarks report...[/{Colors.INFO}]")
        saved_path = report.save_file(output_path)
        console.print(
            f"[{Colors.SUCCESS}]Report saved:[/{Colors.SUCCESS}] "
            f"Benchmarks report saved to {saved_path}"
        )
