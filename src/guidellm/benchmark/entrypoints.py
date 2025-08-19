from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

# TODO: Review Cursor generated code (start)
from loguru import logger

# TODO: Review Cursor generated code (end)
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
        # TODO: Review Cursor generated code (start)
        # Map scenario fields to function parameters
        # Use model_dump() to get values, but handle special cases for problematic fields
        scenario_vars = scenario.model_dump()
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Debug logging to understand the data field issue
        logger.debug(f"DEBUG: scenario.data type: {type(scenario.data)}")
        logger.debug(f"DEBUG: scenario.data value: {scenario.data}")
        logger.debug(
            f"DEBUG: scenario_vars['data'] type: {type(scenario_vars.get('data'))}"
        )
        logger.debug(f"DEBUG: scenario_vars['data'] value: {scenario_vars.get('data')}")
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Handle the data field specially if it's a ValidatorIterator
        # This happens when Pydantic converts string data to an iterable during validation
        if "data" in scenario_vars and "ValidatorIterator" in str(
            type(scenario_vars["data"])
        ):
            logger.debug("DEBUG: Detected ValidatorIterator for data field")
            # Try to get the original string value
            # For CLI usage, the data should be a string like "prompt_tokens=256,output_tokens=128"
            try:
                # Access the actual scenario field to get the real string value
                actual_data = getattr(scenario, "data", None)
                logger.debug(
                    f"DEBUG: actual_data from scenario.data: {actual_data}, type: {type(actual_data)}"
                )
                if isinstance(actual_data, str):
                    scenario_vars["data"] = actual_data
                    logger.debug(
                        f"DEBUG: Updated scenario_vars['data'] to string: {actual_data}"
                    )
                elif hasattr(actual_data, "__iter__") and not isinstance(
                    actual_data, (dict, str)
                ):
                    # If it's an iterable, try to extract the string representation
                    # For now, we'll just keep the original value and let the function handle it
                    logger.debug(
                        "DEBUG: data is iterable but not string/dict, keeping original"
                    )
            except Exception as e:
                logger.debug(f"DEBUG: Exception handling ValidatorIterator: {e}")
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        function_params = {}
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Direct mappings (same name)
        direct_mapping_fields = [
            "target",
            "data",
            "random_seed",
            "model",
            "processor",
            "processor_args",
            "data_args",
            "data_sampler",
            "max_seconds",
            "max_requests",
            "max_error_rate",
            "backend_args",
        ]
        for field in direct_mapping_fields:
            if field in scenario_vars:
                function_params[field] = scenario_vars[field]
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Handle rate specially - only include if not None
        if "rate" in scenario_vars and scenario_vars["rate"] is not None:
            function_params["rate"] = scenario_vars["rate"]
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Field name mappings (different names)
        field_mappings = {
            "backend_type": "backend",
            "rate_type": "profile",
            "warmup_percent": "warmup",
            "cooldown_percent": "cooldown",
            "output_sampling": "request_samples",
        }
        for scenario_field, function_param in field_mappings.items():
            if scenario_field in scenario_vars:
                function_params[function_param] = scenario_vars[scenario_field]
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Handle kwargs mappings (CLI parameters to function parameters)
        final_kwargs = {}
        kwargs_mappings = {
            "output_path": "save_path",
        }
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        for cli_param, function_param in kwargs_mappings.items():
            if cli_param in kwargs:
                final_kwargs[function_param] = kwargs[cli_param]
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Handle special kwargs that need transformation
        if "show_progress" in kwargs:
            # Direct mapping: show_progress=True means print_updates=True
            final_kwargs["print_updates"] = kwargs.get("show_progress", True)
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Filter out CLI-specific parameters that don't map to function parameters
        # These will be handled differently by the function's internal logic
        filtered_kwargs = [
            "show_progress_scheduler_stats",
            "output_console",
            "output_extras",
        ]
        for kwarg in filtered_kwargs:
            # These parameters don't directly map to function parameters, so we skip them
            pass
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Debug logging for function parameters
        logger.debug(
            f"DEBUG: Final function_params keys: {list(function_params.keys())}"
        )
        logger.debug(
            f"DEBUG: Final function_params['data']: {function_params.get('data')}, type: {type(function_params.get('data'))}"
        )
        logger.debug(f"DEBUG: Final final_kwargs: {final_kwargs}")
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        return await benchmark_generative_text(**function_params, **final_kwargs)
        # TODO: Review Cursor generated code (end)
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

    # TODO: Review Cursor generated code (start)
    # Fix ValidatorIterator issue: convert it back to string if needed
    if "ValidatorIterator" in str(type(data)):
        try:
            # Try to extract the original string from the ValidatorIterator
            # For CLI synthetic data like "prompt_tokens=256,output_tokens=128"
            if hasattr(data, "__iter__"):
                # Convert iterator to list and reconstruct the string
                data_list = list(data)
                if len(data_list) > 0 and all(
                    isinstance(item, str) for item in data_list
                ):
                    # If all items are strings (characters), join them back into the original string
                    data = "".join(data_list)
                elif len(data_list) == 1 and isinstance(data_list[0], str):
                    data = data_list[0]
        except Exception:
            pass
    # TODO: Review Cursor generated code (end)

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

    # TODO: Review Cursor generated code (start)
    # Try to get info or description, with fallback
    info_str = "GenerativeRequestLoader"
    if hasattr(request_loader, "info"):
        info_str = request_loader.info
    elif hasattr(request_loader, "description"):
        info_str = str(request_loader.description)
    elif hasattr(request_loader, "data"):
        info_str = f"data={request_loader.data}"
    # TODO: Review Cursor generated code (end)

    console.print(
        f"[{Colors.SUCCESS}]Request loader created:[/{Colors.SUCCESS}] "
        # TODO: Review Cursor generated code (start)
        f"with {unique_requests} unique requests, {info_str}"
        # TODO: Review Cursor generated code (end)
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
        # TODO: Review Cursor generated code (start)
        # Fix rate parameter if it's a list with single value
        rate_param = rate
        if isinstance(rate, list) and len(rate) == 1:
            rate_param = rate[0]
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Handle rate parameter for different profile types
        profile_kwargs = {
            "rate_type": profile,
            "random_seed": random_seed,
            "constraints": {**constraints},
        }
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # For synchronous profiles, rate must be None
        if profile == "synchronous":
            profile_kwargs["rate"] = None
        elif rate_param is not UNSET:
            profile_kwargs["rate"] = rate_param
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        profile = Profile.create(**profile_kwargs)
        # TODO: Review Cursor generated code (end)
    elif constraints:
        raise ValueError(
            "Constraints must be empty or unset when providing a Profile instance. "
            f"Provided constraints: {constraints} ; provided profile: {profile}"
        )
    # TODO: Review Cursor generated code (start)
    # Try to get profile info with fallback
    profile_info = ""
    if hasattr(profile, "info"):
        profile_info = profile.info
    elif hasattr(profile, "type_"):
        profile_info = f"type={profile.type_}"
    else:
        profile_info = str(profile)
    # TODO: Review Cursor generated code (end)

    console.print(
        f"[{Colors.SUCCESS}]Profile created:[/{Colors.SUCCESS}] "
        # TODO: Review Cursor generated code (start)
        f"{profile.__class__.__name__} {profile_info}"
        # TODO: Review Cursor generated code (end)
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

    # TODO: Review Cursor generated code (start)
    # Handle UNSET parameter for progress
    progress_instances = [] if progress is UNSET or progress is None else progress
    progress_enabled = progress is not UNSET and progress is not None
    # TODO: Review Cursor generated code (end)

    progress_group = BenchmarkerProgressGroup(
        # TODO: Review Cursor generated code (start)
        instances=progress_instances,
        enabled=progress_enabled,
        # TODO: Review Cursor generated code (end)
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
            # TODO: Review Cursor generated code (start)
        ]().run(
            # TODO: Review Cursor generated code (end)
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
        # TODO: Review Cursor generated code (start)
        outputs = {"console": {"save_path": save_path}}
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Infer output format from file extension if save_path is provided
        if save_path is not None and save_path is not UNSET:
            save_path_obj = (
                Path(save_path) if not isinstance(save_path, Path) else save_path
            )
            # TODO: Review Cursor generated code (end)

            # TODO: Review Cursor generated code (start)
            # If it's a directory, use default JSON
            if save_path_obj.is_dir():
                outputs["json"] = {"save_path": save_path}
            else:
                # Infer format from file extension
                extension = save_path_obj.suffix.lower()
                if extension == ".json":
                    # JSON output is handled by report.save_file(), don't add duplicate
                    pass
                elif extension == ".yaml" or extension == ".yml":
                    # YAML output is handled by report.save_file(), don't add duplicate
                    pass
                elif extension == ".csv":
                    outputs["csv"] = {"save_path": save_path}
                elif extension == ".html" or extension == ".htm":
                    outputs["html"] = {"save_path": save_path}
                else:
                    # Unknown extension, default to JSON via report.save_file()
                    pass
            # TODO: Review Cursor generated code (end)
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
