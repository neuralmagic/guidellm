from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizer

from guidellm.backend import Backend, BackendType
from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.benchmark.benchmarker import GenerativeBenchmarker
from guidellm.benchmark.profile import ProfileType, create_profile
from guidellm.benchmark.progress import BenchmarkerProgressDisplay
from guidellm.request import GenerativeRequestLoader
from guidellm.scheduler import StrategyType


async def benchmark_generative_text(
    target: str,
    backend_type: BackendType,
    backend_args: Optional[Dict[str, Any]],
    model: Optional[str],
    processor: Optional[Union[str, Path, PreTrainedTokenizer, Callable]],
    processor_args: Optional[Dict[str, Any]],
    data: Union[
        str,
        Path,
        Iterable[Union[str, Dict[str, Any]]],
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
    ],
    data_args: Optional[Dict[str, Any]],
    data_sampler: Optional[Literal["random"]],
    rate_type: Union[StrategyType, ProfileType],
    rate: Optional[Union[int, float, List[Union[int, float]]]],
    max_seconds: Optional[float],
    max_requests: Optional[int],
    warmup_percent: Optional[float],
    cooldown_percent: Optional[float],
    show_progress: bool,
    output_path: Optional[Union[str, Path]],
    output_type: Optional[str],
    output_extras: Optional[Dict[str, Any]],
    random_seed: int,
) -> List[GenerativeBenchmark]:
    backend = Backend.create(
        backend_type, target=target, model=model, **(backend_args or {})
    )
    await backend.validate()

    if processor is None:
        processor = backend.model

    request_loader = GenerativeRequestLoader(
        data=data,
        data_args=data_args,
        processor=processor,
        processor_args=processor_args,
        shuffle=data_sampler == "random",
        iter_type=(
            "finite"  # assume a finite dataset is our limit
            if max_requests is None and max_seconds is None
            else "infinite"  # default to infinite so we don't run out of data
        ),
        random_seed=random_seed,
    )
    profile = create_profile(rate_type=rate_type, rate=rate)

    benchmarker = GenerativeBenchmarker(
        backend=backend,
        request_loader=request_loader,
        request_loader_description=request_loader.description,
        benchmark_save_extras=output_extras,
        processor=processor,
    )
    progress = BenchmarkerProgressDisplay() if show_progress else None
    benchmarks = []

    async for result in benchmarker.run(
        profile=profile,
        max_number_per_strategy=max_requests,
        max_duration_per_strategy=max_seconds,
        warmup_percent_per_strategy=warmup_percent,
        cooldown_percent_per_strategy=cooldown_percent,
    ):
        if progress:
            progress.update(result)

        if result.type_ == "benchmark_compiled":
            benchmarks.append(result.current_benchmark)

    return benchmarks
