from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import PreTrainedTokenizer

from guidellm.backend import Backend, BackendType
from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.benchmark.benchmarker import GenerativeBenchmarker
from guidellm.benchmark.profile import ProfileType, create_profile
from guidellm.benchmark.progress import BenchmarkerProgressDisplay
from guidellm.dataset import load_dataset
from guidellm.request import RequestLoader
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
) -> List[GenerativeBenchmark]:
    backend = Backend.create(
        backend_type, target=target, model=model, **(backend_args or {})
    )
    backend.validate()

    if processor is None:
        processor = backend.model

    if isinstance(processor, (str, Path)):
        processor = PreTrainedTokenizer.from_pretrained(
            processor, **(processor_args or {})
        )

    dataset = load_dataset(data, data_args, processor)
    request_loader, requests_loader_description, processor = RequestLoader(
        dataset, processor, processor_args
    )
    profile = create_profile(rate_type=rate_type, rate=rate)

    benchmarker = GenerativeBenchmarker(
        backend=backend,
        request_loader=request_loader,
        request_loader_description=requests_loader_description,
        benchmark_save_extras=output_extras,
        processor=processor,
    )
    progress = BenchmarkerProgressDisplay() if show_progress else None
    benchmarks = []

    async for result in benchmarker.run(
        profile=profile,
        max_number_per_strategy=max_requests,
        max_duration_per_strategy=max_seconds,
        warmup_number_per_strategy=(
            round(max_requests * warmup_percent)
            if max_requests and warmup_percent
            else None
        ),
        warmup_duration_per_strategy=(
            max_seconds * warmup_percent if max_seconds and warmup_percent else None
        ),
        cooldown_number_per_strategy=(
            round(max_requests * cooldown_percent)
            if max_requests and cooldown_percent
            else None
        ),
        cooldown_duration_per_strategy=(
            max_seconds * cooldown_percent if max_seconds and cooldown_percent else None
        ),
    ):
        if progress:
            progress.update(result)

        if result.type_ == "benchmark_compiled":
            benchmarks.append(result.current_benchmark)

    return benchmarks
