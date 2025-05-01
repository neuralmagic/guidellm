from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, Self, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers.tokenization_utils_base import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend.backend import BackendType
from guidellm.benchmark.profile import ProfileType
from guidellm.objects.pydantic import StandardBaseModel
from guidellm.scheduler.strategy import StrategyType

__ALL__ = ["Scenario", "GenerativeTextScenario"]


class Scenario(StandardBaseModel):
    target: str

    def _update(self, **fields: Any) -> Self:
        for k, v in fields.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid field {k}")
            setattr(self, k, v)

        return self

    def update(self, **fields: Any) -> Self:
        return self._update(**{k: v for k, v in fields.items() if v is not None})


class GenerativeTextScenario(Scenario):
    backend_type: BackendType = "openai_http"
    backend_args: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    processor: Optional[Union[str, Path, PreTrainedTokenizerBase]] = None
    processor_args: Optional[dict[str, Any]] = None
    data: Union[
        str,
        Path,
        Iterable[Union[str, dict[str, Any]]],
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
    ]
    data_args: Optional[dict[str, Any]] = None
    data_sampler: Optional[Literal["random"]] = None
    rate_type: Union[StrategyType, ProfileType]
    rate: Optional[Union[int, float, list[Union[int, float]]]] = None
    max_seconds: Optional[float] = None
    max_requests: Optional[int] = None
    warmup_percent: Optional[float] = None
    cooldown_percent: Optional[float] = None
    show_progress: bool = True
    show_progress_scheduler_stats: bool = True
    output_console: bool = True
    output_path: Optional[Union[str, Path]] = None
    output_extras: Optional[dict[str, Any]] = None
    output_sampling: Optional[int] = None
    random_seed: int = 42
