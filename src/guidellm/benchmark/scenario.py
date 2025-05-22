from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import BeforeValidator, Field, NonNegativeInt, PositiveFloat, PositiveInt
from transformers.tokenization_utils_base import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend.backend import BackendType
from guidellm.benchmark.profile import ProfileType
from guidellm.objects.pydantic import StandardBaseModel
from guidellm.scheduler.strategy import StrategyType

__ALL__ = ["Scenario", "GenerativeTextScenario"]


def parse_float_list(value: Union[str, float, list[float]]) -> list[float]:
    if isinstance(value, (int, float)):
        return [value]
    elif isinstance(value, list):
        return value

    values = value.split(",") if "," in value else [value]

    try:
        return [float(val) for val in values]
    except ValueError as err:
        raise ValueError(
            "must be a number or comma-separated list of numbers."
        ) from err


class Scenario(StandardBaseModel):
    target: str


class GenerativeTextScenario(Scenario):
    # FIXME: This solves an issue with Pydantic and class types
    class Config:
        arbitrary_types_allowed = True

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
    rate: Annotated[
        Optional[list[PositiveFloat]], BeforeValidator(parse_float_list)
    ] = None
    max_seconds: Optional[PositiveFloat] = None
    max_requests: Optional[PositiveInt] = None
    warmup_percent: Annotated[Optional[float], Field(gt=0, le=1)] = None
    cooldown_percent: Annotated[Optional[float], Field(gt=0, le=1)] = None
    output_sampling: Optional[NonNegativeInt] = None
    random_seed: int = 42
