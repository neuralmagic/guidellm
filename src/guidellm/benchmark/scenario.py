import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, Union

import yaml
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from loguru import logger
from transformers.tokenization_utils_base import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend.backend import BackendType
from guidellm.benchmark.profile import ProfileType
from guidellm.objects.pydantic import StandardBaseModel
from guidellm.scheduler.strategy import StrategyType

__ALL__ = ["Scenario", "GenerativeTextScenario"]

T = TypeVar("T", bound="Scenario")


class Scenario(StandardBaseModel):
    target: str

    @classmethod
    def from_file(
        cls: type[T], filename: Union[str, Path], overrides: Optional[dict] = None
    ) -> T:
        try:
            with open(filename) as f:
                if str(filename).endswith(".yaml") or str(filename).endswith(".yml"):
                    data = yaml.safe_load(f)
                else:  # Assume everything else is json
                    data = json.load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error("Failed to parse scenario")
            raise e

        data.update(overrides)
        return cls.model_validate(data)


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
    rate: Optional[Union[float, list[float]]] = None
    max_seconds: Optional[float] = None
    max_requests: Optional[int] = None
    warmup_percent: Optional[float] = None
    cooldown_percent: Optional[float] = None
    output_sampling: Optional[int] = None
    random_seed: int = 42
