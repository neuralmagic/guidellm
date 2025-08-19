from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import BeforeValidator, Field, NonNegativeInt, PositiveFloat, PositiveInt
from transformers.tokenization_utils_base import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend.backend import BackendType
from guidellm.benchmark.profile import ProfileType
from guidellm.scheduler.strategy import StrategyType
from guidellm.utils import StandardBaseModel

__ALL__ = ["Scenario", "GenerativeTextScenario", "get_builtin_scenarios"]

SCENARIO_DIR = Path(__file__).parent / "scenarios/"


@cache
def get_builtin_scenarios() -> list[str]:
    """Returns list of builtin scenario names."""
    return [p.stem for p in SCENARIO_DIR.glob("*.json")]


def parse_float_list(value: str | float | list[float]) -> list[float]:
    """
    Parse a comma separated string to a list of float
    or convert single float list of one or pass float
    list through.
    """
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


T = TypeVar("T", bound="Scenario")


class Scenario(StandardBaseModel):
    """
    Parent Scenario class with common options for all benchmarking types.
    """

    target: str

    @classmethod
    def from_builtin(cls: type[T], name: str, overrides: dict | None = None) -> T:
        filename = SCENARIO_DIR / f"{name}.json"

        if not filename.is_file():
            raise ValueError(f"{name} is not a valid builtin scenario")

        return cls.from_file(filename, overrides)


class GenerativeTextScenario(Scenario):
    """
    Scenario class for generative text benchmarks.
    """

    class Config:
        # NOTE: This prevents errors due to unvalidatable
        # types like PreTrainedTokenizerBase
        arbitrary_types_allowed = True

    backend_type: BackendType = "openai_http"
    backend_args: dict[str, Any] | None = None
    model: str | None = None
    processor: str | Path | PreTrainedTokenizerBase | None = None
    processor_args: dict[str, Any] | None = None
    data: (
        str
        | Path
        | Iterable[str | dict[str, Any]]
        | Dataset
        | DatasetDict
        | IterableDataset
        | IterableDatasetDict
    )
    data_args: dict[str, Any] | None = None
    data_sampler: Literal["random"] | None = None
    rate_type: StrategyType | ProfileType
    rate: Annotated[list[PositiveFloat] | None, BeforeValidator(parse_float_list)] = (
        None
    )
    max_seconds: PositiveFloat | None = None
    max_requests: PositiveInt | None = None
    # TODO: Review Cursor generated code (start)
    max_error_rate: Annotated[float | None, Field(gt=0, lt=1)] = None
    # TODO: Review Cursor generated code (end)
    warmup_percent: Annotated[float | None, Field(gt=0, le=1)] = None
    cooldown_percent: Annotated[float | None, Field(gt=0, le=1)] = None
    output_sampling: NonNegativeInt | None = None
    random_seed: int = 42
