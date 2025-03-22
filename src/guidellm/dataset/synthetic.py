import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import yaml
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.config import settings
from guidellm.dataset.creator import ColumnInputTypes, DatasetCreator
from guidellm.objects import Serializable

__all__ = ["SyntheticDatasetCreator"]


class SyntheticDatasetConfig(Serializable):
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for prompts."
    )
    prompt_tokens_variance: Optional[int] = Field(
        description="The variance of the number of text tokens generated for prompts.",
        default=None,
    )
    prompt_tokens_min: Optional[int] = Field(
        description="The minimum number of text tokens generated for prompts.",
        default=None,
    )
    prompt_tokens_max: Optional[int] = Field(
        description="The maximum number of text tokens generated for prompts.",
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens generated for outputs.",
    )
    output_tokens_variance: Optional[int] = Field(
        description="The variance of the number of text tokens generated for outputs.",
        default=None,
    )
    output_tokens_min: Optional[int] = Field(
        description="The minimum number of text tokens generated for outputs.",
        default=None,
    )
    output_tokens_max: Optional[int] = Field(
        description="The maximum number of text tokens generated for outputs.",
        default=None,
    )
    samples: int = Field(
        description="The number of samples to generate for the dataset.",
        default=10000,
    )
    seed: int = Field(
        description="The seed to use for random number generation.",
        default=42,
    )

    @staticmethod
    def parse_str(data: Union[str, Path]) -> "SyntheticDatasetConfig":
        if (
            isinstance(data, Path)
            or data.strip().endswith(".config")
            or data.strip().endswith(".yaml")
        ):
            return SyntheticDatasetConfig.parse_config_file(data)

        if data.strip().startswith("{"):
            return SyntheticDatasetConfig.parse_json(data)

        if data.count("=") > 1:
            return SyntheticDatasetConfig.parse_key_value_pairs(data)

        raise ValueError(
            f"Unsupported data format. Expected JSON or key-value pairs, got {data}"
        )

    @staticmethod
    def parse_json(data: str) -> "SyntheticDatasetConfig":
        config_dict = json.loads(data.strip())

        return SyntheticDatasetConfig(**config_dict)

    @staticmethod
    def parse_key_value_pairs(data: str) -> "SyntheticDatasetConfig":
        config_dict = {}
        items = data.strip().split(",")
        for item in items:
            key, value = item.split("=")
            config_dict[key.strip()] = (
                int(value.strip()) if value.strip().isnumeric() else value.strip()
            )

        return SyntheticDatasetConfig(**config_dict)

    @staticmethod
    def parse_config_file(data: Union[str, Path]) -> "SyntheticDatasetConfig":
        with Path(data).open("r") as file:
            config_dict = yaml.safe_load(file)

        return SyntheticDatasetConfig(**config_dict)


class IntegerRangeSampler:
    def __init__(
        self,
        average: int,
        variance: Optional[int],
        min_value: Optional[int],
        max_value: Optional[int],
        seed: int,
    ):
        self.average = average
        self.variance = variance
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        calc_min = self.min_value
        if not calc_min:
            calc_min = max(
                0, self.average - 5 * self.variance if self.variance else self.average
            )
        calc_max = self.max_value
        if not calc_max:
            calc_max = (
                self.average + 5 * self.variance if self.variance else self.average
            )

        while True:
            if calc_min == calc_max:
                yield calc_min
            elif not self.variance:
                yield self.rng.randint(calc_min, calc_max + 1)
            else:
                rand = self.rng.gauss(self.average, self.variance)
                yield round(max(calc_min, min(calc_max, rand)))


class EndlessTextCreator:
    """
    A list subclass that allows for endless data generation.
    """

    def __init__(
        self,
        data: Union[str, Path],
        filter_start: Optional[Union[str, int]] = None,
        filter_end: Optional[Union[str, int]] = None,
    ):
        self.data = data
        text = load_text(data)
        text = filter_text(data, filter_start, filter_end)
        self.words = split_text(text)

    def create_text(self, start: int, length: int) -> str:
        """
        Create a text snippet from the specified range.

        :param start: Start index.
        :type start: int
        :param length: Length of the snippet.
        :type length: int
        :return: Text snippet.
        :rtype: str
        """
        start = start % len(self)
        text = ""

        for counter in range(length):
            index = (start + counter) % len(self.words)
            if counter > 0:
                text += " "
            text += self.words[index]

        return text


class SyntheticTextItemsGenerator(Iterable[Dict[str, Union[str, int]]]):
    def __init__(
        self, config: SyntheticDatasetConfig, processor: PreTrainedTokenizerBase
    ):
        self.config = config
        self.processor = processor
        self.tokens = []
        self.text_creator = EndlessTextCreator(
            data=settings.emulated_data.source,
            filter_start=settings.emulated_data.filter_start,
            filter_end=settings.emulated_data.filter_end,
        )

    def __iter__(self) -> Iterator[Tuple[str, int, int]]:
        prompt_tokens_sampler = IntegerRangeSampler(
            average=self.config.prompt_tokens,
            variance=self.config.prompt_tokens_variance,
            min_value=self.config.prompt_tokens_min,
            max_value=self.config.prompt_tokens_max,
            seed=self.config.seed,
        )
        output_tokens_sampler = IntegerRangeSampler(
            average=self.config.output_tokens,
            variance=self.config.output_tokens_variance,
            min_value=self.config.output_tokens_min,
            max_value=self.config.output_tokens_max,
            seed=self.config.seed,
        )
        start_index_sampler = random.Random(self.config.seed).randint(
            0, len(self.text_creator.words)
        )

        for _, prompt_tokens, output_tokens, start_index in zip(
            range(self.config.samples),
            prompt_tokens_sampler,
            output_tokens_sampler,
            start_index_sampler,
        ):
            yield {
                "prompt": self._create_prompt(prompt_tokens, start_index),
                "prompt_tokens_count": prompt_tokens,
                "output_tokens_count": output_tokens,
            }

    def _create_prompt(self, prompt_tokens: int, start_index: int) -> str:
        left = start_index
        right = start_index + 5 * prompt_tokens

        while left < right:
            mid = (left + right) // 2
            test_prompt = self.text_creator.create_text(start_index, mid)
            test_tokens = len(self.processor.tokenize(test_prompt))

            if test_tokens == prompt_tokens:
                return test_prompt
            elif test_tokens < prompt_tokens:
                left = mid + 1
            else:
                right = mid

        return self.text_creator.create_text(start_index, left)


class SyntheticDatasetCreator(DatasetCreator):
    @classmethod
    def is_supported(cls, data: Any, data_args: Optional[Dict[str, Any]]) -> bool:
        if (
            isinstance(data, Path)
            and data.exists()
            and data.suffix in {".config", ".yaml"}
        ):
            return True

        if isinstance(data, str):
            data_str: str = data.strip()
            if (
                data_str.startswith("{")
                or data_str.count("=") > 1
                or data_str.endswith((".config", ".yaml"))
            ):
                return True

        return False

    @classmethod
    def handle_create(
        cls,
        data: Any,
        data_args: Optional[Dict[str, Any]],
        processor: PreTrainedTokenizerBase,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        config = SyntheticDatasetConfig.parse_str(data)
        generator = SyntheticTextItemsGenerator(config, processor)
        items = list(generator)

        return Dataset.from_list(items)

    @classmethod
    def extract_args_column_mappings(
        cls, data_args: Dict[str, Any], processor: PreTrainedTokenizerBase
    ) -> Dict[ColumnInputTypes, str]:
        super().extract_args_column_mappings(data_args)

        return {
            "prompt_column": "prompt",
            "prompt_tokens_count_column": "prompt_tokens_count",
            "output_tokens_count_column": "output_tokens_count",
        }
