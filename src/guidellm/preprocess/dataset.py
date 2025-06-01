import json
import os
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml
from datasets import Dataset
from loguru import logger
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from guidellm.dataset import load_dataset as guidellm_load_dataset
from guidellm.utils import IntegerRangeSampler, check_load_processor

SUPPORTED_TYPES = {
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
}


class PromptTooShortError(Exception):
    pass


class ShortPromptStrategy(str, Enum):
    IGNORE = "ignore"
    CONCATENATE = "concatenate"
    PAD = "pad"
    ERROR = "error"


def handle_ignore_strategy(
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
) -> Optional[str]:
    if len(tokenizer.encode(current_prompt)) < min_prompt_tokens:
        logger.warning("Prompt too short, ignoring")
        return None
    return current_prompt


def handle_concatenate_strategy(
        current_prompt: str,
        min_prompt_tokens: int,
        dataset_iterator: Iterator[dict[str, Any]],
        prompt_column: str,
        tokenizer: PreTrainedTokenizerBase,
        concat_delimiter: str,
        **_kwargs,
) -> Optional[str]:
    tokens_len = len(tokenizer.encode(current_prompt))
    while tokens_len < min_prompt_tokens:
        try:
            next_row = next(dataset_iterator)
        except StopIteration:
            logger.warning(
                "Could not concatenate enough prompts to reach minimum length, ignoring"
            )
            return None
        current_prompt += concat_delimiter + next_row[prompt_column]
        tokens_len = len(tokenizer.encode(current_prompt))
    return current_prompt


def handle_pad_strategy(
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        pad_char: str,
        **_kwargs,
) -> str:
    while len(tokenizer.encode(current_prompt)) < min_prompt_tokens:
        current_prompt += pad_char
    return current_prompt


def handle_error_strategy(
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
) -> Optional[str]:
    prompt_len = len(tokenizer.encode(current_prompt))
    if prompt_len < min_prompt_tokens:
        raise PromptTooShortError(
            f"Found too short prompt: {current_prompt}, with length: {prompt_len}. "
            f"Minimum length required: {min_prompt_tokens}.",
        )
    return current_prompt


STRATEGY_HANDLERS: dict[ShortPromptStrategy, Callable] = {
    ShortPromptStrategy.IGNORE: handle_ignore_strategy,
    ShortPromptStrategy.CONCATENATE: handle_concatenate_strategy,
    ShortPromptStrategy.PAD: handle_pad_strategy,
    ShortPromptStrategy.ERROR: handle_error_strategy,
}


class TokensConfig(BaseModel):
    average: int = Field(
        description="The average number of tokens.",
        gt=0,
    )
    stdev: Optional[int] = Field(
        description="The standard deviation of the tokens.",
        gt=0,
        default=None,
    )
    min: Optional[int] = Field(
        description="The minimum number of tokens.",
        gt=0,
        default=None,
    )
    max: Optional[int] = Field(
        description="The maximum number of tokens.",
        gt=0,
        default=None,
    )

    @staticmethod
    def parse_str(data: Union[str, Path]) -> "TokensConfig":
        if (
            isinstance(data, Path)
            or data.strip().endswith(".config")
            or data.strip().endswith(".yaml")
        ):
            return TokensConfig.parse_config_file(data)

        if data.strip().startswith("{"):
            return TokensConfig.parse_json(data)

        if data.count("=") > 1:
            return TokensConfig.parse_key_value_pairs(data)

        raise ValueError(
            f"Unsupported data format. Expected JSON or key-value pairs, got {data}"
        )

    @staticmethod
    def parse_json(data: str) -> "TokensConfig":
        config_dict = json.loads(data.strip())

        return TokensConfig(**config_dict)

    @staticmethod
    def parse_key_value_pairs(data: str) -> "TokensConfig":
        config_dict = {}
        items = data.strip().split(",")
        for item in items:
            key, value = item.split("=")
            config_dict[key.strip()] = (
                int(value.strip()) if value.strip().isnumeric() else value.strip()
            )

        return TokensConfig(**config_dict)  # type: ignore[arg-type]

    @staticmethod
    def parse_config_file(data: Union[str, Path]) -> "TokensConfig":
        with Path(data).open("r") as file:
            config_dict = yaml.safe_load(file)

        return TokensConfig(**config_dict)

def save_dataset_to_file(dataset: Dataset, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        dataset.to_csv(output_path)
    elif suffix in {".json", ".jsonl"}:
        dataset.to_json(output_path)
    elif suffix == ".parquet":
        dataset.to_parquet(output_path)
    else:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path'{output_path}'."
            f" Only {SUPPORTED_TYPES} are supported."
        )


def _validate_output_suffix(output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path '{output_path}'. "
            f"Only {SUPPORTED_TYPES} are supported."
        )


def process_dataset(
        data: Union[str, Path],
        output_path: Union[str, Path],
        processor: Union[str, Path, PreTrainedTokenizerBase],
        prompt_tokens: Union[str, Path],
        output_tokens: Union[str, Path],
        processor_args: Optional[dict[str, Any]] = None,
        data_args: Optional[dict[str, Any]] = None,
        short_prompt_strategy: ShortPromptStrategy = ShortPromptStrategy.IGNORE,
        pad_char: Optional[str] = None,
        concat_delimiter: Optional[str] = None,
        push_to_hub: bool = False,
        hub_dataset_id: Optional[str] = None,
        random_seed: int = 42,
) -> None:
    _validate_output_suffix(output_path)
    logger.info(
        f"Starting dataset conversion | Input: {data} | "
        f"Output directory: {output_path}"
    )

    dataset, column_mappings = guidellm_load_dataset(
        data, data_args, processor, processor_args
    )
    tokenizer = check_load_processor(
        processor,
        processor_args,
        "dataset conversion.",
    )
    prompt_column = column_mappings.get("prompt_column")
    output_column = column_mappings.get(
        "output_tokens_count_column", "output_tokens_count"
    )

    prompt_tokens_cfg = TokensConfig.parse_str(prompt_tokens)
    output_tokens_cfg = TokensConfig.parse_str(output_tokens)

    prompt_token_sampler = iter(
        IntegerRangeSampler(
            average=prompt_tokens_cfg.average,
            variance=prompt_tokens_cfg.stdev,
            min_value=prompt_tokens_cfg.min,
            max_value=prompt_tokens_cfg.max,
            random_seed=random_seed,
        )
    )

    output_token_sampler = iter(
        IntegerRangeSampler(
            average=output_tokens_cfg.average,
            variance=output_tokens_cfg.stdev,
            min_value=output_tokens_cfg.min,
            max_value=output_tokens_cfg.max,
            random_seed=random_seed,
        )
    )

    dataset_iterator = iter(dataset)
    processed_prompts = []
    prompt_handler = STRATEGY_HANDLERS[short_prompt_strategy]

    for prompt_row in dataset_iterator:
        prompt_text = prompt_row[prompt_column]
        target_prompt_len = next(prompt_token_sampler)

        prompt_text = prompt_handler(
            current_prompt=prompt_text,
            min_prompt_tokens=target_prompt_len,
            dataset_iterator=dataset_iterator,
            prompt_column=prompt_column,
            tokenizer=tokenizer,
            pad_char=pad_char,
            concat_delimiter=concat_delimiter,
        )
        if prompt_text is None:
            continue

        if len(tokenizer.encode(prompt_text)) > target_prompt_len:
            tokens = tokenizer.encode(prompt_text)
            prompt_text = tokenizer.decode(tokens[:target_prompt_len])

        processed_prompt = prompt_row.copy()
        processed_prompt[prompt_column] = prompt_text
        processed_prompt["prompt_tokens_count"] = target_prompt_len
        processed_prompt[output_column] = next(output_token_sampler)

        processed_prompts.append(processed_prompt)

    if not processed_prompts:
        logger.error("No prompts remained after processing")
        return

    logger.info(f"Generated processed dataset with {len(processed_prompts)} prompts")

    processed_dataset = Dataset.from_list(processed_prompts)
    save_dataset_to_file(processed_dataset, output_path)
    logger.info(f"Conversion complete. Dataset saved to: {output_path}")

    if push_to_hub:
        push_dataset_to_hub(hub_dataset_id, processed_dataset)
        logger.info(f"Pushed dataset to: {hub_dataset_id}")


def push_dataset_to_hub(
        hub_dataset_id: Optional[str], processed_dataset: Dataset,
) -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hub_dataset_id or not hf_token:
        raise ValueError(
            "hub_dataset_id and HF_TOKEN env var must be provided when push_to_hub"
            " is True"
        )
    processed_dataset.push_to_hub(hub_dataset_id, token=hf_token)
