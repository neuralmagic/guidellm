import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from transformers import PreTrainedTokenizer  # type: ignore  # noqa: PGH003

from guidellm.config import settings
from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import GenerationMode, RequestGenerator
from guidellm.utils import clean_text, filter_text, load_text, split_text

__all__ = ["EmulatedConfig", "EmulatedRequestGenerator", "EndlessTokens"]


@dataclass
class EmulatedConfig:
    """
    Configuration for emulated text generation requests.

    Args:
        prompt_tokens (int): Number of prompt tokens.
        prompt_tokens_variance (Optional[int]): Variance for prompt tokens.
        prompt_tokens_min (Optional[int]): Minimum number of prompt tokens.
        prompt_tokens_max (Optional[int]): Maximum number of prompt tokens.
        generated_tokens (Optional[int]): Number of generated tokens.
        generated_tokens_variance (Optional[int]): Variance for generated tokens.
        generated_tokens_min (Optional[int]): Minimum number of generated tokens.
        generated_tokens_max (Optional[int]): Maximum number of generated tokens.
    """

    @staticmethod
    def create_config(config: Optional[Union[str, Path, Dict]]) -> "EmulatedConfig":
        """
        Create an EmulatedConfig instance from a configuration source.

        :param config: Configuration source, can be a dictionary, JSON string,
            key=value string, or file path.
        :type config: Union[str, Path, Dict]
        :return: An instance of EmulatedConfig.
        :rtype: EmulatedConfig
        :raises FileNotFoundError: If the configuration file is not found.
        :raises ValueError: If the configuration format is invalid.
        """
        if not config:
            logger.debug("Creating default configuration")
            return EmulatedConfig(prompt_tokens=1024, generated_tokens=256)

        if isinstance(config, dict):
            logger.debug("Loading configuration from dict: {}", config)
            return EmulatedConfig(**config)

        if isinstance(config, Path) or (
            isinstance(config, str) and (config.endswith(".json") or "{" in config)
        ):
            logger.debug("Loading configuration from json: {}", config)

            if isinstance(config, str) and "{" in config:
                json_text = config.strip()
            else:
                if isinstance(config, str):
                    config = Path(config)

                if not config.exists():
                    raise FileNotFoundError(f"Configuration file not found: {config}")

                json_text = config.read_text(encoding="utf-8")

            json_dict = json.loads(json_text)

            return EmulatedConfig(**json_dict)

        if isinstance(config, str) and "=" in config:
            logger.debug("Loading configuration from csv string: {}", config)
            items = config.split(",")
            config_dict = {}
            for item in items:
                key_value = item.strip().split("=")
                if len(key_value) != 2:  # noqa: PLR2004
                    raise ValueError(f"Unexpected format for item: {item}")
                key = key_value[0].strip()
                value = (
                    int(key_value[1].strip())
                    if key_value[1].isnumeric()
                    else key_value[1]
                )
                config_dict[key] = value

            return EmulatedConfig(**config_dict)  # type: ignore # noqa: PGH003

        raise ValueError(
            f"Invalid configuration given for creation of EmulatedConfig: {config}"
        )

    prompt_tokens: int
    prompt_tokens_variance: Optional[int] = None
    prompt_tokens_min: Optional[int] = None
    prompt_tokens_max: Optional[int] = None

    generated_tokens: Optional[int] = None
    generated_tokens_variance: Optional[int] = None
    generated_tokens_min: Optional[int] = None
    generated_tokens_max: Optional[int] = None

    @property
    def prompt_tokens_range(self) -> Tuple[int, int]:
        """
        Get the range (min, max) of prompt tokens to generate.

        :return: The range of prompt tokens.
        :rtype: Tuple[int, int]
        """
        return self._token_range(
            self.prompt_tokens,
            self.prompt_tokens_variance,
            self.prompt_tokens_min,
            self.prompt_tokens_max,
        )

    @property
    def output_tokens_range(self) -> Tuple[int, int]:
        """
        Get the range (min, max) of output tokens to generate.

        :return: The range of generated tokens.
        :rtype: Tuple[int, int]
        """
        if not self.generated_tokens:
            return 0, 0

        return self._token_range(
            self.generated_tokens,
            self.generated_tokens_variance,
            self.generated_tokens_min,
            self.generated_tokens_max,
        )

    def sample_prompt_tokens(self, rng: np.random.Generator) -> int:
        """
        Sample the number of prompt tokens to generate.

        :param rng: The random number generator to use.
        :type rng: np.random.Generator
        :return: The number of prompt tokens to create.
        :rtype: int
        """
        return self._sample_tokens(
            self.prompt_tokens,
            self.prompt_tokens_variance,
            self.prompt_tokens_min,
            self.prompt_tokens_max,
            rng,
        )

    def sample_output_tokens(self, rng: np.random.Generator) -> Optional[int]:
        """
        Sample the number of output tokens to generate.

        :param rng: The random number generator to use.
        :type rng: np.random.Generator
        :return: The number of output tokens to generate.
        :rtype: Optional[int]
        """
        if not self.generated_tokens:
            return None

        return self._sample_tokens(
            self.generated_tokens,
            self.generated_tokens_variance,
            self.generated_tokens_min,
            self.generated_tokens_max,
            rng,
        )

    @staticmethod
    def _sample_tokens(
        base: int,
        variance: Optional[int],
        min_tokens: Optional[int],
        max_tokens: Optional[int],
        rng: np.random.Generator,
    ) -> int:
        min_tokens, max_tokens = EmulatedConfig._token_range(
            base, variance, min_tokens, max_tokens
        )

        if min_tokens == max_tokens:
            return min_tokens

        if not variance:
            return rng.integers(min_tokens, max_tokens + 1)

        rand = rng.normal(base, math.sqrt(variance))

        return int(min(max(rand, min_tokens), max_tokens))

    @staticmethod
    def _token_range(
        base: int,
        variance: Optional[int],
        min_tokens: Optional[int],
        max_tokens: Optional[int],
    ) -> Tuple[int, int]:
        if not variance:
            return (
                min_tokens or base,
                max_tokens or base,
            )

        min_tokens = min_tokens if min_tokens and min_tokens > 0 else 1
        max_tokens = (
            max_tokens if max_tokens and max_tokens > base else base + 5 * variance
        )

        return min_tokens, max_tokens


class EndlessTokens(List[str]):
    """
    A list subclass that allows for endless data generation.
    """

    def __init__(
        self,
        data: Union[str, Path],
        filter_start: Optional[Union[str, int]] = None,
        filter_end: Optional[Union[str, int]] = None,
        clean_text_args: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize EndlessDataWords with data.

        :param data: Source text data.
        :type data: str
        """
        logger.debug("Loading data from: {}", data)
        data = load_text(data)
        data = filter_text(data, filter_start, filter_end)
        data = (
            clean_text(data)
            if not clean_text_args
            else clean_text(data, **clean_text_args)
        )
        self._tokens, self._token_separators, self._line_indices = split_text(data)

        super().__init__(self._tokens)

    @property
    def line_indices(self) -> List[int]:
        """
        Get the list of start indices for lines.

        :return: List of start indices.
        :rtype: List[int]
        """
        return self._line_indices

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
        buff_token_sep = ""

        for counter in range(length):
            index = (start + counter) % len(self)
            text += buff_token_sep + self[index]
            buff_token_sep = self._token_separators[index]

        return text


class EmulatedRequestGenerator(RequestGenerator):
    """
    A request generator that generates emulated requests based on a configuration.

    :param config: The configuration string, file path, or dictionary.
    :type config: Union[str, Dict, Path]
    :param random_seed: The random seed to use for generating requests.
    :type random_seed: Optional[int]
    :param tokenizer: The tokenizer instance or the name/config to use
        for tokenizing prompts.
    :type tokenizer: Optional[Union[str, PreTrainedTokenizer]]
    :param mode: The generation mode, either 'async' or 'sync'.
    :type mode: GenerationMode
    :param async_queue_size: The size of the request queue.
    :type async_queue_size: int
    """

    def __init__(
        self,
        config: Optional[Union[str, Path, Dict]],
        random_seed: Optional[int] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: GenerationMode = "async",
        async_queue_size: int = 50,
    ):
        """
        Initialize EmulatedRequestGenerator with configuration and tokenizer.

        :param config: Configuration source, can be a dictionary,
            JSON string, or file path.
        :type config: Optional[Union[str, Path, Dict]]
        :param random_seed: Optional seed for random number generator.
        :type random_seed: Optional[int]
        :param tokenizer: Tokenizer instance or configuration for tokenizing prompts.
        :type tokenizer: Optional[Union[str, PreTrainedTokenizer]]
        :param mode: Mode of request generation, either 'async' or 'sync'.
        :type mode: str
        :param async_queue_size: Size of the asynchronous queue.
        :type async_queue_size: int
        """
        self._config = EmulatedConfig.create_config(config)
        self._tokens = EndlessTokens(
            settings.emulated_data.source,
            settings.emulated_data.filter_start,
            settings.emulated_data.filter_end,
        )
        self._rng = np.random.default_rng(random_seed)

        # NOTE: Must be after all the parameters since the queue population
        #       function requires attributes above
        super().__init__(
            type_="emulated",
            source=str(config),
            tokenizer=tokenizer,
            mode=mode,
            async_queue_size=async_queue_size,
        )

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new text generation request item from the data.

        :return: A new text generation request.
        :rtype: TextGenerationRequest
        """
        logger.debug("Creating new text generation request")
        target_prompt_token_count = self._config.sample_prompt_tokens(self._rng)
        prompt = self.sample_prompt(target_prompt_token_count)
        prompt_token_count = len(self.tokenizer.tokenize(prompt))
        output_token_count = self._config.sample_output_tokens(self._rng)
        logger.debug("Generated prompt: {}", prompt)

        return TextGenerationRequest(
            prompt=prompt,
            prompt_token_count=prompt_token_count,
            output_token_count=output_token_count,
        )

    def sample_prompt(self, tokens: int) -> str:
        """
        Sample a prompt with the specified number of tokens.

        :param tokens: Number of tokens for the prompt.
        :type tokens: int
        :return: Sampled prompt text.
        :rtype: str
        """
        start_line_index = self._rng.integers(0, len(self._tokens.line_indices))

        # binary search to find the proper number of tokens for the prompt
        # this is because tokenizers differ in tokenization behavior
        left = 0
        right = left + 5 * tokens

        while left < right:
            mid = (left + right) // 2
            prompt = self._tokens.create_text(start_line_index, mid)
            token_count = len(self.tokenizer.tokenize(prompt))

            if token_count == tokens:
                return prompt

            if token_count < tokens:
                left = mid + 1
            else:
                right = mid

        return self._tokens.create_text(start_line_index, left)
