import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from loguru import logger
from transformers import PreTrainedTokenizer

from guidellm.config import settings
from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import RequestGenerator

__all__ = ["EmulatedConfig", "EmulatedRequestGenerator"]


@dataclass
class EmulatedConfig:
    """
    A dataclass to represent the configuration for emulated requests.
    """

    prompt_tokens: int
    prompt_tokens_variance: Optional[int] = None
    prompt_tokens_min: Optional[int] = None
    prompt_tokens_max: Optional[int] = None

    generated_tokens: Optional[int] = None
    generated_tokens_variance: Optional[int] = None
    generated_tokens_min: Optional[int] = None
    generated_tokens_max: Optional[int] = None


class EmulatedRequestGenerator(RequestGenerator):
    """
    A request generator that generates emulated requests based on a configuration.

    :param config: The configuration string or file.
    :type config: Union[str, Dict]
    :param random_seed: The random seed to use for generating requests.
    :type random_seed: Optional[int]
    :param tokenizer: The tokenizer instance or the name/config to use
        for tokenizing prompts.
    :type tokenizer: Union[str, PreTrainedTokenizer]
    :param mode: The generation mode, either 'async' or 'sync'.
    :type mode: str
    :param async_queue_size: The size of the request queue.
    :type async_queue_size: int
    """

    def __init__(
        self,
        config: Union[str, Dict],
        random_seed: Optional[int] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: str = "async",
        async_queue_size: int = 50,
    ):
        super().__init__(tokenizer, mode, async_queue_size)
        self._config = self._load_config(config)
        self._data = self._load_emulated_data()
        self._rng = np.random.default_rng(random_seed)

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new result request item from the data.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """
        prompt, prompt_token_count = self._sample_prompt()
        generated_token_count = self._sample_generated()

        request = TextGenerationRequest(
            prompt=prompt,
            prompt_token_count=prompt_token_count,
        )

        if generated_token_count:
            request.params["generated_tokens"] = generated_token_count

        return request

    def _load_config(self, config: Union[str, Dict]) -> EmulatedConfig:
        # load the config file from a dict, string (json or csv), or file path
        if isinstance(config, dict):
            config_dict = config
            logger.info("Loaded configuration from dict: {}", config)
        elif isinstance(config, str) and config.endswith(".json"):
            with Path(config).open(encoding="utf-8") as file:
                config_dict = json.load(file)

            logger.info("Loaded configuration from file: {}", config)
        elif isinstance(config, str) and (config.index("{") > -1):
            config_dict = json.loads(config.strip())
            logger.info("Loaded configuration from string: {}", config)
        elif isinstance(config, str) and (config.index(",") > -1):
            items = config.split(",")
            config_dict = {}
            for item in items:
                key_value = item.split("=")
                if len(key_value) != 2:  # noqa: PLR2004
                    raise ValueError(f"Unexpected format for item: {item}")
                key, value = key_value
                config_dict[key] = value
            logger.info("Loaded configuration from csv string: {}", config)
        else:
            raise ValueError(
                f"Invalid configuration given for EmulatedRequestGenerator: {config}"
            )

        # map the config to the EmulatedConfig dataclass
        return EmulatedConfig(**config_dict or {})

    def _load_emulated_data(self) -> List[str]:
        url = "https://www.gutenberg.org/files/1342/1342-0.txt"
        logger.info(f"Downloading text corpus from {url}")
        response = requests.get(url, timeout=settings.request_timeout)
        response.raise_for_status()

        content = response.text
        start = content.index(
            "It is a truth universally acknowledged, that a single man in possession"
        )
        end = content.index("CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO.")
        content = content[start:end]

        cleaned_content = (
            content.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        )
        cleaned_content = unicodedata.normalize("NFKD", cleaned_content)
        cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

        # break lines according to punctuation
        lines_text = (
            cleaned_content.replace(". ", ".\n")
            .replace("! ", "!\n")
            .replace("? ", "?\n")
        )
        lines: List[str] = lines_text.split("\n")

        return [line.strip() for line in lines if line.strip()]

    def _token_count(self, text: str) -> int:
        return (
            len(self.tokenizer.tokenize(text)) if self.tokenizer else len(text.split())
        )

    def _sample_prompt(self) -> Tuple[str, int]:
        prompt_token_count = self._sample_tokens(
            self._config.prompt_tokens,
            self._config.prompt_tokens_variance,
            self._config.prompt_tokens_min,
            self._config.prompt_tokens_max,
        )

        prompt = self._data[self._rng.integers(0, len(self._data))]

        while self._token_count(prompt) < prompt_token_count:
            prompt += " " + self._data[self._rng.integers(0, len(self._data))]

        # truncate the prompt to the desired token count
        words = prompt.split()
        left = 0
        right = len(words)
        while left < right:
            mid = (left + right) // 2
            if self._token_count(" ".join(words[:mid])) < prompt_token_count:
                left = mid + 1
            else:
                right = mid
        prompt = " ".join(words[:left])

        return prompt, prompt_token_count

    def _sample_generated(self) -> Optional[int]:
        if not self._config.generated_tokens:
            return None

        return self._sample_tokens(
            self._config.generated_tokens,
            self._config.generated_tokens_variance,
            self._config.generated_tokens_min,
            self._config.generated_tokens_max,
        )

    def _sample_tokens(
        self,
        base: int,
        variance: Optional[int],
        min_tokens: Optional[int],
        max_tokens: Optional[int],
    ) -> int:
        variance = variance or 0
        min_tokens = max(1, min_tokens or 1)
        max_tokens = max(
            min_tokens, max_tokens or base + 5 * variance if variance else 10000
        )

        return max(
            min(
                base + self._rng.integers(-variance, variance + 1),
                max_tokens,
            ),
            min_tokens,
        )
