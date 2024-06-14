import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from loguru import logger
from transformers import PreTrainedTokenizer

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

    generated_tokens: int = None
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
        self._random_seed = random_seed
        self._data = self._load_emulated_data()

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new result request item from the data.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """
        prompt, prompt_token_count = self._sample_prompt()
        generated_token_count = self._sample_generated()

        request = TextGenerationRequest(
            prompt=prompt, prompt_token_count=prompt_token_count
        )

        if generated_token_count:
            request.params["generated_tokens"] = generated_token_count

        return request

    def _load_config(self, config: Union[str, Dict]) -> EmulatedConfig:
        # load the config file from a dict, string (json or csv), or file path
        config_dict = None
        if isinstance(config, dict):
            config_dict = config
            logger.info(f"Loaded configuration from dict: {config}")
        elif isinstance(config, str):
            config = config.strip()

            # check if the string is a file path, json, or csv
            if config.endswith(".json"):
                with open(config, "r", encoding="utf-8") as file:
                    config_dict = json.load(file)

                logger.info(f"Loaded configuration from file: {config}")
            elif config.index("{") > -1:
                config_dict = json.loads(config)
                logger.info(f"Loaded configuration from json string: {config}")
            elif config.index(",") > -1:
                # format: key1=value1,key2=value2
                items = config.split(",")
                config_dict = {
                    key: val for key, val in [item.split("=") for item in items]
                }
                logger.info(f"Loaded configuration from csv string: {config}")
            else:
                raise ValueError(f"Invalid configuration string: {config}")
        else:
            raise ValueError(f"Invalid configuration type: {type(config)}")

        # map the config to the EmulatedConfig dataclass
        config = EmulatedConfig(**config_dict)

        return config

    def _load_emulated_data(self) -> List[str]:
        url = "https://www.gutenberg.org/files/1342/1342-0.txt"
        logger.info(f"Downloading text corpus from {url}")
        response = requests.get(url)
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
        lines = (
            cleaned_content.replace(". ", ".\n")
            .replace("! ", "!\n")
            .replace("? ", "?\n")
        )
        lines = lines.split("\n")
        lines = [line.strip() for line in lines if line and line.strip()]

        return lines

    def _token_count(self, text: str) -> int:
        return (
            len(self.tokenizer.tokenize(text)) if self.tokenizer else len(text.split())
        )

    def _sample_prompt(self) -> (str, int):
        prompt_tokens = self._config.prompt_tokens
        prompt_tokens_variance = self._config.prompt_tokens_variance or 0
        prompt_tokens_min = self._config.prompt_tokens_min or 1
        prompt_tokens_max = (
            self._config.prompt_tokens_max or prompt_tokens + 5 * prompt_tokens_variance
            if prompt_tokens_variance
            else 10000
        )
        prompt_tokens_min = max(1, prompt_tokens_min)
        prompt_tokens_max = max(prompt_tokens_min, prompt_tokens_max)

        # Sample a token count for the prompt
        prompt_token_count = max(
            min(
                prompt_tokens
                + np.random.randint(
                    -prompt_tokens_variance, prompt_tokens_variance + 1
                ),
                prompt_tokens_max,
            ),
            prompt_tokens_min,
        )
        random_line_index = np.random.randint(0, len(self._data))

        # Create a sample prompt above the desired token count
        prompt = self._data[random_line_index]

        while self._token_count(prompt) < prompt_token_count:
            prompt += " " + self._data[np.random.randint(0, len(self._data))]

        # Binary search to find the closest token count to the desired token count
        left = 0
        right = len(prompt)
        while left < right:
            mid = (left + right) // 2
            if self._token_count(prompt[:mid]) < prompt_token_count:
                left = mid + 1
            else:
                right = mid

        prompt = prompt[:left]

        return prompt, prompt_token_count

    def _sample_generated(self) -> Optional[int]:
        if not self._config.generated_tokens:
            return None

        generated_tokens = self._config.generated_tokens
        generated_tokens_variance = self._config.generated_tokens_variance or 0
        generated_tokens_min = self._config.generated_tokens_min or 1
        generated_tokens_max = (
            self._config.generated_tokens_max
            or generated_tokens + 5 * generated_tokens_variance
            if generated_tokens_variance
            else 10000
        )
        generated_tokens_min = max(1, generated_tokens_min)
        generated_tokens_max = max(generated_tokens_min, generated_tokens_max)

        # Sample a token count for the generated text
        generated_token_count = max(
            min(
                generated_tokens
                + np.random.randint(
                    -generated_tokens_variance, generated_tokens_variance + 1
                ),
                generated_tokens_max,
            ),
            generated_tokens_min,
        )

        return generated_token_count
