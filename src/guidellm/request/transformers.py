from pathlib import Path
from typing import Optional, Union

from datasets import (
    Dataset,
    DatasetDict,  # type: ignore  # noqa: PGH003
    IterableDataset,
    IterableDatasetDict,
)
from loguru import logger
from transformers import PreTrainedTokenizer  # type: ignore  # noqa: PGH003

from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import GenerationMode, RequestGenerator
from guidellm.utils import (
    load_transformers_dataset,
    resolve_transformers_dataset_column,
)

__all__ = ["TransformersDatasetRequestGenerator"]


class TransformersDatasetRequestGenerator(RequestGenerator):
    """
    A request generator implementation for Hugging Face datasets.

    :param dataset: The name of the Hugging Face dataset to use or the path
        to a local dataset.
    :type dataset_name: str
    :param split: The split of the dataset to use (e.g., 'train', 'test').
    :type split: str
    :param column: The column/field to use for generating requests.
    :type column: str
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
        dataset: Union[
            str, Path, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
        ],
        split: Optional[str] = None,
        column: Optional[str] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: GenerationMode = "async",
        async_queue_size: int = 50,
        **kwargs,
    ):
        self._dataset = dataset
        self._split = split
        self._column = column
        self._kwargs = kwargs

        self._hf_dataset: Union[Dataset, IterableDataset] = load_transformers_dataset(
            dataset, split=split, **kwargs
        )
        self._hf_column = resolve_transformers_dataset_column(
            self._hf_dataset, column=column
        )
        self._hf_dataset_iterator = iter(self._hf_dataset)

        # NOTE: Must be after all the parameters since the queue population
        #       function requires attributes above
        super().__init__(
            type_="transformers_dataset",
            source=str(dataset),
            tokenizer=tokenizer,
            mode=mode,
            async_queue_size=async_queue_size,
        )

    def __len__(self) -> int:
        if not isinstance(self._hf_dataset, Dataset):
            raise ValueError("Can't get dataset size for IterableDataset object")
        else:
            return len(self._hf_dataset)

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new result request item from the dataset.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """

        logger.debug("Creating new request item from dataset")

        try:
            data = next(self._hf_dataset_iterator)
        except StopIteration:
            self._hf_dataset_iterator = iter(self._hf_dataset)
            data = next(self._hf_dataset_iterator)

        prompt = data[self._hf_column]
        token_count = len(self.tokenizer.tokenize(prompt))
        request = TextGenerationRequest(
            prompt=prompt,
            prompt_token_count=token_count,
        )
        logger.debug(f"Created new TextGenerationRequest: {request}")

        return request
