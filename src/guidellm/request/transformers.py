from typing import Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loguru import logger
from transformers import PreTrainedTokenizer

from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import RequestGenerator
from guidellm.utils import PREFERRED_DATA_COLUMNS, PREFERRED_DATA_SPLITS

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
        dataset: str,
        split: Optional[str] = None,
        column: Optional[str] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: str = "async",
        async_queue_size: int = 50,
        **kwargs,
    ):
        self._dataset = dataset
        self._split = split
        self._column = column
        self._kwargs = kwargs
        self._hf_dataset = self._load_dataset()
        self._iterator = iter(self._hf_dataset)

        # NOTE: Must be after all the parameters since the queue population
        #       function requires attributes above
        super().__init__(tokenizer, mode, async_queue_size)

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new result request item from the dataset.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """

        data = next(self._iterator)

        prompt = data[self._column] if self._column in data else str(data)
        token_count = (
            self._tokenizer(prompt)["input_ids"].shape[0] if self._tokenizer else None
        )
        request = TextGenerationRequest(
            prompt=prompt,
            prompt_token_count=token_count,
        )
        logger.debug(f"Created new TextGenerationRequest: {request}")

        return request

    def _load_dataset(self) -> Dataset:
        dataset = self._load_hf_dataset()

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            split = self._load_data_split(dataset)

            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in dataset")

            dataset = dataset[split]
        else:
            self._split = str(dataset.split) if dataset else None

        column = self._load_data_column(dataset)

        if column not in dataset.column_names:
            raise ValueError(f"Column '{column}' not found in dataset")

        logger.info(
            f"Loaded dataset {self._dataset} with split: {self._split} "
            f"and column: {self._column}",
        )

        return dataset

    def _load_hf_dataset(
        self,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        if self._dataset.endswith(".csv") or self._dataset.endswith(".json"):
            logger.debug(f"Loading dataset from local path: {self._dataset}")
            extension = self._dataset.split(".")[-1]

            return load_dataset(extension, data_files=self._dataset, **self._kwargs)

        if self._dataset.endswith(".py"):
            logger.debug(f"Loading dataset from local script: {self._dataset}")

            return load_dataset(self._dataset, **self._kwargs)

        logger.debug(f"Loading dataset: {self._dataset}")

        return load_dataset(self._dataset, **self._kwargs)

    def _load_data_split(self, dataset: Union[DatasetDict, IterableDatasetDict]) -> str:
        if self._split:
            return self._split

        for split in PREFERRED_DATA_SPLITS:
            if split in dataset:
                self._split = split
                break
        if self._split is None:
            self._split = list(dataset)[0]

        logger.info(f"Inferred split to use: {self._split}")

        return self._split

    def _load_data_column(self, dataset: Union[Dataset, IterableDataset]) -> str:
        if self._column:
            return self._column

        for col in PREFERRED_DATA_COLUMNS:
            if col in dataset.column_names:
                self._column = col
                break
        if self._column is None:
            self._column = list(dataset.column_names)[0]

        logger.info(f"Inferred column to use for prompts: {self._column}")

        return self._column
