from pathlib import Path
from typing import List, Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loguru import logger

from guidellm.config import settings

__all__ = [
    "load_transformers_dataset",
    "resolve_transformers_dataset",
    "resolve_transformers_dataset_column",
    "resolve_transformers_dataset_split",
]


def load_transformers_dataset(
    dataset: Union[
        str, Path, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
    ],
    split: Optional[str] = None,
    preferred_splits: Optional[List[str]] = settings.dataset.preferred_data_splits,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    """
    Load a dataset from a file or a script and resolve the preferred split.

    :param dataset: the dataset file or script to load
    :param split: the dataset split to use
        (overrides preferred_splits, must be in dataset)
    :param preferred_splits: the preferred dataset splits to use
    :param kwargs: additional keyword arguments to pass to the dataset loader
    :return: the loaded dataset
    """
    dataset = resolve_transformers_dataset(dataset, **kwargs)

    return resolve_transformers_dataset_split(dataset, split, preferred_splits)


def resolve_transformers_dataset(
    dataset: Union[
        str, Path, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
    ],
    **kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """
    Resolve the dataset from a file (csv, json, script) or a dataset name.

    :param dataset: the dataset file or script to load
    :param kwargs: additional keyword arguments to pass to the dataset loader
    :return: the loaded dataset
    """
    if isinstance(
        dataset, (DatasetDict, Dataset, IterableDatasetDict, IterableDataset)
    ):
        return dataset

    if not isinstance(dataset, (str, Path)):
        raise ValueError(f"Invalid dataset type: {type(dataset)}")

    dataset = str(dataset)

    if dataset.endswith((".csv", ".json")):
        logger.debug("Loading dataset from local path: {}", dataset)
        extension = dataset.split(".")[-1]

        return load_dataset(extension, data_files=dataset, **kwargs)

    if dataset.endswith(".py"):
        logger.debug("Loading dataset from local script: {}", dataset)

        return load_dataset(dataset, **kwargs)

    logger.debug("Loading dataset: {}", dataset)

    return load_dataset(dataset, **kwargs)


def resolve_transformers_dataset_split(
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    split: Optional[str] = None,
    preferred_splits: Optional[List[str]] = settings.dataset.preferred_data_splits,
) -> Union[Dataset, IterableDataset]:
    """
    Resolve the preferred split from a dataset dictionary.

    :param dataset: the dataset to resolve the split from
    :param split: the dataset split to use
        (overrides preferred_splits, must be in dataset)
    :param preferred_splits: the preferred dataset splits to use
    :return: the resolved dataset split
    """
    if not isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        logger.debug("Dataset is not a dictionary, using default split")
        return dataset

    if split:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset")

        return dataset[split]

    if preferred_splits:
        for spl in preferred_splits:
            if spl not in dataset:
                continue
            return dataset[spl]

    return list(dataset.values())[0]


def resolve_transformers_dataset_column(
    dataset: Union[Dataset, IterableDataset],
    column: Optional[str] = None,
    preferred_columns: Optional[List[str]] = settings.dataset.preferred_data_columns,
) -> str:
    """
    Resolve the preferred column from a dataset.

    :param dataset: the dataset to resolve the column from
    :param column: the dataset column to use
        (overrides preferred_columns, must be in dataset)
    :param preferred_columns: the preferred dataset columns to use
    :return: the resolved dataset column
    """
    column_names = dataset.column_names

    if not column_names:
        # grab from the first item
        first_item = next(iter(dataset))
        column_names = list(first_item.keys())

    if column:
        if column not in column_names:
            raise ValueError(f"Column '{column}' not found in dataset")

        return column

    if preferred_columns:
        for col in preferred_columns:
            if col not in column_names:
                continue
            return col

    return list(column_names)[0]
