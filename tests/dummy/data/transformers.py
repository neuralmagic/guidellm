from typing import Iterable

from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)


def create_sample_dataset(
    column: str = "text", pattern: str = "sample text {}"
) -> Dataset:
    return Dataset.from_dict({column: [pattern.format(ind) for ind in range(1, 4)]})


def create_sample_iterable_dataset(
    column: str = "text", pattern: str = "sample text {}"
) -> IterableDataset:
    def _generator():
        for ind in range(1, 4):
            yield {column: pattern.format(ind)}

    return IterableDataset.from_generator(_generator)


def create_sample_dataset_dict(
    splits: Iterable[str] = ("train", "test"),
    column: str = "text",
    pattern: str = "sample text {}",
):
    return DatasetDict(
        {
            split: create_sample_dataset(column=column, pattern=pattern)
            for split in splits
        }
    )


def create_sample_iterable_dataset_dict(
    splits: Iterable[str] = ("train", "test"),
    column: str = "text",
    pattern: str = "sample text {}",
):
    return IterableDatasetDict(
        {
            split: create_sample_iterable_dataset(column=column, pattern=pattern)
            for split in splits
        }
    )
