from unittest.mock import patch

import pytest
from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)

from guidellm.utils.transformers import (
    load_transformers_dataset,
    resolve_transformers_dataset,
    resolve_transformers_dataset_column,
    resolve_transformers_dataset_split,
)
from tests.dummy.data.transformers import (
    create_sample_dataset,
    create_sample_dataset_dict,
    create_sample_iterable_dataset,
    create_sample_iterable_dataset_dict,
)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("dataset_arg", "dataset", "split", "preferred_splits", "expected_type"),
    [
        (
            "mock/directory/file.csv",
            create_sample_dataset_dict(splits=["train"]),
            "train",
            None,
            Dataset,
        ),
        (
            "mock/directory/file.json",
            create_sample_dataset_dict(splits=["test"]),
            None,
            ("train", "test"),
            Dataset,
        ),
        (
            "mock/directory/file.py",
            create_sample_dataset_dict(splits=["test"], column="output"),
            None,
            None,
            Dataset,
        ),
        (
            create_sample_dataset_dict(splits=["val", "train"], column="custom"),
            None,
            "val",
            None,
            Dataset,
        ),
        (
            create_sample_dataset(),
            None,
            None,
            None,
            Dataset,
        ),
        (
            create_sample_iterable_dataset_dict(splits=["validation"]),
            None,
            None,
            None,
            IterableDataset,
        ),
        (
            create_sample_iterable_dataset(),
            None,
            "validation",
            None,
            IterableDataset,
        ),
    ],
)
def test_load_transformers_dataset(
    dataset_arg, dataset, split, preferred_splits, expected_type
):
    with patch(
        "guidellm.utils.transformers.load_dataset",
        return_value=dataset,
    ):
        loaded_dataset = load_transformers_dataset(
            dataset_arg, split=split, preferred_splits=preferred_splits
        )
        assert isinstance(loaded_dataset, expected_type)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("dataset_arg", "dataset", "split", "preferred_splits", "expected_type"),
    [
        (
            "mock/directory/file.csv",
            create_sample_dataset(),
            "train",
            None,
            Dataset,
        ),
        (
            "mock/directory/file.json",
            create_sample_dataset_dict(splits=["test"]),
            None,
            ("train", "test"),
            DatasetDict,
        ),
        (
            "mock/directory/file.py",
            create_sample_dataset_dict(splits=["test"], column="output"),
            None,
            None,
            DatasetDict,
        ),
        (
            "mock/directory/file.unk",
            create_sample_dataset_dict(splits=["test"], column="output"),
            None,
            None,
            DatasetDict,
        ),
        (
            create_sample_dataset_dict(splits=["val", "train"], column="custom"),
            None,
            "val",
            None,
            DatasetDict,
        ),
        (
            create_sample_dataset(),
            None,
            None,
            None,
            Dataset,
        ),
        (
            create_sample_iterable_dataset_dict(splits=["validation"]),
            None,
            None,
            None,
            IterableDatasetDict,
        ),
        (
            create_sample_iterable_dataset(),
            None,
            "validation",
            None,
            IterableDataset,
        ),
    ],
)
def test_resolve_transformers_dataset(
    dataset_arg, dataset, split, preferred_splits, expected_type
):
    with patch(
        "guidellm.utils.transformers.load_dataset",
        return_value=dataset,
    ):
        loaded_dataset = resolve_transformers_dataset(
            dataset_arg, split=split, preferred_splits=preferred_splits
        )
        assert isinstance(loaded_dataset, expected_type)


@pytest.mark.sanity()
def test_resolve_transformers_dataset_invalid():
    with pytest.raises(ValueError):
        resolve_transformers_dataset(123)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("dataset", "split", "preferred_splits", "expected_type"),
    [
        (
            create_sample_dataset(),
            None,
            None,
            Dataset,
        ),
        (
            create_sample_iterable_dataset_dict(splits=["validation"]),
            None,
            None,
            IterableDataset,
        ),
        (
            create_sample_iterable_dataset(),
            "validation",
            None,
            IterableDataset,
        ),
    ],
)
def test_resolve_transformers_dataset_split(
    dataset, split, preferred_splits, expected_type
):
    loaded_dataset = resolve_transformers_dataset_split(
        dataset, split=split, preferred_splits=preferred_splits
    )
    assert isinstance(loaded_dataset, expected_type)


def test_resolve_transformers_dataset_split_missing():
    dataset = create_sample_dataset_dict()
    with pytest.raises(ValueError):
        resolve_transformers_dataset_split(dataset, split="missing")


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("dataset", "column", "preferred_columns", "expected_column"),
    [
        (create_sample_dataset(), None, None, "text"),
        (create_sample_dataset(), "text", None, "text"),
        (create_sample_dataset(), None, ["text"], "text"),
        (create_sample_dataset(), None, ["data"], "text"),
        (create_sample_iterable_dataset(), None, None, "text"),
    ],
)
def test_resolve_transformers_dataset_column(
    dataset, column, preferred_columns, expected_column
):
    resolved_column = resolve_transformers_dataset_column(
        dataset, column=column, preferred_columns=preferred_columns
    )
    assert resolved_column == expected_column


def test_resolve_transformers_dataset_column_missing():
    dataset = create_sample_dataset()
    with pytest.raises(ValueError):
        resolve_transformers_dataset_column(dataset, column="missing")
