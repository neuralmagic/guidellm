from unittest.mock import patch

import pytest

from guidellm.core.request import TextGenerationRequest
from guidellm.request.transformers import TransformersDatasetRequestGenerator
from tests.dummy.data.transformers import (
    create_sample_dataset,
    create_sample_dataset_dict,
    create_sample_iterable_dataset,
    create_sample_iterable_dataset_dict,
)


@pytest.mark.smoke()
def test_transformers_dataset_request_generator_constructor(
    mock_auto_tokenizer,
):
    dataset = create_sample_dataset()
    with patch(
        "guidellm.request.transformers.load_transformers_dataset",
        return_value=dataset,
    ), patch(
        "guidellm.request.transformers.resolve_transformers_dataset_column",
        return_value="text",
    ):
        generator = TransformersDatasetRequestGenerator(
            dataset="dummy_dataset",
            split="train",
            column="text",
            tokenizer="mock-tokenizer",
        )
        assert generator._dataset == "dummy_dataset"
        assert generator._split == "train"
        assert generator._column == "text"
        assert generator._hf_dataset == dataset
        assert generator._hf_column == "text"
        assert generator._hf_dataset_iterator is not None


@pytest.mark.smoke()
def test_transformers_dataset_request_generator_create_item(
    mock_auto_tokenizer,
):
    generator = TransformersDatasetRequestGenerator(
        dataset=create_sample_dataset_dict(),
        split="train",
        column="text",
        tokenizer="mock-tokenizer",
        mode="sync",
    )
    request = generator.create_item()
    assert isinstance(request, TextGenerationRequest)
    assert request.prompt == "sample text 1"
    assert request.prompt_token_count == 3


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("dataset_arg", "dataset"),
    [
        (
            "mock/directory/file.csv",
            create_sample_dataset_dict(splits=["train"]),
        ),
        (
            "mock/directory/file.json",
            create_sample_dataset(column="prompt"),
        ),
        (
            "mock/directory/file.py",
            create_sample_dataset_dict(splits=["test"], column="output"),
        ),
        (create_sample_dataset_dict(splits=["val", "train"], column="custom"), None),
        (create_sample_dataset(), None),
        (create_sample_iterable_dataset_dict(splits=["validation"]), None),
        (create_sample_iterable_dataset(), None),
    ],
)
def test_transformers_dataset_request_generator_lifecycle(
    mock_auto_tokenizer, dataset_arg, dataset
):
    with patch(
        "guidellm.utils.transformers.load_dataset",
        return_value=dataset,
    ):
        generator = TransformersDatasetRequestGenerator(
            dataset=dataset_arg, tokenizer="mock-tokenizer", mode="sync"
        )

        for index, request in enumerate(generator):
            assert isinstance(request, TextGenerationRequest)
            assert request.prompt == f"sample text {index + 1}"
            assert request.prompt_token_count == 3

            if index == 2:
                break


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("dataset_arg", "dataset"),
    [
        (
            "mock/directory/file.csv",
            create_sample_dataset_dict(splits=["train"]),
        ),
        (
            "mock/directory/file.json",
            create_sample_dataset(column="prompt"),
        ),
        (
            "mock/directory/file.py",
            create_sample_dataset_dict(splits=["test"], column="output"),
        ),
        (create_sample_dataset_dict(splits=["val", "train"], column="custom"), None),
        (create_sample_dataset(), None)
    ],
)
def test_transformers_dataset_request_generator_len(
    mock_auto_tokenizer, dataset_arg, dataset
):
    with patch(
        "guidellm.utils.transformers.load_dataset",
        return_value=dataset,
    ):
        generator = TransformersDatasetRequestGenerator(
            dataset=dataset_arg, tokenizer="mock-tokenizer", mode="sync"
        )

        # Check if __len__ returns the correct length
        assert len(generator) == 3
