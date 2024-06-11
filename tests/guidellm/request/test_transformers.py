import pytest
from unittest.mock import patch
from guidellm.request import TransformersDatasetRequestGenerator


@pytest.fixture
def mock_load_dataset():
    with patch("guidellm.request.transformers.load_dataset") as mock:
        mock.return_value = {
            "train": [{"text": "Sample text 1"}, {"text": "Sample text 2"}],
            "test": [{"text": "Sample text 3"}, {"text": "Sample text 4"}],
        }
        yield mock


@pytest.fixture
def mock_tokenizer():
    with patch("guidellm.request.transformers.AutoTokenizer.from_pretrained") as mock:
        mock_instance = mock.return_value
        mock_instance.return_value = {"input_ids": [0, 1, 2, 3, 4]}
        yield mock_instance


def test_transformers_request_generator_load_dataset(mock_load_dataset):
    generator = TransformersDatasetRequestGenerator(
        dataset="mock_dataset", split=None, column=None, mode="sync"
    )

    assert generator._split == "test"
    assert generator._column == "text"
    assert generator._hf_dataset == mock_load_dataset.return_value["test"]


def test_transformers_request_generator_create_item_no_tokenizer(mock_load_dataset):
    generator = TransformersDatasetRequestGenerator(
        dataset="mock_dataset", split="train", column="text", mode="sync"
    )

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"
    assert items[0].token_count is None


def test_transformers_request_generator_create_item_with_tokenizer(
    mock_load_dataset, mock_tokenizer
):
    generator = TransformersDatasetRequestGenerator(
        dataset="mock_dataset",
        split="train",
        column="text",
        tokenizer="mock_tokenizer",
        mode="sync",
    )

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"
    assert items[0].token_count == 5
    assert items[1].token_count == 5


@pytest.mark.asyncio
async def test_transformers_request_generator_async(mock_load_dataset):
    generator = TransformersDatasetRequestGenerator(
        dataset="mock_dataset",
        split="train",
        column="text",
        mode="async",
        async_queue_size=10,
    )
    generator.start()

    # Collect a few items from the generator
    items = []
    for _ in range(2):
        items.append(await generator._queue.get())
        generator._queue.task_done()

    generator.stop()
    await generator._populating_task

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"
