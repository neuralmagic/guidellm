import pytest
import json
from unittest.mock import patch
from transformers import AutoTokenizer
from guidellm.core.request import BenchmarkRequest
from guidellm.request import EmulatedRequestGenerator


@pytest.fixture
def mock_tokenizer():
    with patch(
        "guidellm.request.emulated_request_generator.AutoTokenizer.from_pretrained"
    ) as mock:
        mock_instance = mock.return_value
        mock_instance.return_value = {"input_ids": [0, 1, 2, 3, 4]}
        yield mock_instance


@pytest.fixture
def mock_text_corpus():
    return (
        "The quick brown fox jumps over the lazy dog. "
        "All work and no play makes Jack a dull boy. "
        "To be or not to be, that is the question. "
        "It was the best of times, it was the worst of times. "
        "Call me Ishmael."
    )


def test_emulated_request_generator_no_tokenizer(mock_text_corpus):
    config = {
        "num_requests": 2,
        "prompt_token_distribution": {"min_length": 3, "max_length": 10},
        "generated_token_distribution": {"min": 10, "max": 15},
    }

    with patch.object(
        EmulatedRequestGenerator, "_load_text_corpus", return_value=mock_text_corpus
    ):
        with patch("transformers.AutoTokenizer"):
            generator = EmulatedRequestGenerator(config=json.dumps(config), mode="sync")

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt.startswith("The quick brown fox")
    assert items[1].prompt.startswith("All work and no play")
    assert items[0].token_count == 5  # Based on mock_tokenizer
    assert items[1].token_count == 5  # Based on mock_tokenizer
    assert items[0].generated_token_count >= 10
    assert items[0].generated_token_count <= 15


def test_emulated_request_generator_with_tokenizer(mock_tokenizer, mock_text_corpus):
    config = {
        "num_requests": 2,
        "prompt_token_distribution": {"min_length": 3, "max_length": 10},
        "generated_token_distribution": {"min": 10, "max": 15},
    }

    with patch.object(
        EmulatedRequestGenerator, "_load_text_corpus", return_value=mock_text_corpus
    ):
        generator = EmulatedRequestGenerator(
            config=json.dumps(config), tokenizer="mock_tokenizer", mode="sync"
        )

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt.startswith("The quick brown fox")
    assert items[1].prompt.startswith("All work and no play")
    assert items[0].token_count == 5  # Based on mock_tokenizer
    assert items[1].token_count == 5  # Based on mock_tokenizer
    assert items[0].generated_token_count >= 10
    assert items[0].generated_token_count <= 15


@pytest.mark.asyncio
async def test_emulated_request_generator_async(mock_tokenizer, mock_text_corpus):
    config = {
        "num_requests": 2,
        "prompt_token_distribution": {"min_length": 3, "max_length": 10},
        "generated_token_distribution": {"min": 10, "max": 15},
    }
