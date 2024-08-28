import re
import time
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from guidellm.core import TextGenerationRequest
from tests.dummy.services import TestRequestGenerator


@pytest.mark.smoke()
def test_request_generator_sync_constructor(mock_auto_tokenizer):
    generator = TestRequestGenerator(mode="sync", tokenizer="mock-tokenizer")
    assert generator.mode == "sync"
    assert generator.async_queue_size == 50  # Default value


@pytest.mark.smoke()
def test_request_generator_async_constructor(mock_auto_tokenizer):
    generator = TestRequestGenerator(
        mode="async", tokenizer="mock-tokenizer", async_queue_size=10
    )
    assert generator.mode == "async"
    assert generator.async_queue_size == 10
    generator.stop()


@pytest.mark.smoke()
def test_request_generator_sync_iter(mock_auto_tokenizer):
    generator = TestRequestGenerator(mode="sync", tokenizer="mock-tokenizer")
    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    assert len(items) == 5
    assert items[0].prompt == "Test prompt"


@pytest.mark.smoke()
def test_request_generator_async_iter(mock_auto_tokenizer):
    generator = TestRequestGenerator(mode="async", tokenizer="mock-tokenizer")
    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    generator.stop()
    assert len(items) == 5
    assert items[0].prompt == "Test prompt"


@pytest.mark.smoke()
def test_request_generator_iter_calls_create_item(mock_auto_tokenizer):
    generator = TestRequestGenerator(mode="sync", tokenizer="mock-tokenizer")
    generator.create_item = Mock(  # type: ignore
        return_value=TextGenerationRequest(prompt="Mock prompt"),
    )

    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    assert len(items) == 5
    generator.create_item.assert_called()


@pytest.mark.smoke()
def test_request_generator_async_iter_calls_create_item(mock_auto_tokenizer):
    generator = TestRequestGenerator(mode="sync", tokenizer="mock-tokenizer")
    generator.create_item = Mock(  # type: ignore
        return_value=TextGenerationRequest(prompt="Mock prompt"),
    )

    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    generator.stop()
    assert len(items) == 5
    generator.create_item.assert_called()


@pytest.mark.sanity()
def test_request_generator_repr(mock_auto_tokenizer):
    generator = TestRequestGenerator(
        mode="sync", tokenizer="mock-tokenizer", async_queue_size=100
    )
    repr_str = repr(generator)
    assert repr_str.startswith("RequestGenerator(")
    assert "mode=sync" in repr_str
    assert "async_queue_size=100" in repr_str
    assert "tokenizer=<MagicMock" in repr_str


@pytest.mark.sanity()
def test_request_generator_stop(mock_auto_tokenizer):
    generator = TestRequestGenerator(mode="async", tokenizer="mock-tokenizer")
    generator.stop()
    assert generator._stop_event.is_set()
    assert not generator._thread.is_alive()


@pytest.mark.regression()
def test_request_generator_with_mock_tokenizer():
    def _fake_tokenize(text: str) -> List[int]:
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [0] * len(tokens)

    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize = MagicMock(side_effect=_fake_tokenize)

    generator = TestRequestGenerator(tokenizer=mock_tokenizer)
    assert generator.tokenizer == mock_tokenizer

    with patch(
        "guidellm.request.base.AutoTokenizer",
    ) as MockAutoTokenizer:  # noqa: N806
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer
        generator = TestRequestGenerator(tokenizer="mock-tokenizer")
        assert generator.tokenizer == mock_tokenizer
        MockAutoTokenizer.from_pretrained.assert_called_with("mock-tokenizer")


@pytest.mark.regression()
def test_request_generator_populate_queue(mock_auto_tokenizer):
    generator = TestRequestGenerator(
        mode="async", tokenizer="mock-tokenizer", async_queue_size=2
    )
    generator.create_item = Mock(  # type: ignore
        return_value=TextGenerationRequest(prompt="Mock prompt")
    )

    time.sleep(0.2)  # Allow some time for the queue to populate
    generator.stop()
    assert generator._queue.qsize() > 0


@pytest.mark.regression()
def test_request_generator_async_stop_during_population(mock_auto_tokenizer):
    generator = TestRequestGenerator(
        mode="async", tokenizer="mock-tokenizer", async_queue_size=2
    )
    generator.create_item = Mock(  # type: ignore
        return_value=TextGenerationRequest(prompt="Mock prompt")
    )

    time.sleep(0.1)  # Allow some time for the queue to start populating
    generator.stop()

    # Ensure the stop event is set and thread is no longer alive
    assert generator._stop_event.is_set()
    assert not generator._thread.is_alive()
