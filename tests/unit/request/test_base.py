import time
from unittest.mock import Mock, patch

import pytest

from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import RequestGenerator


class TestRequestGenerator(RequestGenerator):
    def create_item(self) -> TextGenerationRequest:
        return TextGenerationRequest(prompt="Test prompt")


@pytest.mark.smoke
def test_request_generator_sync_constructor():
    generator = TestRequestGenerator(mode="sync")
    assert generator.mode == "sync"
    assert generator.async_queue_size == 50  # Default value
    assert generator.tokenizer is None


@pytest.mark.smoke
def test_request_generator_async_constructor():
    generator = TestRequestGenerator(mode="async", async_queue_size=10)
    assert generator.mode == "async"
    assert generator.async_queue_size == 10
    assert generator.tokenizer is None
    generator.stop()


@pytest.mark.smoke
def test_request_generator_sync_iter():
    generator = TestRequestGenerator(mode="sync")
    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    assert len(items) == 5
    assert items[0].prompt == "Test prompt"


@pytest.mark.smoke
def test_request_generator_async_iter():
    generator = TestRequestGenerator(mode="async")
    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    generator.stop()
    assert len(items) == 5
    assert items[0].prompt == "Test prompt"


@pytest.mark.regression
def test_request_generator_with_mock_tokenizer():
    mock_tokenizer = Mock()
    generator = TestRequestGenerator(tokenizer=mock_tokenizer)
    assert generator.tokenizer == mock_tokenizer

    with patch("guidellm.request.base.AutoTokenizer") as MockAutoTokenizer:
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer
        generator = TestRequestGenerator(tokenizer="mock-tokenizer")
        assert generator.tokenizer == mock_tokenizer
        MockAutoTokenizer.from_pretrained.assert_called_with("mock-tokenizer")


@pytest.mark.regression
def test_request_generator_repr():
    generator = TestRequestGenerator(mode="sync", async_queue_size=100)
    assert repr(generator) == (
        "RequestGenerator(mode=sync, async_queue_size=100, tokenizer=None)"
    )


@pytest.mark.regression
def test_request_generator_create_item_not_implemented():
    with pytest.raises(TypeError):

        class IncompleteRequestGenerator(RequestGenerator):
            pass

        IncompleteRequestGenerator()

    class IncompleteCreateItemGenerator(RequestGenerator):
        def create_item(self):
            super().create_item()

    generator = IncompleteCreateItemGenerator()
    with pytest.raises(NotImplementedError):
        generator.create_item()


@pytest.mark.regression
def test_request_generator_iter_calls_create_item():
    generator = TestRequestGenerator(mode="sync")
    generator.create_item = Mock(
        return_value=TextGenerationRequest(prompt="Mock prompt")
    )

    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    assert generator._queue.qsize() == 0
    generator.create_item.assert_called()


@pytest.mark.regression
def test_request_generator_async_iter_calls_create_item():
    generator = TestRequestGenerator(mode="sync")
    generator.create_item = Mock(
        return_value=TextGenerationRequest(prompt="Mock prompt")
    )

    items = []
    for item in generator:
        items.append(item)
        if len(items) == 5:
            break

    generator.stop()
    stop_size = generator._queue.qsize()
    time.sleep(0.1)
    assert generator._queue.qsize() == stop_size
    generator.create_item.assert_called()
