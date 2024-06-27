import asyncio
from collections.abc import AsyncGenerator

import pytest

from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import RequestGenerator


class TestRequestGenerator(RequestGenerator):
    def create_item(self) -> TextGenerationRequest:
        return TextGenerationRequest(prompt="Test prompt")


@pytest.mark.smoke
def test_request_generator_sync():
    generator = TestRequestGenerator(mode="sync")
    assert generator.mode == "sync"
    assert generator.tokenizer is None

    items = []
    for item in generator:
        items.append(item)

        if len(items) == 5:
            break

    assert items[0].prompt == "Test prompt"


@pytest.mark.smoke
async def test_request_generator_async():
    generator = TestRequestGenerator(mode="async", async_queue_size=10)

    assert generator.mode == "async"
    assert generator.async_queue_size == 10
    assert generator.tokenizer is None

    items = []
    try:
        async for item in generator:
            items.append(item)

            if len(items) == 5:
                break
    finally:
        generator.stop()

    assert generator._stop_event.is_set()
    for item in items:
        assert item.prompt == "Test prompt"


@pytest.mark.regression
def test_request_generator_repr():
    generator = TestRequestGenerator(mode="sync", async_queue_size=100)
    assert repr(generator) == (
        "RequestGenerator(mode=sync, async_queue_size=100, tokenizer=None)"
    )
