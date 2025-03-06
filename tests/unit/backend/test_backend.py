from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import pytest
from PIL import Image
from pydantic import ValidationError

from guidellm.backend import (
    Backend,
    BackendType,
    StreamingRequestArgs,
    StreamingResponse,
    StreamingResponseTimings,
    StreamingTextResponseStats,
)


@pytest.mark.smoke()
def test_streaming_request_args_creation():
    args = StreamingRequestArgs(
        target="http://example.com",
        headers={"Authorization": "Bearer token"},
        payload={"query": "Hello, world!"},
        timeout=10.0,
        http2=True,
    )
    assert args.target == "http://example.com"
    assert args.headers["Authorization"] == "Bearer token"
    assert args.payload["query"] == "Hello, world!"
    assert args.timeout == 10.0
    assert args.http2 is True

    with pytest.raises(ValidationError):
        StreamingRequestArgs(target="http://example.com", headers={}, payload=None)  # type: ignore


@pytest.mark.smoke()
def test_streaming_response_timings_creation():
    timings_def = StreamingResponseTimings()
    assert timings_def.request_start is None
    assert timings_def.values == []
    assert timings_def.request_end is None
    assert timings_def.delta is None

    timings = StreamingResponseTimings(
        request_start=1.0, values=[1.2, 1.4, 1.6], request_end=2.0, delta=0.2
    )
    assert timings.request_start == 1.0
    assert timings.values == [1.2, 1.4, 1.6]
    assert timings.request_end == 2.0
    assert timings.delta == 0.2


@pytest.mark.smoke()
def test_streaming_text_response_stats_creation():
    stats_def = StreamingTextResponseStats()
    assert stats_def.request_prompt_tokens is None
    assert stats_def.request_output_tokens is None
    assert stats_def.response_prompt_tokens is None
    assert stats_def.response_output_tokens is None
    assert stats_def.response_stream_iterations == 0

    stats = StreamingTextResponseStats(
        request_prompt_tokens=5,
        request_output_tokens=10,
        response_prompt_tokens=5,
        response_output_tokens=10,
        response_stream_iterations=3,
    )
    assert stats.request_prompt_tokens == 5
    assert stats.request_output_tokens == 10
    assert stats.response_prompt_tokens == 5
    assert stats.response_output_tokens == 10
    assert stats.response_stream_iterations == 3

    assert stats.prompt_tokens_count == 5
    assert stats.output_tokens_count == 10


@pytest.mark.smoke()
def test_streaming_response_creation():
    request_args = StreamingRequestArgs(
        target="http://example.com", headers={}, payload={}
    )
    response = StreamingResponse(
        type_="start",
        request_args=request_args,
    )
    assert response.type_ == "start"
    assert response.request_args.target == "http://example.com"
    assert isinstance(response.timings, StreamingResponseTimings)
    assert isinstance(response.stats, StreamingTextResponseStats)
    assert response.delta is None
    assert response.content == ""


class MockBackend(Backend):
    def __init__(self, type_: BackendType = "mock"):  # type: ignore
        super().__init__(type_)

    def check_setup(self):
        pass

    def available_models(self) -> List[str]:
        return ["mock-model"]

    async def text_completions(
        self,
        prompt: Union[str, List[str]],
        id_: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamingResponse, None]:
        yield StreamingResponse(
            type_="start",
            request_args=StreamingRequestArgs(target="", headers={}, payload={}),
        )

        yield StreamingResponse(
            type_="iter",
            request_args=StreamingRequestArgs(target="", headers={}, payload={}),
        )

        yield StreamingResponse(
            type_="final",
            request_args=StreamingRequestArgs(target="", headers={}, payload={}),
        )

    async def chat_completions(
        self,
        content: Union[
            str,
            List[Union[str, Dict[str, Union[str, Dict[str, str]]], Path, Image.Image]],
            Any,
        ],
        id_: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        raw_content: bool = False,
        **kwargs,
    ) -> AsyncGenerator[StreamingResponse, None]:
        yield StreamingResponse(
            type_="start",
            request_args=StreamingRequestArgs(target="", headers={}, payload={}),
        )

        yield StreamingResponse(
            type_="iter",
            request_args=StreamingRequestArgs(target="", headers={}, payload={}),
        )

        yield StreamingResponse(
            type_="final",
            request_args=StreamingRequestArgs(target="", headers={}, payload={}),
        )


@pytest.mark.smoke()
def test_backend_registry():
    Backend.register("mock")(MockBackend)  # type: ignore
    assert Backend._registry["mock"] is MockBackend  # type: ignore

    backend_instance = Backend.create("mock")  # type: ignore
    assert isinstance(backend_instance, MockBackend)

    with pytest.raises(ValueError):
        Backend.create("invalid_type")  # type: ignore


@pytest.mark.smoke()
def test_backend_instantiation():
    backend = MockBackend()
    assert backend.type_ == "mock"


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_text_completions():
    backend = MockBackend()
    index = 0

    async for response in backend.text_completions("Test"):
        if index == 0:
            assert response.type_ == "start"
        elif index == 1:
            assert response.type_ == "iter"
        else:
            assert response.type_ == "final"
        index += 1


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_chat_completions():
    backend = MockBackend()
    index = 0

    async for response in backend.chat_completions("Test"):
        if index == 0:
            assert response.type_ == "start"
        elif index == 1:
            assert response.type_ == "iter"
        else:
            assert response.type_ == "final"
        index += 1


@pytest.mark.smoke()
def test_backend_models():
    backend = MockBackend()
    assert backend.available_models() == ["mock-model"]


@pytest.mark.smoke()
def test_backend_validate():
    backend = MockBackend()
    backend.validate()
