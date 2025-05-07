from typing import get_args

import pytest

from guidellm.backend import (
    RequestArgs,
    ResponseSummary,
    StreamingResponseType,
    StreamingTextResponse,
)


@pytest.mark.smoke
def test_streaming_response_types():
    valid_types = get_args(StreamingResponseType)
    assert valid_types == ("start", "iter")


@pytest.mark.smoke
def test_streaming_text_response_default_initilization():
    response = StreamingTextResponse(
        type_="start",
        value="",
        start_time=0.0,
        first_iter_time=None,
        iter_count=0,
        delta="",
        time=0.0,
    )
    assert response.request_id is None


@pytest.mark.smoke
def test_streaming_text_response_initialization():
    response = StreamingTextResponse(
        type_="start",
        value="Hello, world!",
        start_time=0.0,
        first_iter_time=0.0,
        iter_count=1,
        delta="Hello, world!",
        time=1.0,
        request_id="123",
    )
    assert response.type_ == "start"
    assert response.value == "Hello, world!"
    assert response.start_time == 0.0
    assert response.first_iter_time == 0.0
    assert response.iter_count == 1
    assert response.delta == "Hello, world!"
    assert response.time == 1.0
    assert response.request_id == "123"


@pytest.mark.smoke
def test_streaming_text_response_marshalling():
    response = StreamingTextResponse(
        type_="start",
        value="Hello, world!",
        start_time=0.0,
        first_iter_time=0.0,
        iter_count=0,
        delta="Hello, world!",
        time=1.0,
        request_id="123",
    )
    serialized = response.model_dump()
    deserialized = StreamingTextResponse.model_validate(serialized)

    for key, value in vars(response).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke
def test_request_args_default_initialization():
    args = RequestArgs(
        target="http://example.com",
        headers={},
        payload={},
    )
    assert args.timeout is None
    assert args.http2 is None
    assert args.follow_redirects is None


@pytest.mark.smoke
def test_request_args_initialization():
    args = RequestArgs(
        target="http://example.com",
        headers={
            "Authorization": "Bearer token",
        },
        payload={
            "query": "Hello, world!",
        },
        timeout=10.0,
        http2=True,
        follow_redirects=True,
    )
    assert args.target == "http://example.com"
    assert args.headers == {"Authorization": "Bearer token"}
    assert args.payload == {"query": "Hello, world!"}
    assert args.timeout == 10.0
    assert args.http2 is True
    assert args.follow_redirects is True


@pytest.mark.smoke
def test_response_args_marshalling():
    args = RequestArgs(
        target="http://example.com",
        headers={"Authorization": "Bearer token"},
        payload={"query": "Hello, world!"},
        timeout=10.0,
        http2=True,
    )
    serialized = args.model_dump()
    deserialized = RequestArgs.model_validate(serialized)

    for key, value in vars(args).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke
def test_response_summary_default_initialization():
    summary = ResponseSummary(
        value="Hello, world!",
        request_args=RequestArgs(
            target="http://example.com",
            headers={},
            payload={},
        ),
        start_time=0.0,
        end_time=0.0,
        first_iter_time=None,
        last_iter_time=None,
    )
    assert summary.value == "Hello, world!"
    assert summary.request_args.target == "http://example.com"
    assert summary.request_args.headers == {}
    assert summary.request_args.payload == {}
    assert summary.start_time == 0.0
    assert summary.end_time == 0.0
    assert summary.first_iter_time is None
    assert summary.last_iter_time is None
    assert summary.iterations == 0
    assert summary.request_prompt_tokens is None
    assert summary.request_output_tokens is None
    assert summary.response_prompt_tokens is None
    assert summary.response_output_tokens is None
    assert summary.request_id is None


@pytest.mark.smoke
def test_response_summary_initialization():
    summary = ResponseSummary(
        value="Hello, world!",
        request_args=RequestArgs(
            target="http://example.com",
            headers={},
            payload={},
        ),
        start_time=1.0,
        end_time=2.0,
        iterations=3,
        first_iter_time=1.0,
        last_iter_time=2.0,
        request_prompt_tokens=5,
        request_output_tokens=10,
        response_prompt_tokens=5,
        response_output_tokens=10,
        request_id="123",
    )
    assert summary.value == "Hello, world!"
    assert summary.request_args.target == "http://example.com"
    assert summary.request_args.headers == {}
    assert summary.request_args.payload == {}
    assert summary.start_time == 1.0
    assert summary.end_time == 2.0
    assert summary.iterations == 3
    assert summary.first_iter_time == 1.0
    assert summary.last_iter_time == 2.0
    assert summary.request_prompt_tokens == 5
    assert summary.request_output_tokens == 10
    assert summary.response_prompt_tokens == 5
    assert summary.response_output_tokens == 10
    assert summary.request_id == "123"
