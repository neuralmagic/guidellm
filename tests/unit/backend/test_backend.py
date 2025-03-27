import time

import pytest

from guidellm.backend import (
    Backend,
    ResponseSummary,
    StreamingTextResponse,
)


@pytest.mark.smoke()
def test_backend_registry():
    assert Backend._registry["mock"] is not None  # type: ignore

    backend_instance = Backend.create("mock")  # type: ignore
    assert backend_instance is not None

    with pytest.raises(ValueError):
        Backend.register("mock")("backend")  # type: ignore

    with pytest.raises(ValueError):
        Backend.create("invalid_type")  # type: ignore


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_text_completions(mock_backend):
    index = 0
    prompt = "Test Prompt"
    request_id = "test-request-id"
    prompt_token_count = 3
    output_token_count = 10
    final_resp = None

    async for response in mock_backend.text_completions(
        prompt=prompt,
        request_id=request_id,
        prompt_token_count=prompt_token_count,
        output_token_count=output_token_count,
    ):
        assert isinstance(response, (StreamingTextResponse, ResponseSummary))

        if index == 0:
            assert isinstance(response, StreamingTextResponse)
            assert response.type_ == "start"
            assert response.iter_count == 0
            assert response.delta == ""
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == request_id
        elif not isinstance(response, ResponseSummary):
            assert response.type_ == "iter"
            assert response.iter_count == index
            assert len(response.delta) > 0
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == request_id
        else:
            assert not final_resp
            final_resp = response
            assert isinstance(response, ResponseSummary)
            assert len(response.value) > 0
            assert response.iterations > 0
            assert response.start_time > 0
            assert response.end_time == pytest.approx(time.time(), abs=0.01)
            assert response.request_prompt_tokens == prompt_token_count
            assert response.request_output_tokens == output_token_count
            assert response.response_prompt_tokens == 3
            assert response.response_output_tokens == 10
            assert response.request_id == request_id

        index += 1

    assert final_resp


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_chat_completions(mock_backend):
    index = 0
    prompt = "Test Prompt"
    request_id = "test-request-id"
    prompt_token_count = 3
    output_token_count = 10
    final_resp = None

    async for response in mock_backend.chat_completions(
        content=prompt,
        request_id=request_id,
        prompt_token_count=prompt_token_count,
        output_token_count=output_token_count,
    ):
        assert isinstance(response, (StreamingTextResponse, ResponseSummary))

        if index == 0:
            assert isinstance(response, StreamingTextResponse)
            assert response.type_ == "start"
            assert response.iter_count == 0
            assert response.delta == ""
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == request_id
        elif not isinstance(response, ResponseSummary):
            assert response.type_ == "iter"
            assert response.iter_count == index
            assert len(response.delta) > 0
            assert response.time == pytest.approx(time.time(), abs=0.01)
            assert response.request_id == request_id
        else:
            assert not final_resp
            final_resp = response
            assert isinstance(response, ResponseSummary)
            assert len(response.value) > 0
            assert response.iterations > 0
            assert response.start_time > 0
            assert response.end_time == pytest.approx(time.time(), abs=0.01)
            assert response.request_prompt_tokens == prompt_token_count
            assert response.request_output_tokens == output_token_count
            assert response.response_prompt_tokens == 3
            assert response.response_output_tokens == 10
            assert response.request_id == request_id

        index += 1

    assert final_resp


@pytest.mark.smoke()
def test_backend_models(mock_backend):
    assert mock_backend.available_models() == ["mock-model"]


@pytest.mark.smoke()
def test_backend_validate(mock_backend):
    mock_backend.validate()
