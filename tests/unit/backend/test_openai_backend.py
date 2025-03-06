from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from guidellm.backend import (
    OpenAIHTTPBackend,
)
from guidellm.config import settings


@pytest.fixture()
def mock_httpx_client():
    with patch.object(
        httpx, "AsyncClient", new_callable=AsyncMock
    ) as mock_async_client, patch.object(
        httpx, "Client", new_callable=Mock
    ) as mock_client:
        async_mock_instance = mock_async_client.return_value
        sync_mock_instance = mock_client.return_value

        # Mock synchronous GET response for available models
        sync_mock_instance.get.return_value.json.return_value = {
            "data": [{"id": "test-model"}]
        }

        # Mock asynchronous stream response
        async def mock_stream(*args, **kwargs):
            async def stream_gen():
                for idx in range(3):
                    yield f'data: {{"choices": [{{"text": "token{idx}"}}]}}\n'
                yield "data: [DONE]\n"

            return stream_gen()

        async_mock_instance.stream.side_effect = mock_stream

        yield async_mock_instance, sync_mock_instance


@pytest.mark.smoke()
def test_openai_http_backend_creation():
    backend = OpenAIHTTPBackend(
        target="http://test-target", model="test-model", api_key="test_key"
    )
    assert backend.target == "http://test-target"
    assert backend.model == "test-model"
    assert backend.http2 is True
    assert backend.timeout == settings.request_timeout
    assert backend.authorization.startswith("Bearer")


@pytest.mark.smoke()
def test_openai_http_backend_check_setup(mock_httpx_client):
    async_mock_client, sync_mock_client = mock_httpx_client
    backend = OpenAIHTTPBackend(target="http://test-target")

    with patch.object(httpx, "Client", return_value=sync_mock_client):
        backend.check_setup()

    assert backend.model == "test-model"


@pytest.mark.smoke()
def test_openai_http_backend_available_models(mock_httpx_client):
    async_mock_client, sync_mock_client = mock_httpx_client
    backend = OpenAIHTTPBackend(target="http://test-target")

    with patch.object(httpx, "Client", return_value=sync_mock_client):
        models = backend.available_models()

    assert models == ["test-model"]


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_openai_http_backend_text_completions(mock_httpx_client):
    async_mock_client, _ = mock_httpx_client
    backend = OpenAIHTTPBackend(target="http://test-target")

    with patch.object(httpx, "AsyncClient", return_value=async_mock_client):
        index = 0
        async for response in backend.text_completions("Test Prompt"):
            if index == 0:
                assert response.type_ == "start"
            elif index in [1, 2]:
                assert response.type_ == "iter"
                assert response.delta.startswith("token")
            else:
                assert response.type_ == "final"
            index += 1


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_openai_http_backend_chat_completions(mock_httpx_client):
    async_mock_client, _ = mock_httpx_client
    backend = OpenAIHTTPBackend(target="http://test-target")

    with patch.object(httpx, "AsyncClient", return_value=async_mock_client):
        index = 0
        async for response in backend.chat_completions("Test Chat Content"):
            if index == 0:
                assert response.type_ == "start"
            elif index in [1, 2]:
                assert response.type_ == "iter"
                assert response.delta.startswith("token")
            else:
                assert response.type_ == "final"
            index += 1


@pytest.mark.smoke()
def test_openai_http_backend_headers():
    backend = OpenAIHTTPBackend(
        api_key="test_key", orginization="test_org", project="test_proj"
    )
    headers = backend._headers()
    assert headers["Authorization"] == "Bearer test_key"
    assert headers["OpenAI-Organization"] == "test_org"
    assert headers["OpenAI-Project"] == "test_proj"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.smoke()
def test_openai_http_backend_create_chat_messages():
    messages = OpenAIHTTPBackend._create_chat_messages("Test Message")
    assert messages == [{"role": "user", "content": "Test Message"}]

    messages = OpenAIHTTPBackend._create_chat_messages(["Message 1", "Message 2"])
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Message 1"},
                {"type": "text", "text": "Message 2"},
            ],
        }
    ]


@pytest.mark.smoke()
def test_openai_http_backend_extract_completions_delta_content():
    data = {"choices": [{"text": "Sample Output"}]}
    assert (
        OpenAIHTTPBackend._extract_completions_delta_content("text", data)
        == "Sample Output"
    )

    data = {"choices": [{"delta": {"content": "Chat Output"}}]}
    assert (
        OpenAIHTTPBackend._extract_completions_delta_content("chat", data)
        == "Chat Output"
    )


@pytest.mark.smoke()
def test_openai_http_backend_extract_completions_usage():
    data = {"usage": {"prompt_tokens": 5, "completion_tokens": 10}}
    usage = OpenAIHTTPBackend._extract_completions_usage(data)
    assert usage == {"prompt": 5, "output": 10}
