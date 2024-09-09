from unittest.mock import AsyncMock, Mock, patch

import pytest

from guidellm.backend import Backend, OpenAIBackend
from guidellm.config import reload_settings, settings
from guidellm.core import TextGenerationRequest


@pytest.fixture()
def mock_openai_client():
    with patch("guidellm.backend.openai.AsyncOpenAI") as mock_async_const, patch(
        "guidellm.backend.openai.OpenAI"
    ) as mock_sync_const:
        mock_model = Mock()
        mock_model.id = "mock-model"
        mock_model_2 = Mock()
        mock_model_2.id = "mock-model-2"
        mock_model_data = Mock()
        mock_model_data.data = [mock_model, mock_model_2]

        def create_async_create(inst):
            async def stream():
                for ind in range(3):
                    choice = Mock()
                    choice.delta.content = f"token{ind}" if ind % 2 == 0 else " "
                    choice.finish_reason = None
                    chunk = Mock()
                    chunk.choices = [choice]

                    yield chunk

                choice = Mock()
                choice.finish_reason = "stop"
                chunk = Mock()
                chunk.choices = [choice]
                yield chunk

            async def create(*args, **kwargs):
                inst.create_args = args
                inst.create_kwargs = kwargs
                return stream()

            return create

        def async_constructor(*args, **kwargs):
            mock_async_instance = AsyncMock()
            mock_async_instance.models.list.return_value = mock_model_data
            mock_async_instance.args = args
            mock_async_instance.kwargs = kwargs
            mock_async_instance.chat.completions.create.side_effect = (
                create_async_create(mock_async_instance)
            )

            return mock_async_instance

        def sync_constructor(*args, **kwargs):
            mock_sync_instance = Mock()
            mock_sync_instance.models.list.return_value = mock_model_data
            mock_sync_instance.args = args
            mock_sync_instance.kwargs = kwargs
            return mock_sync_instance

        mock_async_const.side_effect = async_constructor
        mock_sync_const.side_effect = sync_constructor
        yield mock_async_const, mock_sync_const


@pytest.mark.smoke()
@pytest.mark.parametrize(
    (
        "openai_api_key",
        "target",
        "model",
        "request_args",
        "expected_base_url",
    ),
    [
        (
            "test_key",
            "http://test-target",
            "test-model",
            {"arg1": "value1"},
            "http://test-target",
        ),
        (None, None, None, {}, settings.openai.base_url),
    ],
)
def test_openai_backend_create(
    openai_api_key,
    target,
    model,
    request_args,
    expected_base_url,
    mock_openai_client,
):
    backends = [
        Backend.create(
            "openai_server",
            openai_api_key=openai_api_key,
            target=target,
            model=model,
            **request_args,
        ),
        OpenAIBackend(
            openai_api_key=openai_api_key,
            target=target,
            model=model,
            **request_args,
        ),
    ]

    for backend in backends:
        assert backend._async_client.kwargs["api_key"] == (  # type: ignore
            openai_api_key or settings.openai.api_key
        )
        assert backend._async_client.kwargs["base_url"] == expected_base_url  # type: ignore
        assert backend._client.kwargs["api_key"] == (  # type: ignore
            openai_api_key or settings.openai.api_key
        )
        assert backend._client.kwargs["base_url"] == expected_base_url  # type: ignore
        if model:
            assert backend._model == model  # type: ignore


@pytest.mark.smoke()
def test_openai_backend_models(mock_openai_client):
    backend = OpenAIBackend()
    assert backend.available_models() == ["mock-model", "mock-model-2"]
    assert backend.default_model == "mock-model"
    assert backend.model == "mock-model"


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("req", "request_args"),
    [
        (TextGenerationRequest(prompt="Test"), None),
        (
            TextGenerationRequest(prompt="Test", params={"generated_tokens": 10}),
            None,
        ),
        (
            TextGenerationRequest(prompt="Test", params={"generated_tokens": 10}),
            {"max_tokens": 10},
        ),
        (
            TextGenerationRequest(prompt="Test"),
            {"max_tokens": 10, "stop": "stop"},
        ),
    ],
)
@pytest.mark.asyncio()
async def test_openai_backend_make_request(req, request_args, mock_openai_client):
    backend = OpenAIBackend(**(request_args or {}))
    counter = 0

    async for response in backend.make_request(req):
        if counter < 3:
            assert response.type_ == "token_iter"
            assert response.add_token == f"token{counter}" if counter % 2 == 0 else " "
        elif counter == 3:
            assert response.type_ == "final"
        else:
            raise ValueError("Too many responses received from the backend")

        counter += 1

    # check the kwargs passed to the openai client
    # now that the generator has been consumed
    assert backend._async_client.create_args == ()  # type: ignore
    assert backend._async_client.create_kwargs["model"] == "mock-model"  # type: ignore
    assert backend._async_client.create_kwargs["messages"] == [  # type: ignore
        {"role": "user", "content": req.prompt}
    ]
    assert backend._async_client.create_kwargs["stream"]  # type: ignore
    assert backend._async_client.create_kwargs["n"] == 1  # type: ignore

    if req.output_token_count is not None:
        assert (
            backend._async_client.create_kwargs["max_tokens"] == req.output_token_count  # type: ignore
        )
        assert backend._async_client.create_kwargs["stop"] is None  # type: ignore
    elif request_args is not None and "max_tokens" not in request_args:
        assert (
            backend._async_client.create_kwargs["max_tokens"]  # type: ignore
            == settings.openai.max_gen_tokens
        )

    if request_args:
        for key, value in request_args.items():
            assert backend._async_client.create_kwargs[key] == value  # type: ignore


@pytest.mark.sanity()
@pytest.mark.asyncio()
async def test_openai_backend_submit(mock_openai_client):
    backend = OpenAIBackend()
    request = TextGenerationRequest(prompt="Test", prompt_token_count=1)
    result = await backend.submit(request)

    assert result.request == request
    assert result.prompt == request.prompt
    assert result.prompt_token_count == 1
    assert result.output == "token0 token2"
    assert result.output_token_count == 3
    assert result.last_time is not None
    assert result.first_token_set
    assert result.start_time is not None
    assert result.first_token_time is not None
    assert result.end_time is not None
    assert len(result.decode_times) == 2


@pytest.mark.sanity()
def test_openai_backend_api_key(mock_openai_client):
    backend = OpenAIBackend()
    assert backend._async_client.kwargs["api_key"] == settings.openai.api_key  # type: ignore
    assert backend._client.kwargs["api_key"] == settings.openai.api_key  # type: ignore

    backend = OpenAIBackend(openai_api_key="test_key")
    assert backend._async_client.kwargs["api_key"] == "test_key"  # type: ignore
    assert backend._client.kwargs["api_key"] == "test_key"  # type: ignore


@pytest.mark.sanity()
def test_openai_backend_api_key_env(mock_openai_client, mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__OPENAI__API_KEY": "test_key",
        },
    )
    reload_settings()

    backend = OpenAIBackend()
    assert backend._async_client.kwargs["api_key"] == "test_key"  # type: ignore
    assert backend._client.kwargs["api_key"] == "test_key"  # type: ignore


@pytest.mark.sanity()
def test_openai_backend_target(mock_openai_client):
    backend = OpenAIBackend(target="http://test-target")
    assert backend._async_client.kwargs["base_url"] == "http://test-target"  # type: ignore
    assert backend._client.kwargs["base_url"] == "http://test-target"  # type: ignore

    backend = OpenAIBackend()
    assert backend._async_client.kwargs["base_url"] == "http://localhost:8000/v1"  # type: ignore
    assert backend._client.kwargs["base_url"] == "http://localhost:8000/v1"  # type: ignore

    backend = OpenAIBackend()
    assert backend._async_client.kwargs["base_url"] == settings.openai.base_url  # type: ignore
    assert backend._client.kwargs["base_url"] == settings.openai.base_url  # type: ignore


@pytest.mark.sanity()
def test_openai_backend_target_env(mock_openai_client, mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__OPENAI__BASE_URL": "http://test-target",
        },
    )
    reload_settings()

    backend = OpenAIBackend()
    assert backend._async_client.kwargs["base_url"] == "http://test-target"  # type: ignore
    assert backend._client.kwargs["base_url"] == "http://test-target"  # type: ignore


@pytest.mark.regression()
def test_openai_backend_target_none_error(mock_openai_client, mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__OPENAI__BASE_URL": "",
        },
    )
    reload_settings()

    with pytest.raises(ValueError):
        OpenAIBackend(target=None)
