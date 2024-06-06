import pytest
from unittest.mock import patch, MagicMock
from guidellm.backend import Backend, BackendTypes, GenerativeResponse, OpenAIBackend
from guidellm.core.request import BenchmarkRequest
from guidellm.core.result import BenchmarkResult


@pytest.fixture
def mock_openai_completion_create():
    with patch("openai.Completion.create") as mock_create:
        yield mock_create


@pytest.fixture
def mock_openai_engine_list():
    with patch("openai.Engine.list") as mock_list:
        yield mock_list


@pytest.mark.unit
def test_register_backend():
    assert Backend._registry[BackendTypes.OPENAI_SERVER] == OpenAIBackend


@pytest.mark.unit
def test_create_backend():
    backend = Backend.create_backend(
        BackendTypes.OPENAI_SERVER,
        target="http://localhost:8000/v1",
        api_key="dummy_key",
    )
    assert isinstance(backend, OpenAIBackend)


@pytest.mark.unit
def test_openai_backend_make_request(mock_openai_completion_create):
    mock_openai_completion_create.return_value = [
        {"choices": [{"text": "Token1", "finish_reason": None}]},
        {"choices": [{"text": "Token2", "finish_reason": None}]},
        {"choices": [{"text": "Token1Token2", "finish_reason": "stop"}]},
    ]

    request = BenchmarkRequest(prompt="Test prompt")
    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")

    responses = list(backend.make_request(request))

    assert len(responses) == 3
    assert responses[0].type_ == "token_iter"
    assert responses[0].add_token == "Token1"
    assert responses[1].type_ == "token_iter"
    assert responses[1].add_token == "Token2"
    assert responses[2].type_ == "final"
    assert responses[2].output == "Token1Token2"
    assert responses[2].prompt == "Test prompt"
    assert responses[2].prompt_token_count == 2
    assert responses[2].output_token_count == 1


@pytest.mark.unit
def test_openai_backend_available_models(mock_openai_engine_list):
    mock_openai_engine_list.return_value = {
        "data": [{"id": "text-davinci-002"}, {"id": "text-curie-001"}]
    }

    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")
    models = backend.available_models()
    assert models == ["text-davinci-002", "text-curie-001"]


@pytest.mark.unit
def test_openai_backend_default_model(mock_openai_engine_list):
    mock_openai_engine_list.return_value = {
        "data": [{"id": "text-davinci-002"}, {"id": "text-curie-001"}]
    }

    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")
    default_model = backend.default_model()
    assert default_model == "text-davinci-002"


@pytest.mark.unit
def test_openai_backend_model_tokenizer():
    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")
    tokenizer = backend.model_tokenizer("bert-base-uncased")
    assert tokenizer is not None

    tokenizer = backend.model_tokenizer("non-existent-model")
    assert tokenizer is None


@pytest.mark.unit
def test_openai_backend_token_count():
    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")
    token_count = backend._token_count("This is a test.")
    assert token_count == 4


@pytest.mark.unit
def test_backend_submit(mock_openai_completion_create):
    mock_openai_completion_create.return_value = [
        {"choices": [{"text": "Token1", "finish_reason": None}]},
        {"choices": [{"text": "Token2", "finish_reason": None}]},
        {"choices": [{"text": "Token1Token2", "finish_reason": "stop"}]},
    ]

    request = BenchmarkRequest(prompt="Test prompt")
    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")
    result = backend.submit(request)

    assert result.prompt == "Test prompt"
    assert result.output == "Token1Token2"
    assert result.prompt_token_count == 2
    assert result.output_token_count == 1
    assert result.first_token_time is not None
    assert result.end_time is not None


@pytest.mark.end_to_end
def test_openai_backend_full_workflow(mock_openai_completion_create):
    mock_openai_completion_create.return_value = [
        {"choices": [{"text": "Token1", "finish_reason": None}]},
        {"choices": [{"text": "Token2", "finish_reason": None}]},
        {"choices": [{"text": "Token1Token2", "finish_reason": "stop"}]},
    ]

    request = BenchmarkRequest(prompt="Test prompt")
    backend = OpenAIBackend(target="http://localhost:8000/v1", api_key="dummy_key")
    result = backend.submit(request)

    assert result.prompt == "Test prompt"
    assert result.output == "Token1Token2"
    assert result.prompt_token_count == 2
    assert result.output_token_count == 1
    assert result.first_token_time is not None
    assert result.end_time is not None
    assert result.start_time is not None
