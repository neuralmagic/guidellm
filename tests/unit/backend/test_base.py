import pytest

from guidellm.backend import Backend, GenerativeResponse
from guidellm.core import TextGenerationRequest, TextGenerationResult


@pytest.mark.smoke()
def test_backend_registry():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            yield GenerativeResponse(type_="final", output="Test")

        def available_models(self):
            return ["mock-model"]

    backend_type = "test"
    Backend.register(backend_type)(MockBackend)  # type: ignore
    assert Backend._registry[backend_type] is MockBackend  # type: ignore

    backend_instance = Backend.create(backend_type)  # type: ignore
    assert isinstance(backend_instance, MockBackend)

    with pytest.raises(ValueError):
        Backend.create("invalid_type")  # type: ignore


@pytest.mark.smoke()
def test_generative_response_creation():
    response = GenerativeResponse(type_="final", output="Test Output")
    assert response.type_ == "final"
    assert response.output == "Test Output"
    assert response.add_token is None
    assert response.prompt is None

    response = GenerativeResponse(type_="token_iter", add_token="token")
    assert response.type_ == "token_iter"
    assert response.add_token == "token"
    assert response.output is None


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_make_request():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            yield GenerativeResponse(
                type_="token_iter",
                add_token="Token",
                prompt="Hello, world!",
                prompt_token_count=5,
            )
            yield GenerativeResponse(
                type_="final",
                output="This is a final response.",
                prompt="Hello, world!",
                prompt_token_count=5,
                output_token_count=10,
            )

        def available_models(self):
            return ["mock-model"]

    backend = MockBackend()
    index = 0

    async for response in backend.make_request(TextGenerationRequest(prompt="Test")):
        if index == 0:
            assert response.type_ == "token_iter"
            assert response.add_token == "Token"
            assert response.prompt == "Hello, world!"
            assert response.prompt_token_count == 5
        else:
            assert response.type_ == "final"
            assert response.output == "This is a final response."
            assert response.prompt == "Hello, world!"
            assert response.prompt_token_count == 5
            assert response.output_token_count == 10
        index += 1


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_submit_final():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            yield GenerativeResponse(type_="final", output="Test")

        def available_models(self):
            return ["mock-model"]

    backend = MockBackend()
    result = await backend.submit(TextGenerationRequest(prompt="Test"))
    assert isinstance(result, TextGenerationResult)
    assert result.output == "Test"


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_submit_multi():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            yield GenerativeResponse(type_="token_iter", add_token="Token")
            yield GenerativeResponse(type_="token_iter", add_token=" ")
            yield GenerativeResponse(type_="token_iter", add_token="Test")
            yield GenerativeResponse(type_="final")

        def available_models(self):
            return ["mock-model"]

    backend = MockBackend()
    result = await backend.submit(TextGenerationRequest(prompt="Test"))
    assert isinstance(result, TextGenerationResult)
    assert result.output == "Token Test"


@pytest.mark.regression()
@pytest.mark.asyncio()
async def test_backend_submit_no_response():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            if False:  # simulate no yield
                yield

        def available_models(self):
            return ["mock-model"]

    backend = MockBackend()

    with pytest.raises(ValueError):
        await backend.submit(TextGenerationRequest(prompt="Test"))


@pytest.mark.smoke()
@pytest.mark.asyncio()
async def test_backend_submit_multi_final():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            yield GenerativeResponse(type_="token_iter", add_token="Token")
            yield GenerativeResponse(type_="token_iter", add_token=" ")
            yield GenerativeResponse(type_="token_iter", add_token="Test")
            yield GenerativeResponse(type_="final")
            yield GenerativeResponse(type_="final")

        def available_models(self):
            return ["mock-model"]

    backend = MockBackend()

    with pytest.raises(ValueError):
        await backend.submit(TextGenerationRequest(prompt="Test"))


@pytest.mark.smoke()
def test_backend_models():
    class MockBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        def available_models(self):
            return ["mock-model", "mock-model-2"]

        async def make_request(self, request):
            yield GenerativeResponse(type_="final", output="")

    backend = MockBackend()
    assert backend.available_models() == ["mock-model", "mock-model-2"]
    assert backend.default_model == "mock-model"


@pytest.mark.regression()
def test_backend_abstract_methods():
    with pytest.raises(TypeError):
        Backend()  # type: ignore

    class IncompleteBackend(Backend):
        def __init__(self):
            super().__init__("test", "http://localhost:8000", "mock-model")

        async def make_request(self, request):
            yield GenerativeResponse(type_="final", output="Test")

    with pytest.raises(TypeError):
        IncompleteBackend()  # type: ignore
