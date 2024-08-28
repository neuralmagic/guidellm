from typing import Any, Dict, Generator, Optional, Type

import pytest
from pydantic import BaseModel

from guidellm.backend import Backend, DeepsparseBackend
from guidellm.config import reload_settings
from guidellm.core import TextGenerationRequest
from guidellm.utils import random_strings


class TestDeepsparseTextGeneration(BaseModel):
    """The representation of a deepsparse data structure."""

    text: str


class TestTextGenerationPipeline:
    """Deepsparse TextGeneration test interface.

    By default this class generates '10' text responses.

    This class includes an additional development information
    for better testing experience.

    Method `__call__` allows to mock the result object that comes from
    `deepsparse.pipeline.Pipeline()` so everything is encapsulated right here.

    :param self._generation: dynamic representation of generated responses
        from deepsparse interface.
    """

    def __init__(self):
        self._generations: list[TestDeepsparseTextGeneration] = []
        self._prompt: Optional[str] = None
        self._max_new_tokens: Optional[int] = None

    def __call__(
        self, *_, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> Any:
        """Mocks the result from `deepsparse.pipeline.Pipeline()()`.
        Set reserved request arguments on call
        """

        self._prompt = prompt
        self._max_new_tokens = max_new_tokens

        return self

    @property
    def generations(self) -> Generator[TestDeepsparseTextGeneration, None, None]:
        for text in random_strings(
            min=10, max=50, n=self._max_new_tokens if self._max_new_tokens else 10
        ):
            generation = TestDeepsparseTextGeneration(text=text)
            self._generations.append(generation)
            yield generation


@pytest.fixture(autouse=True)
def mock_deepsparse_pipeline(mocker):
    return mocker.patch(
        "deepsparse.Pipeline.create",
        return_value=TestTextGenerationPipeline(),
    )


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "create_payload",
    [
        {},
        {"model": "test/custom_llm"},
    ],
)
def test_backend_creation(create_payload: Dict):
    """Test the "Deepspaarse Backend" class
    with defaults and custom input parameters.
    """

    backends: list[DeepsparseBackend] = [
        Backend.create("deepsparse", **create_payload),
        DeepsparseBackend(**create_payload),
    ]

    for backend in backends:
        assert getattr(backend, "pipeline")
        (
            getattr(backend, "model") == custom_model
            if (custom_model := create_payload.get("model"))
            else getattr(backend, "default_model")
        )


@pytest.mark.smoke()
def test_backend_model_from_env(mocker):
    mocker.patch.dict(
        "os.environ",
        {"GUIDELLM__DEEPSPRASE__MODEL": "test_backend_model_from_env"},
    )

    reload_settings()

    backends: list[DeepsparseBackend] = [
        Backend.create("deepsparse"),
        DeepsparseBackend(),
    ]

    for backend in backends:
        assert getattr(backend, "model") == "test_backend_model_from_env"


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "text_generation_request_create_payload",
    [
        {"prompt": "Test prompt"},
        {"prompt": "Test prompt", "output_token_count": 20},
    ],
)
@pytest.mark.asyncio()
async def test_make_request(text_generation_request_create_payload: Dict):
    backend = DeepsparseBackend()

    output_tokens: list[str] = []
    async for response in backend.make_request(
        request=TextGenerationRequest(**text_generation_request_create_payload)
    ):
        if response.add_token:
            output_tokens.append(response.add_token)
    assert "".join(output_tokens) == "".join(
        (generation.text for generation in backend.pipeline._generations)
    )

    if max_tokens := text_generation_request_create_payload.get("output_token_count"):
        assert len(backend.pipeline._generations) == max_tokens


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "text_generation_request_create_payload,error",
    [
        ({"prompt": "Test prompt", "output_token_count": -1}, ValueError),
    ],
)
@pytest.mark.asyncio()
async def test_make_request_invalid_request_payload(
    text_generation_request_create_payload: Dict, error: Type[Exception]
):
    backend = DeepsparseBackend()
    with pytest.raises(error):
        async for _ in backend.make_request(
            request=TextGenerationRequest(**text_generation_request_create_payload)
        ):
            pass
