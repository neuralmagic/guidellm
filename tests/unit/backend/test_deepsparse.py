"""
This module includes unit tests for the Deepsparse backend.

Notes: tests from this module are going to be skipped in case
    the Python version is >= 3.12 according to the deepsparse limitation.
"""

import sys
from typing import Any, Dict, Generator, List, Optional

import pytest
from pydantic import BaseModel

from guidellm.backend import Backend
from guidellm.config import reload_settings
from guidellm.core import TextGenerationRequest
from guidellm.utils import random_strings

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 12), reason="Unsupported Python version"
)


@pytest.fixture(scope="module")
def backend_class():
    from guidellm.backend.deepsparse import DeepsparseBackend

    return DeepsparseBackend


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
        self._generations: List[TestDeepsparseTextGeneration] = []
        self._prompt: Optional[str] = None
        self._max_new_tokens: Optional[int] = None

    def __call__(
        self, *_, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> Any:
        """Mocks the result from `deepsparse.pipeline.Pipeline()()`.
        Set reserved request arguments on call.

        Note: `**kwargs` is required since it allows to mimic
            the `deepsparse.Pipeline` behavior.
        """

        self._prompt = prompt
        self._max_new_tokens = max_new_tokens

        return self

    @property
    def generations(self) -> Generator[TestDeepsparseTextGeneration, None, None]:
        for text in random_strings(
            min_chars=10,
            max_chars=50,
            n=self._max_new_tokens if self._max_new_tokens else 10,
        ):
            generation = TestDeepsparseTextGeneration(text=text)
            self._generations.append(generation)
            yield generation


@pytest.fixture(autouse=True)
def mock_deepsparse_pipeline(mocker):
    return mocker.patch(
        "deepsparse.Pipeline.create", return_value=TestTextGenerationPipeline()
    )


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "create_payload",
    [
        {},
        {"model": "test/custom_llm"},
    ],
)
def test_backend_creation(create_payload: Dict, backend_class):
    """Test the "Deepspaarse Backend" class
    with defaults and custom input parameters.
    """

    backends = [
        Backend.create("deepsparse", **create_payload),
        backend_class(**create_payload),
    ]

    for backend in backends:
        assert backend.pipeline
        (
            backend.model == custom_model
            if (custom_model := create_payload.get("model"))
            else backend.default_model
        )


@pytest.mark.smoke()
def test_backend_model_from_env(mocker, backend_class):
    mocker.patch.dict(
        "os.environ",
        {"GUIDELLM__LLM_MODEL": "test_backend_model_from_env"},
    )

    reload_settings()

    backends = [Backend.create("deepsparse"), backend_class()]

    for backend in backends:
        assert backend.model == "test_backend_model_from_env"


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "text_generation_request_create_payload",
    [
        {"prompt": "Test prompt"},
        {"prompt": "Test prompt", "output_token_count": 20},
    ],
)
@pytest.mark.asyncio()
async def test_make_request(
    text_generation_request_create_payload: Dict, backend_class
):
    backend = backend_class()

    output_tokens: List[str] = []
    async for response in backend.make_request(
        request=TextGenerationRequest(**text_generation_request_create_payload)
    ):
        if response.add_token:
            output_tokens.append(response.add_token)
    assert "".join(output_tokens) == "".join(
        generation.text for generation in backend.pipeline._generations
    )

    if max_tokens := text_generation_request_create_payload.get("output_token_count"):
        assert len(backend.pipeline._generations) == max_tokens


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("text_generation_request_create_payload", "error"),
    [
        (
            {"prompt": "Test prompt", "output_token_count": -1},
            ValueError,
        ),
    ],
)
@pytest.mark.asyncio()
async def test_make_request_invalid_request_payload(
    text_generation_request_create_payload: Dict, error, backend_class
):
    backend = backend_class()
    with pytest.raises(error):
        [
            respnose
            async for respnose in backend.make_request(
                request=TextGenerationRequest(**text_generation_request_create_payload)
            )
        ]