"""
This module includes unit tests for the OpenAI Backend Service.
"""

from typing import Callable, List, Optional

import pytest
from openai.types import Completion

from guidellm.backend import Backend, BackendEngine, OpenAIBackend
from guidellm.core import TextGenerationRequest
from tests.dummy.services import TestRequestGenerator


@pytest.mark.smoke
def test_openai_backend_creation_with_default_model(openai_backend_factory: Callable):
    """
    Test whether the OpenAI Backend service is created correctly
    with all default parameters.
    Also checks whether the `default_models` parameter does not abuse the OpenAI API.
    """

    backend_service = openai_backend_factory()

    assert isinstance(backend_service, OpenAIBackend)
    assert backend_service.default_model == backend_service.available_models()[0]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"openai_api_key": "dummy"},
        {"internal_callback_url": "dummy"},
    ],
)
def test_openai_backend_creation_required_arguments(mocker, extra_kwargs: dict):
    """
    Both OpenAI key & internal callback URL are required to work with OpenAI Backend.
    """

    # Ignore environment variables
    mocker.patch("os.getenv", return_value=None)

    with pytest.raises(ValueError):
        Backend.create(
            backend_type=BackendEngine.OPENAI_SERVER,
            **extra_kwargs,
        )


@pytest.mark.smoke
def test_model_tokenizer(openai_backend_factory):
    backend_service = openai_backend_factory()
    assert backend_service.model_tokenizer("bert-base-uncased")


@pytest.mark.smoke
def test_model_tokenizer_no_model(openai_backend_factory):
    backend_service = openai_backend_factory()
    tokenizer = backend_service.model_tokenizer("invalid")
    assert tokenizer is None


@pytest.mark.smoke
def test_make_request(
    openai_backend_factory, openai_completion_create_patch: List[Completion]
):
    """
    Test `OpenAIBackend.make_request()` workflow.

    Notes:
    * The output token count is not used without the `TextGenerationResult.start()`
        and `TextGenerationResult.start()`
    """

    request: TextGenerationRequest = TestRequestGenerator().create_item()
    backend_service: OpenAIBackend = openai_backend_factory()
    total_generative_responses = 0

    for generative_response, patched_completion in zip(
        backend_service.make_request(request=request),
        openai_completion_create_patch,
    ):
        total_generative_responses += 1
        expected_token: Optional[str] = getattr(patched_completion, "content") or None

        assert generative_response.add_token == expected_token
        assert (
            generative_response.type_ == "final"
            if getattr(patched_completion, "stop") is True
            else "token_iter"
        )
        if expected_token is not None:
            assert generative_response.prompt_token_count is None
            assert generative_response.output_token_count is None
        else:
            assert generative_response.prompt_token_count == 2
            assert generative_response.output_token_count == 0

    assert total_generative_responses == 3
