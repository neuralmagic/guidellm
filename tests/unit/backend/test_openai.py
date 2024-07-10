"""
This module includes unit tests for the OpenAI Backend Service.
"""

from typing import Callable, List

import pytest
from openai.types import Completion

from domain.backend import Backend, BackendEngine, OpenAIBackend
from domain.core import TextGenerationRequest
from tests.dummy.services import TestRequestGenerator


@pytest.mark.sanity
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
def test_openai_backend_creation_required_arguments(extra_kwargs: dict):
    """
    Both OpenAI key & internal callback URL are required to work with OpenAI Backend.
    """
    with pytest.raises(ValueError):
        Backend.create(
            backend_type=BackendEngine.OPENAI_SERVER,
            **extra_kwargs,
        )


def test_model_tokenizer(openai_backend_factory):
    backend_service = openai_backend_factory()
    assert backend_service.model_tokenizer("bert-base-uncased")


def test_model_tokenizer_no_model(openai_backend_factory):
    backend_service = openai_backend_factory()
    tokenizer = backend_service.model_tokenizer("invalid")
    assert tokenizer is None


def test_make_request(
    openai_backend_factory, openai_completion_create_patch: List[Completion]
):
    request: TextGenerationRequest = TestRequestGenerator().create_item()
    backend_service: OpenAIBackend = openai_backend_factory()
    total_generative_responses = 0

    for generative_response, patched_completion in zip(
        backend_service.make_request(request=request),
        openai_completion_create_patch,
    ):
        total_generative_responses += 1
        expected_output: str = patched_completion.choices[0].text

        assert generative_response.type_ == "final"
        assert generative_response.output == expected_output
        assert generative_response.prompt_token_count == len(request.prompt.split())
        assert generative_response.output_token_count == len(expected_output.split())

    assert total_generative_responses == 1
