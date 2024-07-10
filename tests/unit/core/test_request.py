import pytest

from domain.core import TextGenerationRequest


@pytest.mark.smoke
def test_text_generation_request_initialization():
    prompt = "Generate a story"
    request = TextGenerationRequest(prompt)
    assert request.prompt == prompt
    assert request.prompt_token_count is None
    assert request.generated_token_count is None
    assert request.params == {}


@pytest.mark.sanity
def test_text_generation_request_initialization_with_params():
    prompt = "Generate a story"
    prompt_token_count = 50
    generated_token_count = 100
    params = {"temperature": 0.7}
    request = TextGenerationRequest(
        prompt, prompt_token_count, generated_token_count, params
    )
    assert request.prompt == prompt
    assert request.prompt_token_count == prompt_token_count
    assert request.generated_token_count == generated_token_count
    assert request.params == params


@pytest.mark.regression
def test_text_generation_request_repr():
    prompt = "Generate a story"
    prompt_token_count = 50
    generated_token_count = 100
    params = {"temperature": 0.7}
    request = TextGenerationRequest(
        prompt, prompt_token_count, generated_token_count, params
    )
    assert repr(request) == (
        f"TextGenerationRequest(id={request.id}, prompt={prompt}, "
        f"prompt_token_count={prompt_token_count}, "
        f"generated_token_count={generated_token_count}, params={params})"
    )
