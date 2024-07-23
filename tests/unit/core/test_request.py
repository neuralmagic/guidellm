import pytest

from guidellm.core import TextGenerationRequest


@pytest.mark.smoke
def test_text_generation_request_initialization():
    prompt = "Generate a story"
    request = TextGenerationRequest(prompt=prompt)
    assert request.prompt == prompt
    assert request.prompt_token_count is None
    assert request.generate_token_count is None
    assert request.params == {}


@pytest.mark.sanity
def test_text_generation_request_initialization_with_params():
    prompt = "Generate a story"
    prompt_token_count = 50
    generate_token_count = 100
    params = {"temperature": 0.7}
    request = TextGenerationRequest(
        prompt=prompt,
        prompt_token_count=prompt_token_count,
        generate_token_count=generate_token_count,
        params=params,
    )
    assert request.prompt == prompt
    assert request.prompt_token_count == prompt_token_count
    assert request.generate_token_count == generate_token_count
    assert request.params == params


@pytest.mark.regression
def test_request_json():
    prompt = "Generate text"
    prompt_token_count = 10
    generate_token_count = 50
    params = {"temperature": 0.7}
    request = TextGenerationRequest(
        prompt=prompt,
        prompt_token_count=prompt_token_count,
        generate_token_count=generate_token_count,
        params=params,
    )
    json_str = request.to_json()
    assert '"prompt":"Generate text"' in json_str
    assert '"id":' in json_str

    request_restored = TextGenerationRequest.from_json(json_str)
    assert request.id == request_restored.id
    assert request_restored.prompt == prompt
    assert request_restored.prompt_token_count == prompt_token_count
    assert request_restored.generate_token_count == generate_token_count
    assert request_restored.params == params


@pytest.mark.regression
def test_request_yaml():
    prompt = "Generate text"
    prompt_token_count = 15
    generate_token_count = 55
    params = {"temperature": 0.8}
    request = TextGenerationRequest(
        prompt=prompt,
        prompt_token_count=prompt_token_count,
        generate_token_count=generate_token_count,
        params=params,
    )
    yaml_str = request.to_yaml()
    assert "prompt: Generate text" in yaml_str
    assert "id:" in yaml_str

    request_restored = TextGenerationRequest.from_yaml(yaml_str)
    assert request.id == request_restored.id
    assert request_restored.prompt == prompt
    assert request_restored.prompt_token_count == prompt_token_count
    assert request_restored.generate_token_count == generate_token_count
    assert request_restored.params == params
