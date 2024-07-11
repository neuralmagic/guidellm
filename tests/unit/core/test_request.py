import pytest

from guidellm.core import TextGenerationRequest


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
        prompt=prompt,
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        params=params,
    )
    assert request.prompt == prompt
    assert request.prompt_token_count == prompt_token_count
    assert request.generated_token_count == generated_token_count
    assert request.params == params


@pytest.mark.regression
def test_request_to_json():
    prompt = "Generate text"
    request = TextGenerationRequest(prompt=prompt)
    json_str = request.to_json()
    assert '"prompt":"Generate text"' in json_str
    assert '"id":' in json_str


@pytest.mark.regression
def test_request_from_json():
    json_str = (
        '{"id": "12345", "prompt": "Generate text", "prompt_token_count": 10, '
        '"generated_token_count": 50, "params": {"temperature": 0.7}}'
    )
    request = TextGenerationRequest.from_json(json_str)
    assert request.id == "12345"
    assert request.prompt == "Generate text"
    assert request.prompt_token_count == 10
    assert request.generated_token_count == 50
    assert request.params == {"temperature": 0.7}


@pytest.mark.regression
def test_request_to_yaml():
    prompt = "Generate text"
    request = TextGenerationRequest(prompt=prompt)
    yaml_str = request.to_yaml()
    assert "prompt: Generate text" in yaml_str
    assert "id:" in yaml_str


@pytest.mark.regression
def test_request_from_yaml():
    yaml_str = (
        "id: '12345'\nprompt: Generate text\nprompt_token_count: 10\n"
        "generated_token_count: 50\nparams:\n  temperature: 0.7\n"
    )
    request = TextGenerationRequest.from_yaml(yaml_str)
    assert request.id == "12345"
    assert request.prompt == "Generate text"
    assert request.prompt_token_count == 10
    assert request.generated_token_count == 50
    assert request.params == {"temperature": 0.7}
