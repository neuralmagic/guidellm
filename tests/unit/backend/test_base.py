from typing import Iterator, List, Optional
import time

import pytest

from guidellm.backend import Backend, BackendEngine, GenerativeResponse
from guidellm.core import TextGenerationRequest


TEST_TOKENS = ["test_token_1", "test_token_2", "test_token_3"]
TEST_TOKEN_GEN_TIME = 0.1


@Backend.register(backend_type=BackendEngine.TEST)
class TestBackend(Backend):
    """
    The test implementation of a LLM Backend.
    """

    def __init__(self, target: str, model: str):
        self.target: str = target
        self.model: str = model

    def make_request(
        self,
        request: TextGenerationRequest,
    ) -> Iterator[GenerativeResponse]:
        for token in TEST_TOKENS:
            time.sleep(TEST_TOKEN_GEN_TIME)
            yield GenerativeResponse(
                type_="token_iter",
                add_token=token,
            )

        yield GenerativeResponse(
            type_="final",
            prompt=request.prompt,
            output=" ".join(TEST_TOKENS),
            prompt_token_count=request.prompt_token_count,
        )

    def available_models(self) -> List[str]:
        raise NotImplementedError

    @property
    def default_model(self) -> str:
        raise NotImplementedError

    def model_tokenizer(self, model: str) -> Optional[str]:
        raise NotImplementedError


@pytest.mark.smoke()
def test_backend_registry():
    """
    Ensure that all registered classes exist in the Backend._registry.
    """

    assert BackendEngine.TEST in Backend._registry


@pytest.mark.smoke
def test_backend_creation():
    backend = Backend.create(
        BackendEngine.TEST, target="test_target", model="test_model"
    )
    assert backend is not None
    assert isinstance(backend, TestBackend)
    assert backend.target == "test_target"
    assert backend.model == "test_model"


@pytest.mark.smoke
def test_backend_submit():
    backend = Backend.create(
        BackendEngine.TEST, target="test_target", model="test_model"
    )
    request = TextGenerationRequest(prompt="test_prompt")

    result = backend.submit(request)
    assert result.request == request
    assert result.prompt == request.prompt
    assert result.prompt_word_count == 1
    assert result.prompt_token_count == 1
    assert result.output == " ".join(TEST_TOKENS)
    assert result.output_word_count == len(TEST_TOKENS)
    assert result.output_token_count == len(TEST_TOKENS)
    assert result.last_time is not None
    assert result.first_token_set
    assert result.start_time is not None
    assert result.end_time is not None
    assert result.end_time > result.start_time
    assert result.end_time - result.start_time >= len(TEST_TOKENS) * TEST_TOKEN_GEN_TIME
    assert result.first_token_time is not None
    assert result.first_token_time < result.end_time - result.start_time
    assert result.decode_times is not None
    assert result.decode_times.mean >= 0
    assert result.decode_times.mean < TEST_TOKEN_GEN_TIME * 1.1  # 10% tolerance
