from typing import Iterator, List, Optional

import pytest
from guidellm.backend import Backend, BackendEngine, GenerativeResponse, OpenAIBackend
from guidellm.core import TextGenerationRequest


@Backend.register(backend_type=BackendEngine.TEST)
class TestBackend(Backend):
    """
    The test implementation of a LLM Backend.
    """

    def __init__(self, target: str, model: str = "test"):
        self.target: str = target
        self.model: str = model

    def make_request(
        self,
        request: TextGenerationRequest,
    ) -> Iterator[GenerativeResponse]:
        raise NotImplementedError

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

    assert Backend._registry == {
        BackendEngine.TEST: TestBackend,
        BackendEngine.OPENAI_SERVER: OpenAIBackend,
    }
