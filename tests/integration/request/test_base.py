import pytest
from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import RequestGenerator
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TestRequestGenerator(RequestGenerator):
    def create_item(self) -> TextGenerationRequest:
        return TextGenerationRequest(prompt="Test prompt")


@pytest.mark.smoke()
def test_request_generator_with_hf_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    generator = TestRequestGenerator(tokenizer=tokenizer)
    assert generator.tokenizer == tokenizer


@pytest.mark.smoke()
def test_request_generator_with_string_tokenizer():
    generator = TestRequestGenerator(tokenizer="bert-base-uncased")
    assert isinstance(generator.tokenizer, PreTrainedTokenizerBase)
    assert generator.tokenizer.name_or_path == "bert-base-uncased"
