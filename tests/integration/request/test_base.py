import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tests.dummy.services import TestRequestGenerator


@pytest.mark.smoke
def test_request_generator_with_hf_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    generator = TestRequestGenerator(tokenizer=tokenizer)
    assert generator.tokenizer == tokenizer


@pytest.mark.smoke
def test_request_generator_with_string_tokenizer():
    generator = TestRequestGenerator(tokenizer="bert-base-uncased")
    assert isinstance(generator.tokenizer, PreTrainedTokenizerBase)
    assert generator.tokenizer.name_or_path == "bert-base-uncased"
