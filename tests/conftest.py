from pathlib import Path
from typing import Callable, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests_mock
from loguru import logger

from guidellm.backend import Backend, OpenAIBackend
from guidellm.config import settings


def pytest_configure() -> None:
    logger.disable("guidellm")


@pytest.fixture()
def openai_backend_factory() -> Callable[..., OpenAIBackend]:
    """
    OpenAI Backend factory method.
    Call without provided arguments returns default Backend service.
    """

    def inner_wrapper(*_, base_url: Optional[str] = None, **kwargs) -> OpenAIBackend:
        defaults = {
            "backend_type": "openai_server",
            "openai_api_key": "required but not used",
            "target": base_url or settings.openai.base_url,
        }

        defaults.update(kwargs)

        return Backend.create(**defaults)  # type: ignore

    return inner_wrapper


@pytest.fixture()
def mock_auto_tokenizer():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:

        def _fake_tokenize(text: str) -> List[int]:
            tokens = text.split()
            return [0] * len(tokens)

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = MagicMock(side_effect=_fake_tokenize)
        mock_from_pretrained.return_value = mock_tokenizer
        yield mock_tokenizer


@pytest.fixture()
def mock_requests_pride_and_prejudice():
    text_path = Path(__file__).parent / "dummy" / "data" / "pride_and_prejudice.txt"
    text_content = text_path.read_text()

    with requests_mock.Mocker() as mock:
        mock.get(
            "https://www.gutenberg.org/files/1342/1342-0.txt",
            text=text_content,
        )
        yield mock
