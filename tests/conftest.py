from typing import Callable, Optional

import pytest
from guidellm.backend import Backend, BackendEngine, OpenAIBackend
from guidellm.config import settings
from loguru import logger


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
            "backend_type": BackendEngine.OPENAI_SERVER,
            "openai_api_key": "required but not used",
            "target": base_url or settings.openai.base_url,
        }

        defaults.update(kwargs)

        return Backend.create(**defaults)  # type: ignore

    return inner_wrapper
