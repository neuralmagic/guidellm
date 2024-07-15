import os
from typing import Callable, Optional

import pytest
from loguru import logger

from guidellm.backend import Backend, BackendEngine, OpenAIBackend


def pytest_configure() -> None:
    logger.disable("guidellm")


@pytest.fixture
def openai_backend_factory() -> Callable[..., OpenAIBackend]:
    """
    OpenAI Backend factory method.
    Call without provided arguments returns default Backend service.
    """

    def inner_wrapper(*_, base_url: Optional[str] = None, **kwargs) -> OpenAIBackend:
        defaults = {
            "backend_type": BackendEngine.OPENAI_SERVER,
            "openai_api_key": "required but not used",
            "internal_callback_url": base_url
            or os.getenv("OPENAI_BASE_URL", "http://localhost:8080"),
        }

        defaults.update(kwargs)

        return Backend.create(**defaults)

    return inner_wrapper
