import random
from typing import List

import pytest
from loguru import logger
from openai.pagination import SyncPage
from openai.types import Completion, Model

from domain.backend import Backend, BackendEngine, OpenAIBackend

from . import dummy


def pytest_configure() -> None:
    logger.disable("guidellm")


@pytest.fixture(autouse=True)
def openai_models_list_patch(mocker) -> List[Model]:
    """
    Mock available models function to avoid OpenAI API call.
    """

    items = dummy.data.OpenAIModel.batch(3)
    mocker.patch(
        "openai.resources.models.Models.list",
        return_value=SyncPage(object="list", data=items),
    )

    return items


@pytest.fixture(autouse=True)
def openai_completion_create_patch(mocker) -> List[Completion]:
    """
    Mock available models function to avoid OpenAI API call.
    """

    items = dummy.data.OpenAICompletion.batch(random.randint(2, 5))
    mocker.patch("openai.resources.completions.Completions.create", return_value=items)

    return items


@pytest.fixture
def openai_backend_factory():
    """
    OpenAI Backend factory method.
    Call without provided arguments returns default Backend service.
    """

    def inner_wrapper(*_, **kwargs) -> OpenAIBackend:
        static = {"backend_type": BackendEngine.OPENAI_SERVER}
        defaults = {
            "openai_api_key": "required but not used",
            "internal_callback_url": "http://localhost:8080",
        }

        defaults.update(kwargs)
        defaults.update(static)

        return Backend.create(**defaults)

    return inner_wrapper
