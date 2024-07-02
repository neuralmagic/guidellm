import random
from typing import List

import pytest
from openai.pagination import SyncPage
from openai.types import Completion, Model

from guidellm.backend import Backend, BackendEngine, OpenAIBackend

from . import factories


@pytest.fixture(autouse=True)
def openai_models_list_patch(mocker) -> List[Model]:
    """
    Mock available models function to avoid OpenAI API call.
    """

    items = factories.OpenAIModel.batch(3)
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

    items = factories.OpenAICompletion.batch(random.randint(2, 5))
    mocker.patch("openai.resources.completions.Completions.create", return_value=items)

    return items


@pytest.fixture
def openai_backend_factory():
    """
    Create a test openai backend service.
    Call without provided arguments returns default Backend service.
    """

    def inner_wrapper(*_, **kwargs) -> OpenAIBackend:
        static = {"backend_type": BackendEngine.OPENAI_SERVER}
        defaults = {
            "openai_api_key": "dummy api key",
            "internal_callback_url": "http://localhost:8000",
        }

        defaults.update(kwargs)
        defaults.update(static)

        return Backend.create(**defaults)

    return inner_wrapper
