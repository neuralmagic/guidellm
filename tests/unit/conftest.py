from typing import List, cast

import openai
import pytest
from openai.pagination import SyncPage

from tests import dummy


@pytest.fixture(autouse=True)
def openai_completion_create_patch(mocker) -> openai.Stream[openai.types.Completion]:
    """
    Mock available models function to avoid OpenAI API call.
    """

    items = [item for item in dummy.data.openai_completion_factory()]
    mocker.patch("openai.resources.completions.Completions.create", return_value=items)

    return cast(openai.Stream[openai.types.Completion], items)


@pytest.fixture(autouse=True)
def openai_models_list_patch(mocker) -> List[openai.types.Model]:
    """
    Mock available models function to avoid OpenAI API call.
    """

    items: List[openai.types.Model] = [
        item for item in dummy.data.openai_model_factory()
    ]
    mocker.patch(
        "openai.resources.models.Models.list",
        return_value=SyncPage(object="list", data=items),
    )

    return items
