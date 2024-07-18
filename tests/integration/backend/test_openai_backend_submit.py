import os
import time
from typing import Callable

import pytest
import requests
from openai.pagination import SyncPage
from openai.types import Model

from guidellm.backend import OpenAIBackend
from guidellm.core import TextGenerationRequest, TextGenerationResult


@pytest.fixture(scope="session", autouse=True)
def openai_server_healthcheck():
    """
    Check if the openai server is running
    """

    if not (openai_server := os.getenv("OPENAI_BASE_URL", None)):
        raise ValueError(
            "Integration backend tests can't be run without OPENAI_BASE_URL specified"
        )

    try:
        requests.get(openai_server)
    except requests.ConnectionError:
        raise SystemExit(
            "Integration backend tests can't be run without "
            f"OpenAI compatible server running. Please check the {openai_server}"
        )


@pytest.mark.skip("OpenAI compatible service is not deployed yet")
@pytest.mark.integration
def test_openai_submit_request(
    mocker, openai_backend_factory: Callable[..., OpenAIBackend]
):
    """
    Check the OpenAI making request and checking the results.

    Check if the total time that is stored in the TextGenerationResult corresponds
        to the real execution time
    """

    openai_resources_models_list_patch = mocker.patch(
        "openai.resources.models.Models.list",
        return_value=SyncPage(
            object="list",
            data=[
                Model(
                    id="d69244a8-3f30-4f08-a432-8c83d5f254ad",
                    created=1719814049,
                    object="model",
                    owned_by="guidellm",
                )
            ],
        ),
    )
    backend: OpenAIBackend = openai_backend_factory()
    request = TextGenerationRequest(prompt="Generate numbers from 1 to 10")

    start_time = time.perf_counter()
    result: TextGenerationResult = backend.submit(request=request)
    total_for_submit = time.perf_counter() - start_time

    assert result.start_time is not None
    assert result.end_time is not None
    assert openai_resources_models_list_patch.call_count == 1
    assert abs((result.end_time - result.start_time) - total_for_submit) < 1
