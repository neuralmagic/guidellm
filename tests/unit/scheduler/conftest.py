import pytest
from guidellm.core import TextGenerationRequest, TextGenerationResult


@pytest.fixture(autouse=True)
def backend_submit_patch(mocker):
    patch = mocker.patch(
        "guidellm.backend.base.Backend.submit",
        return_value=TextGenerationResult(
            request=TextGenerationRequest(prompt="Test prompt"),
        ),
    )
    patch.__name__ = "Backend.submit fallbackBackend.submit fallback"

    return patch
