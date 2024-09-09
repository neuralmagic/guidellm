"""
This module includes unit tests for the vLLM backend.

Notes: tests from this module are going to be skipped in case
    the rimtime platform is not a Linux / WSL according to vllm documentation.
"""

import sys
from typing import Callable, Dict, List, Optional

import pytest

from guidellm.backend import Backend
from guidellm.config import reload_settings, settings
from guidellm.core import TextGenerationRequest
from tests import dummy

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Unsupported Platform. Try using Linux or WSL instead.",
)


@pytest.fixture(scope="session")
def backend_class():
    from guidellm.backend.vllm import VllmBackend

    return VllmBackend


@pytest.fixture()
def vllm_patch_factory(mocker) -> Callable[[str], dummy.vllm.TestLLM]:
    """
    Skip VLLM initializer due to external calls.
    Replace VllmBackend.llm object with mock representation.

    This vllm patch is injected into each test automatically. If you need
    to override the Mock object - use this fixture.
    """

    def inner(model: Optional[str] = None, max_tokens: Optional[int] = None):

        return mocker.patch(
            "vllm.LLM.__new__",
            return_value=dummy.vllm.TestLLM(
                model=model or settings.llm_model,
                max_num_batched_tokens=max_tokens or 4096,
            ),
        )

    return inner


@pytest.fixture(autouse=True)
def vllm_auto_patch(vllm_patch_factory):
    """
    Automatically patch the ``vllm.LLM`` with defaults.
    """

    return vllm_patch_factory()


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "create_payload",
    [
        {},
        {"model": "test/custom_llm"},
    ],
)
def test_backend_creation(create_payload: Dict, backend_class, vllm_patch_factory):
    """Test the "Deepspaarse Backend" class
    with defaults and custom input parameters.
    """

    vllm_patch_factory(model=create_payload.get("model"))

    backends = [
        Backend.create("vllm", **create_payload),
        backend_class(**create_payload),
    ]

    for backend in backends:
        assert backend.llm
        (
            backend.model == custom_model
            if (custom_model := create_payload.get("model"))
            else backend.default_model
        )


@pytest.mark.smoke()
def test_backend_model_from_env(mocker, backend_class):
    mocker.patch.dict(
        "os.environ",
        {"GUIDELLM__LLM_MODEL": "test_backend_model_from_env"},
    )

    reload_settings()

    backends = [Backend.create("vllm"), backend_class()]

    for backend in backends:
        assert backend.model == "test_backend_model_from_env"


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "text_generation_request_create_payload",
    [
        # {"prompt": "Test prompt"},
        {"prompt": "Test prompt", "output_token_count": 20},
    ],
)
@pytest.mark.asyncio()
async def test_make_request(
    text_generation_request_create_payload: Dict, backend_class
):
    backend = backend_class()

    output_tokens: List[str] = []
    async for response in backend.make_request(
        request=TextGenerationRequest(**text_generation_request_create_payload)
    ):
        if response.add_token:
            output_tokens.append(response.add_token)

    assert "".join(output_tokens) == "".join(
        generation.text for generation in getattr(backend.llm, "_expected_outputs")
    )

    if max_tokens := text_generation_request_create_payload.get("output_token_count"):
        assert len(getattr(backend.llm, "_expected_outputs")) == max_tokens


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("text_generation_request_create_payload", "error"),
    [
        ({"prompt": "Test prompt"}, ValueError),
    ],
)
@pytest.mark.asyncio()
async def test_make_request_invalid_request_payload(
    text_generation_request_create_payload: Dict, error, backend_class
):
    backend = backend_class()
    with pytest.raises(error):
        [
            respnose
            async for respnose in backend.make_request(
                request=TextGenerationRequest(**text_generation_request_create_payload)
            )
        ]
