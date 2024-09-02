"""
This module includes unit tests for the vLLM backend.

Notes: tests from this module are going to be skipped in case
    the rimtime platform is not a Linux / WSL according to vllm documentation.
"""

from typing import Dict, List

import pytest

from guidellm.backend import Backend
from guidellm.config import reload_settings
from guidellm.core import TextGenerationRequest
from tests import dummy

# pytestmark = pytest.mark.skipif(
#     sys.platform != "linux",
#     reason="Unsupported Platform. Try using Linux or WSL instead.",
# )


@pytest.fixture(scope="module")
def backend_class():
    from guidellm.backend.vllm import VllmBackend

    return VllmBackend


@pytest.fixture(autouse=True)
def mock_vllm_llm(mocker):
    llm = dummy.vllm.TestLLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=4096,
    )

    return mocker.patch("vllm.LLM", return_value=llm)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    "create_payload",
    [
        {},
        {"model": "test/custom_llm"},
    ],
)
def test_backend_creation(create_payload: Dict, backend_class):
    """Test the "Deepspaarse Backend" class
    with defaults and custom input parameters.
    """

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
        {"prompt": "Test prompt"},
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
        generation.text for generation in backend.pipeline._generations
    )

    if max_tokens := text_generation_request_create_payload.get("output_token_count"):
        assert len(backend.pipeline._generations) == max_tokens


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
