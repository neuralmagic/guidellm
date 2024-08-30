from typing import Dict, List, cast

import pytest
from vllm import LLM

from guidellm.backend import Backend, VllmBackend


@pytest.fixture(autouse=True)
def mock_vllm_llm(mocker):
    llm = LLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
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
def test_backend_creation(create_payload: Dict):
    """Test the "Deepspaarse Backend" class
    with defaults and custom input parameters.
    """

    backends: List[VllmBackend] = cast(
        List[VllmBackend],
        [
            Backend.create("vllm", **create_payload),
            VllmBackend(**create_payload),
        ],
    )

    for backend in backends:
        assert backend.llm
        (
            backend.model == custom_model
            if (custom_model := create_payload.get("model"))
            else backend.default_model
        )
