"""
This module includes unit tests for the vLLM backend.

Notes: tests from this module are going to be skipped in case
    the rimtime platform is not a Linux / WSL according to vllm documentation.
"""

import importlib
import sys
from typing import Dict

import pytest

from guidellm.backend import Backend

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Unsupported Platform. Try using Linux or WSL instead.",
)


@pytest.fixture(scope="module")
def backend_class():
    from guidellm.backend.vllm import VllmBackend

    return VllmBackend


@pytest.fixture(autouse=True)
def mock_vllm_llm(mocker):
    module = importlib.import_module("vllm")
    llm = module.LLM(
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
