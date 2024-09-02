"""
This package encapsulates the "vLLM Backend" implementation.

ref: https://github.com/vllm-project/vllm

The `vllm` package supports Python3.8..Python3.11,
when the `guidellm` start from Python3.8.

Safe range of versions is Python3.8..Python3.11
for the vLLM Backend implementation.

In the end ensure that the `vllm` package is installed.
"""

from guidellm.utils import check_python_version, module_is_available

check_python_version(min_version="3.8", max_version="3.12")

# module_is_available(
#     module="vllm",
#     helper=("`vllm` package is not available. Try run: `pip install -e '.[vllm]'`"),
# )

from .backend import VllmBackend  # noqa: E402

__all__ = ["VllmBackend"]
