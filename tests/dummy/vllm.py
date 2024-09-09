"""
This module includes data models factories for the `vllm` 3-rd party package
"""

import random
from functools import partial
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from guidellm.utils import random_strings

__all__ = ["TestLLM", "CompletionOutput"]


class CompletionOutput(BaseModel):
    """Test interface of `vllm.CompletionOutput`."""

    text: str


class SamplingParams(BaseModel):
    """Test interface of `vllm.SamplingParams`."""

    max_tokens: int


class TestLLM(BaseModel):
    """Test interface of `vllm.LLM`.

    Args:
        _outputs_number(int | None): the number of generated tokens per output.
            Should be used only for testing purposes.
            Default: randint(10..20)

    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )

    model: str
    max_num_batched_tokens: int

    # NOTE: This value is used only for testing purposes
    outputs_number: int = Field(default_factory=partial(random.randint, 10, 20))

    def _generate_completion_outputs(self, max_tokens: int) -> List[CompletionOutput]:
        self._outputs_number = random.randint(10, 20)

        return [
            CompletionOutput(text=text)
            for text in random_strings(
                min_chars=0, max_chars=max_tokens, n=self._outputs_number
            )
        ]

    def generate(
        self, inputs: List[str], sampling_params: SamplingParams
    ) -> Optional[List[List[CompletionOutput]]]:
        breakpoint()  # TODO: remove
        return [
            self._generate_completion_outputs(max_tokens=sampling_params.max_tokens)
        ]
