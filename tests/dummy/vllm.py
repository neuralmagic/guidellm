"""
This module includes data models factories for the `vllm` 3-rd party package
"""

import random
from typing import Generator, List, Optional

from pydantic import BaseModel, ConfigDict

from guidellm.utils import random_strings

__all__ = ["TestLLM", "CompletionOutput"]


class CompletionOutput(BaseModel):
    """Test interface of `vllm.CompletionOutput`."""

    text: str


class SamplingParams(BaseModel):
    """Test interface of `vllm.SamplingParams`."""

    max_tokens: Optional[int] = 16


class CompletionOutputs(BaseModel):
    outputs: List[CompletionOutput]


class TestLLM(BaseModel):
    """Test interface of `vllm.LLM`.

    Args:
        _outputs_number(int | None): the number of generated tokens per output.
            Should be used only for testing purposes.
            Default: randint(10..20)
        _generations: dynamic representation of generated responses
            from deepsparse interface.

    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )

    model: str
    max_num_batched_tokens: int

    def _generate_completion_outputs(
        self, max_tokens: Optional[int]
    ) -> Generator[CompletionOutput, None, None]:

        # NOTE: This value is used only for testing purposes
        self._expected_outputs: List[CompletionOutput] = []

        for text in random_strings(
            min_chars=5,
            max_chars=random.randint(10, 20),
            n=max_tokens or random.randint(10, 20),
        ):
            instance = CompletionOutput(text=text)
            self._expected_outputs.append(instance)

            yield instance

    def generate(
        self, inputs: List[str], sampling_params: SamplingParams
    ) -> List[CompletionOutputs]:
        return [
            CompletionOutputs(
                outputs=list(
                    self._generate_completion_outputs(
                        max_tokens=sampling_params.max_tokens
                    )
                )
            )
        ]
