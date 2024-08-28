import uuid
from typing import Any, Dict, Optional

from pydantic import Field

from guidellm.core.serializable import Serializable


class TextGenerationRequest(Serializable):
    """
    A class to represent a text generation request for generative AI workloads.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the request.",
    )
    prompt: str = Field(description="The input prompt for the text generation.")
    prompt_token_count: Optional[int] = Field(
        default=None,
        description="The number of tokens in the input prompt.",
    )
    output_token_count: Optional[int] = Field(
        default=None,
        description="The number of tokens to generate.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="The parameters for the text generation request.",
    )

    def __str__(self) -> str:
        prompt_short = (
            self.prompt[:32] + "..."
            if self.prompt and len(self.prompt) > 32  # noqa: PLR2004
            else self.prompt
        )

        return (
            f"TextGenerationRequest(id={self.id}, "
            f"prompt={prompt_short}, prompt_token_count={self.prompt_token_count}, "
            f"output_token_count={self.output_token_count}, "
            f"params={self.params})"
        )
