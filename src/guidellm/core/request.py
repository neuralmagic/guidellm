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
