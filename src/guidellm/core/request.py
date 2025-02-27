import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from guidellm.core.serializable import Serializable
from guidellm.utils import ImageDescriptor


class TextGenerationRequest(Serializable):
    """
    A class to represent a text generation request for generative AI workloads.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the request.",
    )
    prompt: str = Field(description="The input prompt for the text generation.")
    images: Optional[List[ImageDescriptor]] = Field(
        default=None,
        description="Input images.",
    )
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

    @property
    def number_images(self) -> int:
        if self.images is None:
            return 0
        else:
            return len(self.images)

    @property
    def image_resolution(self) -> List[Tuple[int, int]]:
        if self.images is None:
            return None
        else:
            return [im.size for im in self.images]


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
            f"image_resolution={self.image_resolution}"
        )
