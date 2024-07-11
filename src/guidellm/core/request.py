import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger


@dataclass(frozen=True)
class TextGenerationRequest:
    """
    A class to represent a text generation request for generative AI workloads.

    :param prompt: The input prompt for the text generation request.
    :type prompt: str
    :param prompt_token_count: The number of tokens in the prompt, defaults to None.
    :type prompt_token_count: Optional[int]
    :param generated_token_count: The number of tokens to generate, defaults to None.
    :type generated_token_count: Optional[int]
    :param params: Optional parameters for the text generation request,
        defaults to None.
    :type params: Optional[Dict[str, Any]]
    """

    prompt: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    prompt_token_count: Optional[int] = None
    generated_token_count: Optional[int] = None
    params: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Log the initialization of the TextGenerationRequest instance.
        """

        logger.debug(
            f"Initialized TextGenerationRequest with id={self.id}, "
            f"prompt={self.prompt}, prompt_token_count={self.prompt_token_count}, "
            f"generated_token_count={self.generated_token_count}, params={self.params}"
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the TextGenerationRequest.

        :return: String representation of the TextGenerationRequest.
        :rtype: str
        """
        return (
            f"TextGenerationRequest("
            f"id={self.id}, "
            f"prompt={self.prompt}, "
            f"prompt_token_count={self.prompt_token_count}, "
            f"generated_token_count={self.generated_token_count}, "
            f"params={self.params})"
        )
