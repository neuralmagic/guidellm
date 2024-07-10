import uuid
from typing import Any, Dict, Optional

from loguru import logger


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

    def __init__(
        self,
        prompt: str,
        prompt_token_count: Optional[int] = None,
        generated_token_count: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TextGenerationRequest with a prompt and optional parameters.

        :param prompt: The input prompt for the text generation request.
        :type prompt: str
        :param prompt_token_count: The number of tokens in the prompt, defaults to None.
        :type prompt_token_count: Optional[int]
        :param generated_token_count: The number of tokens to generate,
            defaults to None.
        :type generated_token_count: Optional[int]
        :param params: Optional parameters for the text generation request,
            defaults to None.
        :type params: Optional[Dict[str, Any]]
        """
        self._id = str(uuid.uuid4())
        self._prompt = prompt
        self._prompt_token_count = prompt_token_count
        self._generated_token_count = generated_token_count
        self._params = params or {}

        logger.debug(
            f"Initialized TextGenerationRequest with id={self._id}, "
            f"prompt={prompt}, prompt_token_count={prompt_token_count}, "
            f"generated_token_count={generated_token_count}, params={params}"
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the TextGenerationRequest.

        :return: String representation of the TextGenerationRequest.
        :rtype: str
        """
        return (
            f"TextGenerationRequest("
            f"id={self._id}, "
            f"prompt={self._prompt}, "
            f"prompt_token_count={self._prompt_token_count}, "
            f"generated_token_count={self._generated_token_count}, "
            f"params={self._params})"
        )

    @property
    def id(self) -> str:
        """
        Get the unique identifier for the text generation request.

        :return: The unique identifier.
        :rtype: str
        """
        return self._id

    @property
    def prompt(self) -> str:
        """
        Get the input prompt for the text generation request.

        :return: The input prompt.
        :rtype: str
        """
        return self._prompt

    @property
    def prompt_token_count(self) -> Optional[int]:
        """
        Get the number of tokens in the prompt for the text generation request.

        :return: The number of tokens in the prompt.
        :rtype: Optional[int]
        """
        return self._prompt_token_count

    @property
    def generated_token_count(self) -> Optional[int]:
        """
        Get the number of tokens to generate for the text generation request.

        :return: The number of tokens to generate.
        :rtype: Optional[int]
        """
        return self._generated_token_count

    @property
    def params(self) -> Dict[str, Any]:
        """
        Get the optional parameters for the text generation request.

        :return: The optional parameters.
        :rtype: Dict[str, Any]
        """
        return self._params
