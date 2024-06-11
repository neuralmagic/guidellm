from typing import Dict, Any, Optional
import uuid


__all__ = ["BenchmarkRequest"]


class BenchmarkRequest:
    """
    A class to represent a benchmark request for generative AI workloads.

    :param prompt: The input prompt for the benchmark request.
    :type prompt: str
    :param token_count: The number of tokens to generate, defaults to None.
    :type token_count: Optional[int]
    :param params: Optional parameters for the benchmark request, defaults to None.
    :type params: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        prompt: str,
        token_count: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the BenchmarkRequest with a prompt and optional parameters.

        :param prompt: The input prompt for the benchmark request.
        :type prompt: str
        :param params: Optional parameters for the benchmark request, defaults to None.
        :type params: Optional[Dict[str, Any]]
        """
        self._id = str(uuid.uuid4())
        self._prompt = prompt
        self._token_count = token_count
        self._params = params or {}

    @property
    def id(self) -> str:
        """
        Get the unique identifier for the benchmark request.

        :return: The unique identifier.
        :rtype: str
        """
        return self._id

    @property
    def prompt(self) -> str:
        """
        Get the input prompt for the benchmark request.

        :return: The input prompt.
        :rtype: str
        """
        return self._prompt

    @property
    def token_count(self) -> Optional[int]:
        """
        Get the number of tokens to generate for the benchmark request.

        :return: The number of tokens to generate.
        :rtype: Optional[int]
        """
        return self._token_count

    @property
    def params(self) -> Dict[str, Any]:
        """
        Get the optional parameters for the benchmark request.

        :return: The optional parameters.
        :rtype: Dict[str, Any]
        """
        return self._params

    def __str__(self) -> str:
        """
        Return a string representation of the BenchmarkRequest.

        :return: String representation of the BenchmarkRequest.
        :rtype: str
        """
        return f"BenchmarkRequest(id={self.id}, prompt={self._prompt}, params={self._params})"

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the BenchmarkRequest for debugging.

        :return: Unambiguous string representation of the BenchmarkRequest.
        :rtype: str
        """
        return f"BenchmarkRequest(id={self.id}, prompt={self._prompt}, params={self._params})"
