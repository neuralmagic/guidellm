from typing import Dict, Any, Optional


__all__ = ["BenchmarkRequest"]


class BenchmarkRequest:
    """
    A class to represent a benchmark request for generative AI workloads.

    :param prompt: The input prompt for the benchmark request.
    :type prompt: str
    :param params: Optional parameters for the benchmark request, defaults to None.
    :type params: Optional[Dict[str, Any]]
    """

    def __init__(self, prompt: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the BenchmarkRequest with a prompt and optional parameters.

        :param prompt: The input prompt for the benchmark request.
        :type prompt: str
        :param params: Optional parameters for the benchmark request, defaults to None.
        :type params: Optional[Dict[str, Any]]
        """
        self._prompt = prompt
        self._params = params or {}

    @property
    def prompt(self) -> str:
        """
        Get the input prompt for the benchmark request.

        :return: The input prompt.
        :rtype: str
        """
        return self._prompt

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
        return f"BenchmarkRequest(prompt={self._prompt}, params={self._params})"

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the BenchmarkRequest for debugging.

        :return: Unambiguous string representation of the BenchmarkRequest.
        :rtype: str
        """
        return f"BenchmarkRequest(prompt={self._prompt}, params={self._params})"
