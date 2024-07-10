"""
The `guidellm/request/test.py` package includes test domain components
of the `guidellm.request` subdomain.
"""

from domain.core import TextGenerationRequest
from domain.request import RequestGenerator


class TestRequestGenerator(RequestGenerator):
    """
    This class represents the Testing Request Generator.
    The purpose - to be used for testing.
    """

    def create_item(self) -> TextGenerationRequest:
        return TextGenerationRequest(prompt="Test prompt")
