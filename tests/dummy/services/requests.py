from guidellm.core import TextGenerationRequest
from guidellm.request import RequestGenerator


class TestRequestGenerator(RequestGenerator):
    """
    This class represents the Testing Request Generator.
    The purpose - to be used for testing.
    """

    def create_item(self) -> TextGenerationRequest:
        return TextGenerationRequest(prompt="Test prompt")
