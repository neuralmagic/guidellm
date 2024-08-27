from typing import Optional

from guidellm.core import TextGenerationRequest
from guidellm.request import GenerationMode, RequestGenerator


class TestRequestGenerator(RequestGenerator):
    """
    This class represents the Testing Request Generator.
    The purpose - to be used for testing.
    """

    def __init__(
        self,
        tokenizer: Optional[str] = None,
        mode: GenerationMode = "async",
        async_queue_size: int = 50,
    ):
        super().__init__(
            type_="test",
            source="test",
            tokenizer=tokenizer,
            mode=mode,
            async_queue_size=async_queue_size,
        )

    def create_item(self) -> TextGenerationRequest:
        return TextGenerationRequest(prompt="Test prompt")
