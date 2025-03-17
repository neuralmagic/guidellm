from pathlib import Path
from typing import Optional, Union

from loguru import logger
from transformers import PreTrainedTokenizer  # type: ignore  # noqa: PGH003

from guidellm.config import settings
from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import GenerationMode, RequestGenerator
from guidellm.utils import load_text_lines

__all__ = ["FileRequestGenerator"]


class FileRequestGenerator(RequestGenerator):
    """
    A request generator implementation for files.

    :param path: The path to the file containing the data.
    :type path: Optional[Union[str, Path]]
    :param tokenizer: The tokenizer instance or the name/config to use
        for tokenizing prompts.
    :type tokenizer: Union[str, PreTrainedTokenizer]
    :param mode: The generation mode, either 'async' or 'sync'.
    :type mode: str
    :param async_queue_size: The size of the request queue.
    :type async_queue_size: int
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]],
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: GenerationMode = "async",
        async_queue_size: int = 50,
    ):
        if not path:
            raise ValueError("File path must be provided for FileRequestGenerator")

        self._path = path
        self._data = load_text_lines(
            path,
            filters=settings.dataset.preferred_data_columns,
        )
        self._iterator = iter(self._data)

        # NOTE: Must be after all the parameters since the queue population
        #       function requires attributes above
        super().__init__(
            type_="file",
            source=str(path),
            tokenizer=tokenizer,
            mode=mode,
            async_queue_size=async_queue_size,
        )

    def __len__(self) -> int:
        """
        Return the number of text lines.
        """

        return len(self._data)

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new result request item from the data.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """
        logger.debug("Creating new request item from file data")

        try:
            data = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._data)
            data = next(self._iterator)

        token_count = len(self.tokenizer.tokenize(data))
        request = TextGenerationRequest(prompt=data, prompt_token_count=token_count)
        logger.debug("Created new TextGenerationRequest: {}", request)

        return request
