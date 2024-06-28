import threading
import time
from abc import ABC, abstractmethod
from queue import Empty, Full, Queue
from typing import Iterator, Optional, Union

from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizer

from guidellm.core.request import TextGenerationRequest

__all__ = ["RequestGenerator"]


class RequestGenerator(ABC):
    """
    A base class for request generators that generate result requests.

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
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: str = "async",
        async_queue_size: int = 50,
    ):
        self._async_queue_size = async_queue_size
        self._mode = mode
        self._queue = Queue(maxsize=async_queue_size)
        self._stop_event = threading.Event()

        if tokenizer is not None:
            self._tokenizer = (
                AutoTokenizer.from_pretrained(tokenizer)
                if isinstance(tokenizer, str)
                else tokenizer
            )
            logger.info("Tokenizer initialized: {}", self._tokenizer)
        else:
            self._tokenizer = None
            logger.debug("No tokenizer provided")

        if self._mode == "async":
            self._thread = threading.Thread(target=self._populate_queue)
            self._thread.daemon = True
            self._thread.start()
            logger.info(
                "RequestGenerator started in async mode with queue size: {}",
                self._async_queue_size,
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the RequestGenerator.

        :return: String representation of the RequestGenerator.
        :rtype: str
        """
        return (
            f"RequestGenerator("
            f"mode={self._mode}, "
            f"async_queue_size={self._async_queue_size}, "
            f"tokenizer={self._tokenizer})"
        )

    def __iter__(self) -> Iterator[TextGenerationRequest]:
        """
        Provide an iterator interface to generate new requests.

        :return: An iterator over result requests.
        :rtype: Iterator[TextGenerationRequest]
        """
        if self.mode == "async":
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get_nowait()
                    self._queue.task_done()
                    yield item
                except Empty:
                    continue
        else:
            while not self._stop_event.is_set():
                yield self.create_item()

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """
        Get the tokenizer instance.

        :return: The tokenizer instance.
        :rtype: Optional[PreTrainedTokenizer]
        """
        return self._tokenizer

    @property
    def mode(self) -> str:
        """
        Get the generation mode.

        :return: The generation mode.
        :rtype: str
        """
        return self._mode

    @property
    def async_queue_size(self) -> int:
        """
        Get the size of the request queue.

        :return: The size of the request queue.
        :rtype: int
        """
        return self._async_queue_size

    @abstractmethod
    def create_item(self) -> TextGenerationRequest:
        """
        Abstract method to create a new result request item.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """
        raise NotImplementedError()

    def stop(self):
        """
        Stop the background task that populates the queue.
        """
        logger.info("Stopping RequestGenerator...")
        self._stop_event.set()
        if self._mode == "async":
            self._thread.join()
        logger.info("RequestGenerator stopped")

    def _populate_queue(self):
        """
        Populate the request queue in the background.
        """
        while not self._stop_event.is_set():
            try:
                if self._queue.qsize() < self._async_queue_size:
                    item = self.create_item()
                    self._queue.put(item, timeout=0.1)
                    logger.debug(
                        "Item added to queue. Current queue size: {}",
                        self._queue.qsize(),
                    )
                else:
                    time.sleep(0.1)
            except Full:
                continue
        logger.info("RequestGenerator stopped populating queue")
