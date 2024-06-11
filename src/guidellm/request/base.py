import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Union, Iterator
from transformers import AutoTokenizer, PreTrainedTokenizer
from loguru import logger
from guidellm.core.request import BenchmarkRequest


class RequestGenerator(ABC):
    """
    A base class for request generators that generate benchmark requests.

    :param tokenizer: The tokenizer instance or the name/config to use for tokenizing prompts.
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
        self._queue = asyncio.Queue(maxsize=async_queue_size)
        self._stop_event = asyncio.Event()
        self._populating_task = None

        if self.mode == "async":
            self._populating_task = asyncio.create_task(self._populate_queue())
            logger.info(
                f"RequestGenerator initialized in async mode with queue size: {async_queue_size}"
            )
        else:
            logger.info("RequestGenerator initialized in sync mode")

        if tokenizer is not None:
            self._tokenizer = (
                AutoTokenizer.from_pretrained(tokenizer)
                if isinstance(tokenizer, str)
                else tokenizer
            )
            logger.info(f"Tokenizer initialized: {self._tokenizer}")
        else:
            self._tokenizer = None
            logger.debug("No tokenizer provided")

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """
        Get the tokenizer instance.

        :return: The tokenizer instance.
        """
        return self._tokenizer

    @property
    def mode(self) -> str:
        """
        Get the generation mode.

        :return: The generation mode.
        """
        return self._mode

    @property
    def async_queue_size(self) -> int:
        """
        Get the size of the request queue.

        :return: The size of the request queue.
        """
        return self._async_queue_size

    def __iter__(self) -> Iterator[BenchmarkRequest]:
        """
        Provide an iterator interface to generate new requests.

        :return: An iterator over benchmark requests.
        :rtype: Iterator[BenchmarkRequest]
        """
        if self.mode == "async":
            while not self._stop_event.is_set():
                try:
                    item = asyncio.run(self._queue.get())
                    self._queue.task_done()
                    yield item
                except asyncio.CancelledError:
                    break
        else:
            while not self._stop_event.is_set():
                yield self.create_item()

    @abstractmethod
    def create_item(self) -> BenchmarkRequest:
        """
        Abstract method to create a new benchmark request item.

        :return: A new benchmark request.
        :rtype: BenchmarkRequest
        """
        raise NotImplementedError()

    def start(self):
        """
        Start the background task that populates the queue.
        """
        if self._populating_task is not None:
            logger.warning("RequestGenerator is already running")
            return

        logger.info("Starting RequestGenerator...")
        self._populating_task = asyncio.create_task(self._populate_queue())

    def stop(self):
        """
        Stop the background task that populates the queue.
        """
        logger.info("Stopping RequestGenerator...")
        self._stop_event.set()

    async def _populate_queue(self):
        """
        Populate the request queue in the background.
        """
        while not self._stop_event.is_set():
            if self._queue.qsize() < self._async_queue_size:
                item = self.create_item()
                await self._queue.put(item)
                logger.debug(
                    f"Item added to queue. Current queue size: {self._queue.qsize()}"
                )
            else:
                await asyncio.sleep(0.1)
