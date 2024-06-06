import asyncio
from abc import ABC, abstractmethod
from typing import Iterator
from collections import deque
from loguru import logger
from guidellm.core.request import BenchmarkRequest


class RequestGenerator(ABC):
    """
    A base class for request generators that generate benchmark requests.

    :param queue_size: The size of the request queue.
    :type queue_size: int
    """

    def __init__(self, queue_size: int = 10):
        self.queue_size = queue_size
        self._queue = deque(maxlen=queue_size)
        self._stop_event = asyncio.Event()
        self._populating_task = asyncio.create_task(self._populate_queue())
        logger.info(f"RequestGenerator initialized with queue size: {queue_size}")

    def __iter__(self) -> Iterator[BenchmarkRequest]:
        """
        Provide an iterator interface to generate new requests.

        :return: An iterator over benchmark requests.
        :rtype: Iterator[BenchmarkRequest]
        """
        while not self._stop_event.is_set():
            if self._queue:
                yield self._queue.popleft()
            else:
                asyncio.run(self._populate_queue())

    @abstractmethod
    async def _create_item(self) -> BenchmarkRequest:
        """
        Abstract method to create a new benchmark request item.

        :return: A new benchmark request.
        :rtype: BenchmarkRequest
        """
        raise NotImplementedError()

    async def stop(self):
        """
        Stop the background task that populates the queue.
        """
        logger.info("Stopping RequestGenerator...")
        self._stop_event.set()
        await self._populating_task

    async def _populate_queue(self):
        """
        Populate the request queue in the background.
        """
        while not self._stop_event.is_set():
            while len(self._queue) < self.queue_size:
                item = await self._create_item()
                self._queue.append(item)
                logger.debug(
                    f"Item added to queue. Current queue size: {len(self._queue)}")
            await asyncio.sleep(0.1)