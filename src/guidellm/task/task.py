import asyncio
from typing import Callable, Any, Dict, Optional
from loguru import logger

__all__ = ["Task"]


class Task:
    """
    A class representing a general unit of work that can be run either asynchronously or synchronously.

    :param func: The function to be called.
    :type func: Callable
    :param params: The parameters to call the function with.
    :type params: Dict[str, Any]
    """

    def __init__(
        self, func: Callable[..., Any], params: Optional[Dict[str, Any]] = None
    ):
        self.func = func
        self.params = params or {}
        self._cancel_event = asyncio.Event()
        logger.info(
            f"Task created with function: {self.func.__name__} and params: {self.params}"
        )

    async def run_async(self) -> Any:
        """
        Run the task asynchronously.

        :return: The output of the function.
        :rtype: Any
        """
        logger.info(f"Running task asynchronously with function: {self.func.__name__}")
        try:
            result = await asyncio.gather(
                asyncio.to_thread(self.func, **self.params),
                self._check_cancelled(),
                return_exceptions=True,
            )
            if isinstance(result[0], Exception):
                raise result[0]
            if self.is_cancelled():
                logger.warning("Task was cancelled")
                return None
            logger.info(f"Task completed with result: {result[0]}")
            return result[0]
        except asyncio.CancelledError:
            logger.warning("Task was cancelled")
            return None
        except Exception as e:
            logger.error(f"Task failed with error: {e}")
            return None

    def run_sync(self) -> Any:
        """
        Run the task synchronously.

        :return: The output of the function.
        :rtype: Any
        """
        logger.info(f"Running task synchronously with function: {self.func.__name__}")
        try:
            result = self.func(**self.params)
            logger.info(f"Task completed with result: {result}")
            return result
        except Exception as e:
            logger.error(f"Task failed with error: {e}")
            return None

    def cancel(self):
        """
        Cancel the task.
        """
        logger.info("Cancelling task")
        self._cancel_event.set()

    async def _check_cancelled(self):
        """
        Check if the task is cancelled.
        """
        await self._cancel_event.wait()

    def is_cancelled(self) -> bool:
        """
        Check if the task is cancelled.

        :return: True if the task is cancelled, False otherwise.
        :rtype: bool
        """
        return self._cancel_event.is_set()
