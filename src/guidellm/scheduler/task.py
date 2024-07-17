import asyncio
import functools
from typing import Any, Callable, Dict, Optional

from loguru import logger

__all__ = ["Task"]


class Task:
    """
    A class representing a general unit of work that can be run
    either asynchronously or synchronously.

    :param func: The function to be called.
    :type func: Callable
    :param params: The parameters to call the function with.
    :type params: Dict[str, Any]
    :param err_container: The container to return errors in if the task fails.
    :type err_container: Optional[Callable]
    """

    def __init__(
        self,
        func: Callable[..., Any],
        params: Optional[Dict[str, Any]] = None,
        err_container: Optional[Callable] = None,
    ):
        self._func: Callable[..., Any] = func
        self._params: Dict[str, Any] = params or {}
        self._err_container: Optional[Callable] = err_container
        self._cancel_event: asyncio.Event = asyncio.Event()

        logger.info(
            f"Task created with function: {self._func.__name__} and "
            f"params: {self._params}"
        )

    async def run_async(self, event_loop: asyncio.AbstractEventLoop) -> Any:
        """
        Run the task asynchronously.

        :return: The output of the function.
        :rtype: Any
        """

        logger.info(f"Running task asynchronously with function: {self._func.__name__}")

        try:
            result = await event_loop.run_in_executor(
                None, functools.partial(self._func, **self._params)
            )
            if isinstance(result, Exception):
                raise result

            logger.info(f"Task completed with result: {result}")

            return result
        except Exception as err:
            logger.error(f"Task failed with error: {err}")
            return (
                err
                if not self._err_container
                else self._err_container(**self._params, error=err)
            )

    def run_sync(self) -> Any:
        """
        Run the task synchronously.

        :return: The output of the function.
        :rtype: Any
        """
        logger.info(f"Running task synchronously with function: {self._func.__name__}")
        try:
            result = self._func(**self._params)
            logger.info(f"Task completed with result: {result}")
            return result
        except Exception as err:
            logger.error(f"Task failed with error: {err}")
            return (
                err
                if not self._err_container
                else self._err_container(**self._params, error=err)
            )
