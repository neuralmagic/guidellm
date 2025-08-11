import asyncio
import contextlib
import time
from collections.abc import Iterable
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from threading import Barrier as ThreadingBarrier
from threading import Event as ThreadingEvent
from typing import Any, Callable, Literal, Optional, Union

__all__ = ["synchronous_to_exitable_async"]


async def synchronous_to_exitable_async(
    synchronous: Optional[Union[Iterable, Callable]],
    exit_events: Optional[dict[str, Union[ThreadingEvent, ProcessingEvent]]] = None,
    exit_barrier: Optional[Union[ThreadingBarrier, ProcessingBarrier]] = None,
    poll_interval: float = 0.1,
    *args,
    **kwargs,
) -> tuple[Union[Literal["completed", "canceled", "barrier"], str], Any]:
    """
    Convert synchronous iterable to async execution with lifecycle management.

    Enables synchronous iterables to execute within async contexts while respecting
    multiprocessing synchronization primitives like barriers and events.

    :param iter_func: Iterable function to execute, or "infinite" for polling.
    :param exit_events: Optional event mappings for monitoring termination signals.
    :param exit_barrier: Optional barrier for synchronization before exit.
    :param poll_interval: Time between iteration cycles and event checks.
    :param args: Positional arguments passed to iter_func.
    :param kwargs: Keyword arguments passed to iter_func.
    :return: Tuple of (exit_reason, last_item) from iterator termination.
    :raises RuntimeError: If error event is detected during iteration.
    :raises asyncio.CancelledError: If the async operation is cancelled.
    """

    if exit_events is None:
        exit_events = {}

    canceled_event = ThreadingEvent()
    barrier_event = ThreadingEvent()
    events_list = [("canceled", canceled_event), ("barrier", barrier_event)] + list(
        exit_events.items()
    )

    def _watch_barrier_thread():
        try:
            exit_barrier.wait()
            barrier_event.set()
        except Exception:
            barrier_event.set()

    def _check_event_set() -> Optional[str]:
        for name, event in events_list:
            if event.is_set():
                return name

        return None

    def _run_thread():
        finish_reason = "completed"
        return_val = None

        try:
            while (check_event := _check_event_set()) is not None:
                if isinstance(synchronous, Callable):
                    return_val = synchronous(*args, **kwargs)
                    break

                if isinstance(synchronous, Iterable):
                    try:
                        return_val = next(synchronous)
                    except StopIteration:
                        break

                time.sleep(poll_interval)

            if synchronous is not None:
                # Call again to ensure nothing was set while callable executing
                check_event = _check_event_set()

            if check_event is not None:
                finish_reason = check_event
        finally:
            if exit_barrier is not None:
                exit_barrier.abort()

        return finish_reason, return_val

    try:
        if exit_barrier is not None:
            asyncio.to_thread(_watch_barrier_thread)
        return await asyncio.to_thread(_run_thread)
    except asyncio.CancelledError:
        if exit_barrier is not None:
            with contextlib.suppress(Exception):
                exit_barrier.abort()
        canceled_event.set()
        raise
