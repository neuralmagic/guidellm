import asyncio
import multiprocessing
import time
import logging
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    datefmt='%H:%M:%S'
)

# A multiprocessing queue that will remain empty
# Naming it mp_queue to distinguish from asyncio.Queue
mp_queue = multiprocessing.Queue()


async def get_item_from_mp_queue(q: multiprocessing.Queue, worker_id: int):
    """
    Coroutine that tries to get an item from a multiprocessing.Queue
    using asyncio.to_thread.
    """
    logging.info(f"Worker {worker_id}: get_item_from_mp_queue: ENTERED. Awaiting asyncio.to_thread(q.get).")
    try:
        # This is the blocking call in a separate thread
        item = await asyncio.to_thread(q.get)
        # We don't expect this to be reached if the queue is empty
        logging.info(
            f"Worker {worker_id}: get_item_from_mp_queue: asyncio.to_thread RETURNED NORMALLY with item: {item}.")
        return item
    except asyncio.CancelledError:
        # This is where it SHOULD go if the task awaiting this coroutine is cancelled,
        # and asyncio.to_thread correctly propagates the cancellation to its awaiter.
        logging.error(
            f"Worker {worker_id}: get_item_from_mp_queue: CAUGHT CancelledError from asyncio.to_thread directly!")
        raise  # Re-raise to propagate the cancellation
    except Exception as e:
        logging.error(f"Worker {worker_id}: get_item_from_mp_queue: CAUGHT an UNEXPECTED EXCEPTION {type(e)}: {e}",
                      exc_info=True)
        raise
    finally:
        # This finally block will execute. The key is whether the CancelledError was caught above.
        logging.info(f"Worker {worker_id}: get_item_from_mp_queue: EXITED (finally block).")


async def worker_coroutine(worker_id: int, q: multiprocessing.Queue):
    """
    The main coroutine for our worker task. It will try to get an item
    from the queue.
    """
    logging.info(f"Worker {worker_id}: worker_coroutine: STARTED.")
    try:
        logging.info(f"Worker {worker_id}: worker_coroutine: About to await get_item_from_mp_queue.")
        # This is the await point where CancelledError should be injected
        # if this worker_coroutine task is cancelled.
        await get_item_from_mp_queue(q, worker_id)
        logging.info(f"Worker {worker_id}: worker_coroutine: get_item_from_mp_queue completed (unexpectedly).")
    except asyncio.CancelledError:
        logging.error(f"Worker {worker_id}: worker_coroutine: SUCCESSFULLY CAUGHT CancelledError.")
        # Perform any task-specific cleanup here if needed
    except Exception as e:
        logging.error(f"Worker {worker_id}: worker_coroutine: CAUGHT UNEXPECTED EXCEPTION {type(e)}: {e}",
                      exc_info=True)
    finally:
        logging.info(f"Worker {worker_id}: worker_coroutine: FINISHED (finally block).")


async def main_orchestrator():
    """
    Orchestrates the test: creates, runs, and cancels the worker.
    """
    logging.info("Main Orchestrator: Starting worker task.")
    worker_task = asyncio.create_task(worker_coroutine(1, mp_queue), name="WorkerCoroutine-1")

    # Give the worker task a moment to start and block on the queue
    logging.info("Main Orchestrator: Sleeping for 1 second to let worker block...")
    await asyncio.sleep(1)

    logging.info(f"Main Orchestrator: Current active threads: {[t.name for t_ in threading.enumerate()]}...")

    # Cancel the worker task
    print("Main Orchestrator: Cancelling worker_task...")
    worker_task.cancel()

    # Wait for the worker task to finish, with a timeout.
    # If cancellation works as expected, worker_task should complete (by handling CancelledError)
    # well before the timeout.
    # If it gets stuck, asyncio.TimeoutError will be raised.
    timeout_seconds = 5.0
    logging.info(f"Main Orchestrator: Awaiting worker_task with timeout {timeout_seconds}s...")
    try:
        await asyncio.wait_for(worker_task, timeout=timeout_seconds)
        logging.info("Main Orchestrator: worker_task completed WITHOUT timeout.")
    except asyncio.TimeoutError:
        logging.error(
            f"Main Orchestrator: TIMEOUT! worker_task did not finish within {timeout_seconds}s after cancellation.")
        logging.error(
            f"Main Orchestrator: worker_task.done() = {worker_task.done()}, worker_task.cancelled() = {worker_task.cancelled()}")
        # At this point, the thread running mp_queue.get() is likely still blocked.
    except asyncio.CancelledError:
        # This would happen if main_orchestrator itself was cancelled, not expected here.
        logging.error("Main Orchestrator: main_orchestrator itself was cancelled (unexpected).")
    except Exception as e:
        logging.error(f"Main Orchestrator: An unexpected error occurred while waiting for worker_task: {e}",
                      exc_info=True)
    finally:
        logging.info("Main Orchestrator: Test finished.")
        # Note: The thread started by asyncio.to_thread for mp_queue.get()
        # might still be alive and blocked if q.get() wasn't unblocked.
        # It's a daemon thread by default, so it won't prevent program exit.
        # To clean it up, one would typically put a sentinel into mp_queue.
        # For this test, we are focused on the asyncio task cancellation.
        logging.info(
            f"Main Orchestrator: Final check: worker_task.done() = {worker_task.done()}, worker_task.cancelled() = {worker_task.cancelled()}")

        # Attempt to unblock the queue to allow the thread to exit,
        # though the test's focus is on the asyncio cancellation.
        try:
            mp_queue.put_nowait(None)  # Sentinel
            logging.info("Main Orchestrator: Put sentinel in mp_queue to unblock thread.")
        except Exception:
            logging.warning("Main Orchestrator: Could not put sentinel in mp_queue.")


if __name__ == "__main__":
    # For multiprocessing queues to work correctly, especially on Windows/macOS
    # with 'spawn' or 'forkserver' start methods, it's good practice
    # to ensure the queue is created in the main process scope before tasks.
    # In this simple script, it's fine.
    try:
        asyncio.run(main_orchestrator())
    except KeyboardInterrupt:
        logging.info("Main Orchestrator: Keyboard interrupt received.")
    finally:
        mp_queue.close()
        mp_queue.join_thread()  # Ensure queue's feeder thread is joined
        logging.info("Main Orchestrator: mp_queue resources released.")
