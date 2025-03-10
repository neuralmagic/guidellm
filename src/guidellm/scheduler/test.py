import asyncio
import uuid

from guidellm.backend.openai import OpenAIHTTPBackend
from guidellm.scheduler import Scheduler
from guidellm.scheduler.backend_worker import BackendRequestsWorker, GenerationRequest


def test_scheduler():
    backend = OpenAIHTTPBackend(target="http://192.168.4.13:8000")
    backend.validate()
    worker = BackendRequestsWorker(
        backend=backend,
    )
    request_loader = [
        GenerationRequest(
            request_id=str(uuid.uuid4()),
            request_type="text",
            content="Create a test prompt for LLMs: ",
            constraints={"output_tokens": 256},
        )
        for _ in range(1000)
    ]
    scheduler = Scheduler(
        worker=worker,
        request_loader=request_loader,
        scheduling_strategy="throughput",
    )

    async def _run_scheduler():
        async for result in scheduler.run():
            print(
                f"<processes>: {result.run_info.processes} <queued>: {result.run_info.queued_requests} <pending>: {result.run_info.pending_requests} <completed>: {result.run_info.completed_requests}"
            )

    asyncio.run(_run_scheduler())


if __name__ == "__main__":
    test_scheduler()
