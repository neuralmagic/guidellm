import asyncio
import uuid

from guidellm.backend.openai import OpenAIHTTPBackend
from guidellm.scheduler.backend_worker import BackendRequestsWorker, GenerationRequest
from guidellm.scheduler.scheduler import Scheduler
from guidellm.scheduler.strategy import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    StrategyType,
)


def test_scheduler(strategy_type: StrategyType, output_tokens: int = 256):
    if strategy_type == "synchronous":
        num_requests = 20
        strategy = "synchronous"
    elif strategy_type == "concurrent":
        num_requests = 100
        strategy = ConcurrentStrategy(
            streams=6,
        )
    elif strategy_type == "throughput":
        num_requests = 1000
        strategy = "throughput"
    elif strategy_type == "constant":
        num_requests = 100
        strategy = AsyncConstantStrategy(
            rate=5.5,
        )
    elif strategy_type == "poisson":
        num_requests = 100
        strategy = AsyncPoissonStrategy(
            rate=5.5,
        )
    else:
        raise ValueError(f"Invalid strategy type: {strategy_type}")

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
            constraints={"output_tokens": output_tokens},
        )
        for _ in range(num_requests)
    ]
    scheduler = Scheduler(
        worker=worker,
        request_loader=request_loader,
        scheduling_strategy=strategy,
    )

    async def _run_scheduler():
        async for result in scheduler.run():
            print(
                f"<processes>: {result.run_info.processes} <queued>: {result.run_info.queued_requests} <pending>: {result.run_info.pending_requests} <completed>: {result.run_info.completed_requests}"
            )

    asyncio.run(_run_scheduler())


if __name__ == "__main__":
    test_scheduler("poisson")
