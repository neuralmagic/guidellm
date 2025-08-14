from __future__ import annotations

import asyncio
import random
import uuid
from collections import defaultdict
from typing import Any

import pytest
from pydantic import BaseModel, Field

from guidellm.scheduler import (
    BackendInterface,
    ConstraintInitializer,
    Environment,
    MaxNumberConstraintInitializer,
    NonDistributedEnvironment,
    ScheduledRequestInfo,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
    SynchronousStrategy,
)


class MockRequest(BaseModel):
    payload: str
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))


class MockBackend(BackendInterface):
    """Mock backend for integration testing with predictable responses."""

    def __init__(
        self,
        processes_limit_value: int | None = None,
        requests_limit_value: int | None = None,
        error_rate: float = 0.2,
        response_delay: float = 0.0,
    ):
        self._processes_limit = processes_limit_value
        self._requests_limit = requests_limit_value
        self._error_rate = error_rate
        self._response_delay = response_delay

    @property
    def processes_limit(self) -> int | None:
        return self._processes_limit

    @property
    def requests_limit(self) -> int | None:
        return self._requests_limit

    def info(self) -> dict[str, Any]:
        return {"type": "mock_integration", "delay": self._response_delay}

    async def process_startup(self):
        pass

    async def validate(self):
        pass

    async def process_shutdown(self):
        pass

    async def resolve(self, request: MockRequest, request_info, request_history):
        """Return predictable response based on input request."""
        # Simulate processing time
        await asyncio.sleep(self._response_delay)

        if (
            self._error_rate
            and self._error_rate > 0
            and random.random() < self._error_rate
        ):
            raise RuntimeError(f"mock_error_for_{request.payload}")

        yield f"response_for_{request.payload}"


@pytest.mark.smoke
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy", "env", "constraint_inits"),
    [
        (
            SynchronousStrategy(),
            NonDistributedEnvironment(),
            {"max_number": MaxNumberConstraintInitializer(max_num=100)},
        ),
    ],
)
async def test_scheduler_run(
    strategy: SchedulingStrategy,
    env: Environment,
    constraint_inits: dict[str, ConstraintInitializer],
):
    scheduler = Scheduler()
    constraints = {
        key: init.create_constraint() for key, init in constraint_inits.items()
    }
    received_updates = defaultdict(list)
    received_responses = []
    last_state = None
    num_requests = 50

    async for resp, req, info, state in scheduler.run(
        requests=[
            MockRequest(payload=f"req_{ind}") for ind in range(num_requests)
        ],  # less than total requests sent to test new request iter
        backend=MockBackend(),
        strategy=strategy,
        env=env,
        **constraints,
    ):
        assert req is not None
        assert isinstance(req, MockRequest)
        assert isinstance(info, ScheduledRequestInfo)
        assert info.status != "cancelled"
        assert isinstance(state, SchedulerState)
        if info.status == "completed":
            assert resp == f"response_for_{req.payload}"
            received_responses.append(resp)
        elif info.status == "errored":
            assert resp is None
            assert info.error is not None
            assert info.error == f"mock_error_for_{req.payload}"
            received_responses.append(info.error)

        if len(received_updates[req.payload]) < 3:
            received_updates[req.payload].append(info.status)
        last_state = state

    assert len(received_updates) == num_requests
    assert len(received_responses) == constraints["max_number"].max_num
    assert last_state.created_requests == constraints["max_number"].max_num
    assert last_state.queued_requests == 0
    assert last_state.processing_requests == 0
    assert last_state.processed_requests == constraints["max_number"].max_num
    assert last_state.cancelled_requests == 0
    assert (
        last_state.successful_requests + last_state.errored_requests
    ) == constraints["max_number"].max_num

    def _request_indices():
        while True:
            yield from range(num_requests)

    for index, req, statuses, resp in zip(
        _request_indices(),
        received_updates.keys(),
        received_updates.values(),
        received_responses,
    ):
        assert req == f"req_{index}"
        assert resp in (f"response_for_{req}", f"mock_error_for_{req}")
        assert statuses in (
            ["queued", "in_progress", "completed"],
            ["queued", "in_progress", "errored"],
        )
