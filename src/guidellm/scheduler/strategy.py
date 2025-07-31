import random
from abc import ABC, abstractmethod
from typing import (
    Literal,
    Optional,
    Union,
)

from pydantic import Field

from guidellm.objects import StandardBaseModel
from guidellm.scheduler.objects import ScheduledRequestInfo

__all__ = [
    "AsyncConstantStrategy",
    "AsyncPoissonStrategy",
    "ConcurrentStrategy",
    "SchedulingStrategy",
    "StrategyType",
    "SynchronousStrategy",
    "ThroughputStrategy",
    "strategy_display_str",
]


StrategyType = Literal["synchronous", "concurrent", "throughput", "constant", "poisson"]


class ScheduledRequestTimings(StandardBaseModel, ABC):
    @abstractmethod
    def next_offset(self) -> float:
        pass

    @abstractmethod
    def request_completed(self, request_info: ScheduledRequestInfo):
        pass


class LastCompletionRequestTimings(ScheduledRequestTimings):
    """
    Implementation of ScheduledRequestTimings meant for synchronous or concurrent
    scheduling strategies where the next request is scheduled immediately after
    the last received request is completed.
    """

    def __init__(self):
        self.offset = 0.0
        self.concurrency = 0

    def next_offset(self) -> float:
        self.concurrency += 1

        return self.offset

    def request_completed(self, request_info: ScheduledRequestInfo):
        self.concurrency -= 1

        if request_info.completed_at is not None:
            # set the next sync offset to the time when the previous request completed
            self.offset = request_info.completed_at - request_info.scheduler_start_time


class NoDelayRequestTimings(ScheduledRequestTimings):
    """
    Implementation of ScheduledRequestTimings for throughput scheduling strategies
    enabling a worker process to schedule requests as close to the start time
    as possible and measuring any delays based on closeness to the start time.
    """

    def next_offset(self) -> float:
        return 0.0

    def request_completed(self, request_info: ScheduledRequestInfo):
        pass


class ConstantRateRequestTimings(ScheduledRequestTimings):
    """
    Implementation of ScheduledRequestTimings for constant rate scheduling strategies
    enabling a worker process to schedule requests at a constant rate.
    The rate is defined as requests per second, and the offset is calculated
    based on the rate.
    """

    def __init__(self, rate: float):
        self.rate = rate
        self.offset_count = 0

    def next_offset(self) -> float:
        interval = 1.0 / self.rate
        self.offset_count += 1

        return interval * self.offset_count

    def request_completed(self, request_info: ScheduledRequestInfo):
        pass


class PoissonRateRequestTimings(ScheduledRequestTimings):
    """
    Implementation of ScheduledRequestTimings for Poisson rate scheduling strategies
    enabling a worker process to schedule requests at a Poisson rate.
    The rate is defined as requests per second, and the offset is calculated
    based on the Poisson distribution.
    """

    def __init__(self, rate: float, random_seed: int = 42):
        self.rate = rate
        self.offset = 0.0
        self.random = random.Random(random_seed)

    def next_offset(self) -> float:
        next_delay = self.random.expovariate(self.rate)
        self.offset += next_delay

        return self.offset

    def request_completed(self, request_info: ScheduledRequestInfo):
        pass


class SchedulingStrategy(StandardBaseModel):
    """
    An abstract base class for scheduling strategies enabling control over how
    requests are processed by the scheduler.

    :param type_: The type of scheduling strategy to use.
        This should be one of the predefined strategy types.
    """

    type_: Literal["strategy"] = Field(
        description="The type of scheduling strategy to schedule requests with.",
    )

    @property
    def processes_limit(self) -> Optional[int]:
        """
        :return: The maximum number of worker processes supported by the
            scheduling strategy. None if not limited.
        """
        return None

    @property
    def requests_limit(self) -> Optional[int]:
        """
        :return: The maximum number of concurrent requests that can be processed
            at once by the scheduling strategy. None if not limited.
        """
        return None

    def create_worker_timings(
        self, local_rank: int, local_world_size: int
    ) -> ScheduledRequestTimings:
        """
        Create a ScheduledRequestTimings instance to define the timing behavior
        for the worker process to schedule requests.

        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :return: A ScheduledRequestTimings instance for the worker process.
        """
        raise NotImplementedError(
            "create_worker_timings method must be implemented by subclasses."
        )


class SynchronousStrategy(SchedulingStrategy):
    """
    A class representing a synchronous scheduling strategy.
    This strategy schedules requests synchronously with the maximum rate possible,
    meaning that only one request is processed at a time.
    """

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    @property
    def processes_limit(self) -> Optional[int]:
        """
        :return: 1 for the synchronous scheduling strategy to limit
            the worker processes to handle just one request at a time.
        """
        return 1

    @property
    def requests_limit(self) -> Optional[int]:
        """
        :return: 1 for the synchronous scheduling strategy to limit
            the requests to just one request at a time.
        """
        return 1

    def create_worker_timings(
        self, local_rank: int, local_world_size: int
    ) -> ScheduledRequestTimings:
        """
        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :return: A request timings instance for synchronous scheduling.
        """
        if local_world_size > 1 or local_rank != 0:
            raise ValueError(
                "SynchronousStrategy can only be used with a single worker process."
            )

        return LastCompletionRequestTimings()


class ConcurrentStrategy(SchedulingStrategy):
    """
    A class representing a concurrent scheduling strategy.
    This strategy schedules requests concurrently with the maximum rate possible,
    meaning that multiple requests up to the number of streams are processed at a time.
    """

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: int = Field(
        description=(
            "The number of concurrent streams to use for scheduling requests. "
            "This must be a positive integer."
        ),
        gt=0,
    )

    @property
    def processes_limit(self) -> int:
        """
        :return: {self.streams} for the concurrent scheduling strategy to limit
            the maximum number of worker processes to the number of streams.
        """
        return self.streams

    @property
    def requests_limit(self) -> int:
        """
        :return: {self.streams} for the concurrent scheduling strategy to limit
            the maximum number of concurrent requests to the number of streams.
        """
        return self.streams

    def create_worker_timings(
        self, local_rank: int, local_world_size: int
    ) -> LastCompletionRequestTimings:
        """
        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :return: A request timings instance for concurrent scheduling.
        """
        if local_world_size > self.streams:
            raise ValueError(
                "ConcurrentStrategy can only be used with up to "
                f"{self.streams} worker processes."
            )

        if local_rank >= self.streams:
            raise ValueError(
                f"Local rank {local_rank} exceeds the number of streams {self.streams}."
            )

        return LastCompletionRequestTimings()


class ThroughputStrategy(SchedulingStrategy):
    """
    A class representing a throughput scheduling strategy.
    This strategy schedules requests to maximize throughput without any limits
    on concurrency or processes, allowing the system to process as many requests
    as possible simultaneously.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: Optional[int] = Field(
        default=None,
        description=(
            "The maximum number of concurrent requests to schedule. "
            "This must be a positive integer greater than 0."
        ),
        gt=0,
    )

    @property
    def processes_limit(self) -> Optional[int]:
        """
        :return: max_concurrency if set, otherwise None for the throughput
            scheduling strategy to allow unlimited worker processes.
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> Optional[int]:
        """
        :return: max_concurrency if set, otherwise None for the throughput
            scheduling strategy to allow unlimited concurrent requests.
        """
        return self.max_concurrency

    def create_worker_timings(
        self, local_rank: int, local_world_size: int
    ) -> ScheduledRequestTimings:
        """
        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :return: A NoDelayRequestTimings instance for throughput scheduling.
        """
        return NoDelayRequestTimings()


class AsyncConstantStrategy(ThroughputStrategy):
    """
    A class representing an asynchronous constant rate scheduling strategy.
    This strategy schedules requests asynchronously at a constant rate
    in requests per second, with the rate spread evenly across worker processes.
    It inherits from the `ThroughputStrategy` base class which defines the
    process and request limits, and utilizes the `ConstantRateRequestTimings`
    to manage the timing of requests.
    """

    type_: Literal["constant"] = "constant"  # type: ignore[assignment]
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )

    def create_worker_timings(
        self, local_rank: int, local_world_size: int
    ) -> ScheduledRequestTimings:
        """
        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :return: A ConstantRateRequestTimings instance for constant rate scheduling
            with the rate divided evenly across worker processes.
        """
        # Divide the rate evenly across all worker processes
        worker_rate = self.rate / local_world_size
        return ConstantRateRequestTimings(worker_rate)


class AsyncPoissonStrategy(ThroughputStrategy):
    """
    A class representing an asynchronous Poisson rate scheduling strategy.
    This strategy schedules requests asynchronously at a Poisson rate
    in requests per second, with the rate spread evenly across worker processes.
    It inherits from the `ThroughputStrategy` base class which defines the
    process and request limits, and utilizes the `PoissonRateRequestTimings`
    to manage the timing of requests.
    """

    type_: Literal["poisson"] = "poisson"  # type: ignore[assignment]
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    random_seed: int = Field(
        default=42,
        description=("The random seed to use for the Poisson distribution."),
    )

    def create_worker_timings(
        self, local_rank: int, local_world_size: int
    ) -> ScheduledRequestTimings:
        """
        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :return: A PoissonRateRequestTimings instance for Poisson rate scheduling
            with the rate divided evenly across worker processes.
        """
        # Divide the rate evenly across all worker processes
        worker_rate = self.rate / local_world_size
        # Use a different seed for each worker to ensure different sequences
        worker_seed = self.random_seed + local_rank
        return PoissonRateRequestTimings(worker_rate, worker_seed)


def strategy_display_str(strategy: Union[StrategyType, SchedulingStrategy]) -> str:
    strategy_type = strategy if isinstance(strategy, str) else strategy.type_
    strategy_instance = strategy if isinstance(strategy, SchedulingStrategy) else None

    if strategy_type == "concurrent":
        rate = f"@{strategy_instance.streams}" if strategy_instance else "@##"  # type: ignore[attr-defined]
    elif strategy_type in ("constant", "poisson"):
        rate = f"@{strategy_instance.rate:.2f}" if strategy_instance else "@#.##"  # type: ignore[attr-defined]
    else:
        rate = ""

    return f"{strategy_type}{rate}"
