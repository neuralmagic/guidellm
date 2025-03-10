import math
import random
import time
from abc import ABC, abstractmethod
from typing import (
    Generator,
    Literal,
    Optional,
)

from pydantic import BaseModel, Field

__all__ = [
    "StrategyType",
    "SchedulingStrategy",
    "SynchronousStrategy",
    "ConcurrentStrategy",
    "ThroughputStrategy",
    "AsyncConstantStrategy",
    "AsyncPoissonStrategy",
]


StrategyType = Literal["synchronous", "concurrent", "throughput", "constant", "poisson"]


class SchedulingStrategy(ABC, BaseModel):
    """
    An abstract base class for scheduling strategies.
    This class defines the interface for scheduling requests and provides
    a common structure for all scheduling strategies.
    Subclasses should implement the `request_times` method to provide
    specific scheduling behavior.

    :param type_: The type of scheduling strategy to use.
        This should be one of the predefined strategy types.
    """

    type_: StrategyType = Field(
        description="The type of scheduling strategy schedule requests with.",
    )

    @property
    @abstractmethod
    def processing_mode(self) -> Literal["sync", "async"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.
        This property should be implemented by subclasses to return
        the appropriate processing mode.

        :return: The processing mode for the scheduling strategy,
            either 'sync' or 'async'.
        """
        ...

    @property
    @abstractmethod
    def processes_limit(self) -> Optional[int]:
        """
        The limit on the number of worker processes for the scheduling strategy
        or None if the strategy does not restrict the number of processes.
        This property determines how many worker processes are created
        for the scheduling strategy. This property should be implemented
        by subclasses to return the appropriate number of processes.

        :return: The number of processes for the scheduling strategy
            or None if the strategy does not restrict the number of processes.
        """
        ...

    @property
    @abstractmethod
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy or None
        if the strategy does not restrict the number of queued requests.
        This property determines how many requests can be queued at one time
        for the scheduling strategy. This property should be implemented
        by subclasses to return the appropriate number of queued requests.

        :return: The maximum number of queued requests for the scheduling strategy
            or None if the strategy does not restrict the number of queued requests.
        """
        ...

    @property
    @abstractmethod
    def processing_requests_limit(self) -> Optional[int]:
        """
        The maximum number of processing requests for the scheduling strategy
        or None if the strategy does not restrict the number of processing requests.
        This property determines how many requests can be processed at one time
        for the scheduling strategy. This property should be implemented
        by subclasses to return the appropriate number of processing requests.

        :return: The maximum number of processing requests for the scheduling strategy
            or None if the strategy does not restrict the number of processing requests.
        """
        ...

    @abstractmethod
    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields timestamps for when requests should be sent.
        This method should be implemented by subclasses to provide specific
        scheduling behavior.

        :return: A generator that yields timestamps for request scheduling
            or -1 for requests that should be sent immediately.
        """
        ...


class SynchronousStrategy(SchedulingStrategy):
    """
    A class representing a synchronous scheduling strategy.
    This strategy schedules requests synchronously, one at a time,
    with the maximum rate possible.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for synchronous scheduling.

    :param type_: The synchronous StrategyType to schedule requests synchronously.
    """

    type_: Literal["synchronous"] = "synchronous"

    @property
    def processing_mode(self) -> Literal["sync"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'sync' for synchronous scheduling strategy
            for the single worker process.
        """
        return "sync"

    @property
    def processes_limit(self) -> Optional[int]:
        """
        The limit on the number of worker processes for the scheduling strategy
        or None if the strategy does not restrict the number of processes.
        This property determines how many worker processes are created
        for the scheduling strategy.

        :return: 1 for the synchronous scheduling strategy to limit
            the worker processes to one.
        """
        return 1

    @property
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy or None
        if the strategy does not restrict the number of queued requests.
        This property determines how many requests can be queued at one time
        for the scheduling strategy.

        :return: 1 for the synchronous scheduling strategy to limit
            the queued requests to one that is ready to be processed.
        """
        return 1

    @property
    def processing_requests_limit(self) -> Optional[int]:
        """
        The maximum number of processing requests for the scheduling strategy
        or None if the strategy does not restrict the number of processing requests.
        This property determines how many requests can be processed at one time
        for the scheduling strategy.

        :return: 1 for the synchronous scheduling strategy to limit
            the processing requests to one that is ready to be processed.
        """
        return 1

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields time.time() so requests are sent immediately,
            while scheduling them synchronously.

        :return: A generator that yields time.time() for immediate request scheduling.
        """
        while True:
            yield time.time()


class ConcurrentStrategy(SchedulingStrategy):
    """
    A class representing a concurrent scheduling strategy.
    This strategy schedules requests concurrently with the specified
    number of streams.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for concurrent scheduling.

    :param type_: The concurrent StrategyType to schedule requests concurrently.
    :param streams: The number of concurrent streams to use for scheduling requests.
        Each stream runs synchronously with the maximum rate possible.
        This must be a positive integer.
    """

    type_: Literal["concurrent"] = "concurrent"
    streams: int = Field(
        description=(
            "The number of concurrent streams to use for scheduling requests. "
            "Each stream runs sychronously with the maximum rate possible. "
            "This must be a positive integer."
        ),
        gt=0,
    )

    @property
    def processing_mode(self) -> Literal["sync"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'sync' for synchronous scheduling strategy
            for the multiple worker processes equal to streams.
        """
        return "sync"

    @property
    def processes_limit(self) -> Optional[int]:
        """
        The limit on the number of worker processes for the scheduling strategy
        or None if the strategy does not restrict the number of processes.
        This property determines how many worker processes are created
        for the scheduling strategy.

        :return: {self.streams} for the concurrent scheduling strategy to limit
            the worker processes to the number of streams.
        """
        return self.streams

    @property
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy or None
        if the strategy does not restrict the number of queued requests.
        This property determines how many requests can be queued at one time
        for the scheduling strategy.

        :return: {self.streams} for the concurrent scheduling strategy to limit
            the queued requests to the number of streams that are ready to be processed.
        """
        return self.streams

    @property
    def processing_requests_limit(self) -> Optional[int]:
        """
        The maximum number of processing requests for the scheduling strategy
        or None if the strategy does not restrict the number of processing requests.
        This property determines how many requests can be processed at one time
        for the scheduling strategy.

        :return: {self.streams} for the concurrent scheduling strategy to limit
            the processing requests to the number of streams that ready to be processed.
        """
        return self.streams

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields time.time() so requests are sent
        immediately, while scheduling them concurrently with the specified
        number of streams.

        :return: A generator that yields time.time() for immediate request scheduling.
        """
        while True:
            yield time.time()


class ThroughputStrategy(SchedulingStrategy):
    """
    A class representing a throughput scheduling strategy.
    This strategy schedules as many requests asynchronously as possible,
    with the maximum rate possible.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for throughput scheduling.

    :param type_: The throughput StrategyType to schedule requests asynchronously.
    """

    type_: Literal["throughput"] = "throughput"
    max_concurrency: Optional[int] = Field(
        default=None,
        description=(
            "The maximum number of concurrent requests to schedule. "
            "If set to None, the concurrency value from settings will be used. "
            "This must be a positive integer greater than 0."
        ),
        gt=0,
    )

    @property
    def processing_mode(self) -> Literal["async"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'async' for asynchronous scheduling strategy
            for the multiple worker processes handling requests.
        """
        return "async"

    @property
    def processes_limit(self) -> Optional[int]:
        """
        The limit on the number of worker processes for the scheduling strategy
        or None if the strategy does not restrict the number of processes.
        This property determines how many worker processes are created
        for the scheduling strategy.

        :return: None for the throughput scheduling strategy to apply
            no limit on the number of processes.
        """
        return None

    @property
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy or None
        if the strategy does not restrict the number of queued requests.
        This property determines how many requests can be queued at one time
        for the scheduling strategy.

        :return: None for the throughput scheduling strategy to apply
            no limit on the number of queued requests.
        """
        return None

    @property
    def processing_requests_limit(self) -> Optional[int]:
        """
        The maximum number of processing requests for the scheduling strategy
        or None if the strategy does not restrict the number of processing requests.
        This property determines how many requests can be processed at one time
        for the scheduling strategy.

        :return: {self.max_concurrency} for the throughput scheduling strategy to limit
            the processing requests to the maximum concurrency.
            If max_concurrency is None, this will be set to None.
        """
        return self.max_concurrency

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields the start time.time() so requests are sent
        immediately, while scheduling as many asynchronously as possible.

        :return: A generator that yields the start time.time()
            for immediate request scheduling.
        """
        start_time = time.time()

        while True:
            yield start_time


class AsyncConstantStrategy(SchedulingStrategy):
    """
    A class representing an asynchronous constant scheduling strategy.
    This strategy schedules requests asynchronously at a constant request rate
    in requests per second.
    If initial_burst is set, it will send an initial burst of math.floor(rate)
    requests to reach the target rate.
    This is useful to ensure that the target rate is reached quickly
    and then maintained.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for asynchronous constant scheduling.

    :param type_: The constant StrategyType to schedule requests asynchronously.
    :param rate: The rate at which to schedule requests asynchronously in
        requests per second. This must be a positive float.
    :param initial_burst: True to send an initial burst of requests
        (math.floor(self.rate)) to reach target rate.
        False to not send an initial burst.
    """

    type_: Literal["constant"] = "constant"
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    initial_burst: bool = Field(
        default=True,
        description=(
            "True to send an initial burst of requests (math.floor(self.rate)) "
            "to reach target rate. False to not send an initial burst."
        ),
    )

    @property
    def processing_mode(self) -> Literal["async"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'async' for asynchronous scheduling strategy
            for the multiple worker processes handling requests.
        """
        return "async"

    @property
    def processes_limit(self) -> Optional[int]:
        """
        The limit on the number of worker processes for the scheduling strategy
        or None if the strategy does not restrict the number of processes.
        This property determines how many worker processes are created
        for the scheduling strategy.

        :return: None for the async constant scheduling strategy to apply
            no limit on the number of processes.
        """
        return None

    @property
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy or None
        if the strategy does not restrict the number of queued requests.
        This property determines how many requests can be queued at one time
        for the scheduling strategy.

        :return: None for the async constant scheduling strategy to apply
            no limit on the number of queued requests.
        """
        return None

    @property
    def processing_requests_limit(self) -> Optional[int]:
        """
        The maximum number of processing requests for the scheduling strategy
        or None if the strategy does not restrict the number of processing requests.
        This property determines how many requests can be processed at one time
        for the scheduling strategy.

        :return: None for the async constant scheduling strategy to apply
            no limit on the number of processing requests.
        """
        return None

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields timestamps for when requests should be sent.
        This method schedules requests asynchronously at a constant rate
        in requests per second.
        If burst_time is set, it will send an initial burst of requests
        to reach the target rate.
        This is useful to ensure that the target rate is reached quickly
        and then maintained.

        :return: A generator that yields timestamps for request scheduling.
        """
        start_time = time.time()
        constant_increment = 1.0 / self.rate

        # handle bursts first to get to the desired rate
        if self.initial_burst is not None:
            # calcualte total burst count based on sending initial at rate
            # plus any within the time to ramp up
            burst_count = math.floor(self.rate)
            for _ in range(burst_count):
                yield start_time

            start_time += constant_increment

        counter = 0

        # continue with constant rate after bursting
        while True:
            yield start_time + constant_increment * counter
            counter += 1


class AsyncPoissonStrategy(SchedulingStrategy):
    """
    A class representing an asynchronous Poisson scheduling strategy.
    This strategy schedules requests asynchronously at a Poisson request rate
    in requests per second.
    If initial_burst is set, it will send an initial burst of math.floor(rate)
    requests to reach the target rate.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for asynchronous Poisson scheduling.

    :param type_: The Poisson StrategyType to schedule requests asynchronously.
    :param rate: The rate at which to schedule requests asynchronously in
        requests per second. This must be a positive float.
    :param initial_burst: True to send an initial burst of requests
        (math.floor(self.rate)) to reach target rate.
        False to not send an initial burst.
    """

    type_: Literal["poisson"] = "poisson"
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    initial_burst: bool = Field(
        default=True,
        description=(
            "True to send an initial burst of requests (math.floor(self.rate)) "
            "to reach target rate. False to not send an initial burst."
        ),
    )

    @property
    def processing_mode(self) -> Literal["async"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'async' for asynchronous scheduling strategy
            for the multiple worker processes handling requests.
        """
        return "async"

    @property
    def processes_limit(self) -> Optional[int]:
        """
        The limit on the number of worker processes for the scheduling strategy
        or None if the strategy does not restrict the number of processes.
        This property determines how many worker processes are created
        for the scheduling strategy.

        :return: None for the async poisson scheduling strategy to apply
            no limit on the number of processes.
        """
        return None

    @property
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy or None
        if the strategy does not restrict the number of queued requests.
        This property determines how many requests can be queued at one time
        for the scheduling strategy.

        :return: None for the async poisson scheduling strategy to apply
            no limit on the number of queued requests.
        """
        return None

    @property
    def processing_requests_limit(self) -> Optional[int]:
        """
        The maximum number of processing requests for the scheduling strategy
        or None if the strategy does not restrict the number of processing requests.
        This property determines how many requests can be processed at one time
        for the scheduling strategy.

        :return: None for the async poisson scheduling strategy to apply
            no limit on the number of processing requests.
        """
        return None

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields timestamps for when requests should be sent.
        This method schedules requests asynchronously at a Poisson rate
        in requests per second.
        The inter arrival time between requests is exponentially distributed
        based on the rate.

        :return: A generator that yields timestamps for request scheduling.
        """
        start_time = time.time()

        if self.initial_burst is not None:
            # calcualte total burst count based on sending initial at rate
            # plus any within the time to ramp up
            burst_count = math.floor(self.rate)
            for _ in range(burst_count):
                yield start_time
        else:
            yield start_time

        while True:
            inter_arrival_time = random.expovariate(self.rate)
            start_time += inter_arrival_time
            yield start_time
