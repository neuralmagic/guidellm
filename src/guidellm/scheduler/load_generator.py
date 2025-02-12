import time
from typing import Generator, Literal, Optional, get_args

import numpy as np
from loguru import logger

__all__ = ["LoadGenerationMode", "LoadGenerator"]

LoadGenerationMode = Literal[
    "synchronous", "constant", "poisson", "throughput", "consistent"
]


class LoadGenerator:
    """
    Load Generator class that generates timestamps for load generation.

    This class supports multiple load generation modes: "constant", "poisson",
    "throughput", and "synchronous". Each mode has its own method for generating
    timestamps based on the rate provided during initialization.

    :param mode: The mode of load generation. Valid options are "constant",
        "poisson", "throughput", and "synchronous", "consistent"
    :type mode: LoadGenerationMode
    :param rate: The rate at which to generate timestamps. This value is
        interpreted differently depending on the mode.
    :type rate: float

    :raises ValueError: If an invalid mode is provided.
    """

    def __init__(self, mode: LoadGenerationMode, rate: Optional[float] = None):
        """
        Initialize the Load Generator with the mode and rate.

        :param mode: The mode of load generation ("constant", "poisson", "throughput",
            or "synchronous").
        :type mode: LoadGenerationMode
        :param rate: The rate at which to generate timestamps. In the "constant"
            mode, this represents the frequency of events. In the "poisson" mode,
            it represents the average frequency.
        :type rate: Optional[float]
        """
        if mode not in get_args(LoadGenerationMode):
            error = ValueError(
                f"{mode} is not a valid Load Generation Mode. "
                f"Valid options are {get_args(LoadGenerationMode)}"
            )
            logger.error(error)
            raise error

        if mode not in ["synchronous", "throughput"] and (rate is None or rate <= 0):
            error = ValueError(f"Rate must be > 0 for mode: {mode}. Given: {rate}")
            logger.error(error)
            raise error

        self._mode: LoadGenerationMode = mode
        self._rate: Optional[float] = rate
        logger.debug(
            "Initialized LoadGenerator with mode: {mode}, rate: {rate}",
            mode=mode,
            rate=rate,
        )

    @property
    def mode(self) -> LoadGenerationMode:
        """
        Get the mode of load generation.

        :return: The mode of load generation.
        :rtype: LoadGenerationMode
        """
        return self._mode

    @property
    def rate(self) -> Optional[float]:
        """
        Get the rate of load generation.

        :return: The rate of load generation.
        :rtype: Optional[float]
        """
        return self._rate

    def times(self) -> Generator[float, None, None]:
        """
        Generate timestamps for load generation based on the selected mode.

        :return: A generator that yields timestamps at which each load
            should be initiated.
        :rtype: Generator[float, None, None]

        :raises ValueError: If the mode is invalid.
        """
        logger.debug(f"Generating timestamps using mode: {self._mode}")

        if self._mode == "throughput":
            yield from self.throughput_times()
        elif self._mode == "constant":
            yield from self.constant_times()
        elif self._mode == "poisson":
            yield from self.poisson_times()
        elif self._mode == "synchronous":
            yield from self.synchronous_times()
        else:
            logger.error(f"Invalid mode encountered: {self._mode}")
            raise ValueError(f"Invalid mode: {self._mode}")

    def synchronous_times(self) -> Generator[float, None, None]:
        """
        Generate invalid timestamps for the "synchronous" mode.

        :return: A generator that yields a constant invalid timestamp (-1.0).
        :rtype: Generator[float, None, None]
        """
        logger.debug("Generating invalid timestamps for synchronous mode")
        while True:
            yield -1.0

    def throughput_times(self) -> Generator[float, None, None]:
        """
        Generate timestamps at the maximum rate possible, returning the current time.

        :return: A generator that yields the current time in seconds.
        :rtype: Generator[float, None, None]
        """
        logger.debug("Generating timestamps at throughput rate")
        while True:
            yield time.time()

    def constant_times(self) -> Generator[float, None, None]:
        """
        Generate timestamps at a constant rate based on the specified rate.

        :return: A generator that yields timestamps incremented by 1/rate seconds.
        :rtype: Generator[float, None, None]
        """
        logger.debug("Generating constant rate timestamps with rate: {}", self._rate)

        if self._rate is None or self._rate == 0:
            raise ValueError(
                "Rate must be > 0 for constant mode, given: {}", self._rate
            )

        start_time = time.time()
        time_increment = 1.0 / self._rate
        counter = 0

        while True:
            yield_time = start_time + time_increment * counter
            logger.debug(f"Yielding timestamp: {yield_time}")
            yield yield_time
            counter += 1

    def poisson_times(self) -> Generator[float, None, None]:
        """
        Generate timestamps based on a Poisson process, where the number
        of requests to be sent per second is drawn from a Poisson distribution.
        The inter arrival time between requests is exponentially distributed.

        :return: A generator that yields timestamps based on a Poisson distribution.
        :rtype: Generator[float, None, None]
        """
        logger.debug("Generating Poisson rate timestamps with rate: {}", self._rate)

        if self._rate is None or self._rate == 0:
            raise ValueError("Rate must be > 0 for poisson mode, given: {}", self._rate)

        time_tracker = time.time()
        rng = np.random.default_rng()
        time_increment = 1.0

        while True:
            num_requests = rng.poisson(self._rate)

            if num_requests == 0:
                yield time_tracker + time_increment
            else:
                inter_arrival_times = rng.exponential(1.0 / self._rate, num_requests)
                logger.debug(
                    "Calculated new inter-arrival times for poisson process: {}",
                    inter_arrival_times,
                )
                arrival_time_tracker = time_tracker

                for arrival_time in inter_arrival_times:
                    arrival_time_tracker += arrival_time

                    if arrival_time_tracker > time_tracker + time_increment:
                        logger.debug(
                            "Arrival time tracker: {} is greater than current time",
                            arrival_time_tracker,
                        )
                        break

                    yield arrival_time_tracker

            time_tracker += time_increment  # Move on to the next time period
