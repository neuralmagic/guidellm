import time
from enum import Enum
from typing import Generator

import numpy as np

__all__ = ["LoadGenerationMode", "LoadGenerator"]


class LoadGenerationMode(str, Enum):
    """
    Available values:
        * SYNCHRONOUS
        * CONSTANT (async)
        * POISSON (async)

    """

    SYNCHRONOUS = "sync"
    CONSTANT = "constant"
    POISSON = "poisson"


class LoadGenerator:
    def __init__(self, mode: LoadGenerationMode, rate: float):
        if mode == LoadGenerationMode.SYNCHRONOUS:
            raise ValueError("Synchronous mode not supported by LoadGenerator")

        self._mode = mode
        self._rate = rate

    def times(self) -> Generator[float, None, None]:
        if self._mode == LoadGenerationMode.SYNCHRONOUS:
            raise ValueError("Synchronous mode not supported by LoadGenerator")

        elif self._mode == LoadGenerationMode.CONSTANT:
            yield from self._constant_times()

        elif self._mode == LoadGenerationMode.POISSON:
            yield from self._poisson_times()
        else:
            raise NotImplementedError(
                f"{self._mode} is not supported Load Generation Mode"
            )

    def _constant_times(self) -> Generator[float, None, None]:
        start_time = time.time()
        time_increment = 1.0 / self._rate
        counter = 0

        while True:
            yield start_time + time_increment * counter
            counter += 1

    def _poisson_times(self) -> Generator[float, None, None]:
        time_tracker = time.time()

        while True:
            yield time_tracker
            time_tracker += np.random.poisson(1.0 / self._rate)
