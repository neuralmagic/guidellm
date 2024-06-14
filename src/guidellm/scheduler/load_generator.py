import time
from enum import Enum
from typing import Iterator

import numpy as np

__all__ = ["LoadGenerationModes", "LoadGenerator"]


class LoadGenerationModes(Enum):
    SYNCHRONOUS = "sync"
    CONSTANT = "constant"
    POISSON = "poisson"


class LoadGenerator:
    def __init__(self, mode: LoadGenerationModes, rate: float):
        if mode == LoadGenerationModes.SYNCHRONOUS:
            raise ValueError("Synchronous mode not supported by LoadGenerator")

        self._mode = mode
        self._rate = rate

    def times(self) -> Iterator[float]:
        if self._mode == LoadGenerationModes.SYNCHRONOUS:
            raise ValueError("Synchronous mode not supported by LoadGenerator")

        if self._mode == LoadGenerationModes.CONSTANT:
            return self._constant_times()

        if self._mode == LoadGenerationModes.POISSON:
            return self._poisson_times()

    def _constant_times(self) -> Iterator[float]:
        start_time = time.time()
        time_increment = 1.0 / self._rate
        counter = 0

        while True:
            yield start_time + time_increment * counter
            counter += 1

    def _poisson_times(self) -> Iterator[float]:
        time_tracker = time.time()

        while True:
            yield time_tracker
            time_tracker += np.random.poisson(1.0 / self._rate)
