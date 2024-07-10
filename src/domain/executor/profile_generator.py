import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Iterator, List, Optional

import numpy

from domain.core import TextGenerationBenchmarkReport
from domain.load_generator import LoadGenerationMode


class ProfileGenerationMode(str, Enum):
    """
    Available values:
    * SINGLE
    * SWEEP
    * SYNCHRONOUS
    * CONSTANT
    * POISSON
    """

    SINGLE = "single"
    SWEEP = "sweep"
    SYNCHRONOUS = "synchronous"
    CONSTANT = "constant"
    POISSON = "poisson"

    @classmethod
    @functools.lru_cache(maxsize=1)
    def values(cls) -> "List[ProfileGenerationMode]":
        return [item for item in cls]


@dataclass
class Profile:
    load_gen_mode: LoadGenerationMode
    load_gen_rate: Optional[float]


class ProfileGenerator(ABC):
    _registry = {}

    def __init__(self, mode: ProfileGenerationMode):
        self._mode: ProfileGenerationMode = mode

    @abstractmethod
    def __iter__(self) -> Iterator[Profile]:
        pass

    @classmethod
    def register(cls, mode: ProfileGenerationMode):
        def inner_wrapper(wrapped_class):
            cls._registry[mode] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, mode: ProfileGenerationMode, **kwargs) -> "ProfileGenerator":
        try:
            profile_generator = cls._registry[mode](**kwargs)
        except KeyError:
            raise ValueError(f"Invalid profile generation mode: {mode}")
        else:
            return profile_generator


@ProfileGenerator.register(ProfileGenerationMode.SINGLE)
class SingleProfileGenerator(ProfileGenerator):
    def __init__(self, rate: float, rate_type: ProfileGenerationMode, **kwargs):
        super().__init__(ProfileGenerationMode.SINGLE)
        self._rate = rate
        self._rate_type = rate_type
        self._generated = False

    def __iter__(self) -> Generator[Profile, None, None]:
        if self._generated is True:
            return None
        else:
            self._generated = True

        if self._rate_type == ProfileGenerationMode.SINGLE:
            yield Profile(
                load_gen_mode=LoadGenerationMode.CONSTANT, load_gen_rate=self._rate
            )
        elif self._rate_type == ProfileGenerationMode.SYNCHRONOUS:
            yield Profile(
                load_gen_mode=LoadGenerationMode.SYNCHRONOUS, load_gen_rate=None
            )
        elif self._rate_type == ProfileGenerationMode.POISSON:
            yield Profile(
                load_gen_mode=LoadGenerationMode.POISSON, load_gen_rate=self._rate
            )
        else:
            raise ValueError(f"Invalid rate type: {self._rate_type}")


@ProfileGenerator.register(ProfileGenerationMode.SWEEP)
class SweepProfileGenerator(ProfileGenerator):
    def __init__(self, report: TextGenerationBenchmarkReport, **kwargs):
        super().__init__(ProfileGenerationMode.SWEEP)
        self._sync_run = False
        self._max_found = False
        self._pending_rates = None
        self._current_report: TextGenerationBenchmarkReport = report

    def __iter__(self) -> Generator[Profile, None, None]:
        if not self._sync_run:
            self._sync_run = True

            yield Profile(
                load_gen_mode=LoadGenerationMode.SYNCHRONOUS,
                load_gen_rate=None,
            )

        if not self._max_found:
            # check if we've found the maximum rate based on the last result
            # if not, double the rate; if so, set the flag to fill in missing data
            last_benchmark = self._current_report.benchmarks[-1]

            # TODO: fix the attribute error
            if not last_benchmark.overloaded:
                last_rate = (
                    last_benchmark.args_rate
                    if last_benchmark.args_rate
                    else last_benchmark.request_rate
                )
                yield Profile(
                    load_gen_mode=LoadGenerationMode.CONSTANT,
                    load_gen_rate=last_rate * 2,
                )
            else:
                self._max_found = True
                first_benchmark = self._current_report.benchmarks[0]

                min_rate = (
                    first_benchmark.args_rate
                    if first_benchmark.args_rate
                    else first_benchmark.request_rate
                )
                max_rate = (
                    last_benchmark.args_rate
                    if last_benchmark.args_rate
                    else last_benchmark.request_rate
                )

                self._pending_rates = list(numpy.linspace(min_rate, max_rate, 10))

        if self._pending_rates:
            rate = self._pending_rates.pop(0)
            yield Profile(load_gen_mode=LoadGenerationMode.CONSTANT, load_gen_rate=rate)
