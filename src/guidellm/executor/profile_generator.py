from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Type, Union

import numpy

from guidellm.core import TextGenerationBenchmarkReport
from guidellm.scheduler import LoadGenerationMode

__all__ = [
    "ProfileGenerationMode",
    "Profile",
    "ProfileGenerator",
    "FixedRateProfileGenerator",
    "SweepProfileGenerator",
]

class ProfileGenerationMode(Enum):
    FIXED_RATE = "fixed_rate"
    SWEEP = "sweep"


@dataclass
class Profile:
    load_gen_mode: LoadGenerationMode
    load_gen_rate: Optional[float]


class ProfileGenerator(ABC):
    _registry: Dict[ProfileGenerationMode, "Type[ProfileGenerator]"] = {}

    @staticmethod
    def register(mode: ProfileGenerationMode):
        def inner_wrapper(wrapped_class):
            ProfileGenerator._registry[mode] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @staticmethod
    def create(mode: ProfileGenerationMode, **kwargs) -> "ProfileGenerator":
        if mode not in ProfileGenerator._registry:
            raise ValueError(f"Invalid profile generation mode: {mode}")

        return ProfileGenerator._registry[mode](**kwargs)

    def __init__(self, mode: Union[str, ProfileGenerationMode]):
        self._mode = ProfileGenerationMode(mode)

    @abstractmethod
    def next(self, current_report: TextGenerationBenchmarkReport) -> Optional[Profile]:
        """ """
        pass


@ProfileGenerator.register_generator(ProfileGenerationMode.FIXED_RATE)
class FixedRateProfileGenerator(ProfileGenerator):
    def __init__(self, rate_type: LoadGenerationMode, rates: Optional[List[float]] = None, **kwargs):
        super().__init__(ProfileGenerationMode.FIXED_RATE)
        if rate_type == "synchronous" and rates and len(rates) > 0:
            raise ValueError("custom rates are not supported in synchronous mode")
        self._rates: Optional[List[float]] = rates
        self._rate_type: LoadGenerationMode = rate_type
        self._generated: bool = False
        self._rate_index: int = 0

    def next(
        self, current_report: TextGenerationBenchmarkReport
    ) -> Optional[Profile]:
        if self._rate_type.name == LoadGenerationMode.SYNCHRONOUS.name:
            if self._generated:
                return None
            self._generated = True
            return Profile(
                load_gen_mode=LoadGenerationMode.SYNCHRONOUS, load_gen_rate=None
            )
        elif self._rate_type.name in {LoadGenerationMode.CONSTANT.name, LoadGenerationMode.POISSON.name}:
            if self._rate_index >= len(self._rates):
                return None
            current_rate = self._rates[self._rate_index]
            self._rate_index += 1
            return Profile(
                load_gen_mode=self.rate_type, load_gen_rate=current_rate
            )

        raise ValueError(f"Invalid rate type: {self._rate_type}")


@ProfileGenerator.register(ProfileGenerationMode.SWEEP)
class SweepProfileGenerator(ProfileGenerator):
    def __init__(self, **kwargs):
        super().__init__(ProfileGenerationMode.SWEEP)
        self._sync_run = False
        self._max_found = False
        self._pending_rates = None

    def next(self, current_report: TextGenerationBenchmarkReport) -> Optional[Profile]:
        if not self._sync_run:
            self._sync_run = True

            return Profile(
                load_gen_mode=LoadGenerationMode.SYNCHRONOUS, load_gen_rate=None
            )

        if not self._max_found:
            # check if we've found the maximum rate based on the last result
            # if not, double the rate; if so, set the flag to fill in missing data
            last_benchmark = current_report.benchmarks[-1]

            if not last_benchmark.overloaded:
                last_rate = (
                    last_benchmark.rate
                    if last_benchmark.rate
                    else last_benchmark.completed_request_rate
                )
                return Profile(
                    load_gen_mode=LoadGenerationMode.CONSTANT,
                    load_gen_rate=last_rate * 2,
                )
            else:
                self._max_found = True
                first_benchmark = current_report.benchmarks[0]

                min_rate = (
                    first_benchmark.rate
                    if first_benchmark.rate
                    else first_benchmark.completed_request_rate
                )
                max_rate = (
                    last_benchmark.rate
                    if last_benchmark.rate
                    else last_benchmark.completed_request_rate
                )

                self._pending_rates = list(numpy.linspace(min_rate, max_rate, 10))

        if self._pending_rates:
            rate = self._pending_rates.pop(0)
            return Profile(
                load_gen_mode=LoadGenerationMode.CONSTANT, load_gen_rate=rate
            )

        return None