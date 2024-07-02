from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy

from guidellm.core import TextGenerationBenchmarkReport
from guidellm.scheduler import LoadGenerationModes

__all__ = [
    "ProfileGenerationModes",
    "Profile",
    "ProfileGenerator",
    "FixedRateProfileGenerator",
    "SweepProfileGenerator",
]


class ProfileGenerationModes(Enum):
    FIXED = "fixed_rate"
    SWEEP = "sweep"


@dataclass()
class Profile:
    load_gen_mode: LoadGenerationModes
    load_gen_rate: Optional[float]


class ProfileGenerator(ABC):
    _registry = {}

    @staticmethod
    def register_generator(mode: ProfileGenerationModes):
        def inner_wrapper(wrapped_class):
            ProfileGenerator._registry[mode] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @staticmethod
    def create_generator(
        mode: Union[str, ProfileGenerationModes], **kwargs
    ) -> "ProfileGenerator":
        if isinstance(mode, str):
            mode = ProfileGenerationModes(mode)

        if mode not in ProfileGenerator._registry:
            raise ValueError(f"Invalid profile generation mode: {mode}")

        return ProfileGenerator._registry[mode](**kwargs)

    def __init__(self, mode: Union[str, ProfileGenerationModes]):
        self._mode = ProfileGenerationModes(mode)

    @abstractmethod
    def next_profile(
        self, current_report: TextGenerationBenchmarkReport
    ) -> Optional[Profile]:
        pass


@ProfileGenerator.register_generator(ProfileGenerationModes.FIXED)
class FixedRateProfileGenerator(ProfileGenerator):
    def __init__(self, rate: List[float], rate_type: str, **kwargs):
        super().__init__(ProfileGenerationModes.FIXED)
        self._rates = rate
        self._rate_index = 0
        self._rate_type = rate_type
        self._generated = False

    def next_profile(
        self, current_report: TextGenerationBenchmarkReport
    ) -> Optional[Profile]:
        if self._rate_index >= len(self._rates):
            return None

        if self._rate_type == "constant":
            return Profile(
                load_gen_mode=LoadGenerationModes.CONSTANT, load_gen_rate=self._rates[self._rate_index]
            )

        if self._rate_type == "synchronous":
            return Profile(
                load_gen_mode=LoadGenerationModes.SYNCHRONOUS, load_gen_rate=None
            )

        if self._rate_type == "poisson":
            return Profile(
                load_gen_mode=LoadGenerationModes.POISSON, load_gen_rate=self._rates[self._rate_index]
            )

        raise ValueError(f"Invalid rate type: {self._rate_type}")


@ProfileGenerator.register_generator(ProfileGenerationModes.SWEEP)
class SweepProfileGenerator(ProfileGenerator):
    def __init__(self, **kwargs):
        super().__init__(ProfileGenerationModes.SWEEP)
        self._sync_run = False
        self._max_found = False
        self._pending_rates = None

    def next_profile(
        self, current_report: TextGenerationBenchmarkReport
    ) -> Optional[Profile]:
        if not self._sync_run:
            self._sync_run = True

            return Profile(
                load_gen_mode=LoadGenerationModes.SYNCHRONOUS, load_gen_rate=None
            )

        if not self._max_found:
            # check if we've found the maximum rate based on the last result
            # if not, double the rate; if so, set the flag to fill in missing data
            last_benchmark = current_report.benchmarks[-1]

            if not last_benchmark.overloaded:
                last_rate = (
                    last_benchmark.args_rate
                    if last_benchmark.args_rate
                    else last_benchmark.request_rate
                )
                return Profile(
                    load_gen_mode=LoadGenerationModes.CONSTANT,
                    load_gen_rate=last_rate * 2,
                )
            else:
                self._max_found = True
                first_benchmark = current_report.benchmarks[0]

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
            return Profile(
                load_gen_mode=LoadGenerationModes.CONSTANT, load_gen_rate=rate
            )

        return None