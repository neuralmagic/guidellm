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

RateTypeLoadGenModeMap = {
    "constant": LoadGenerationModes.CONSTANT,
    "poisson": LoadGenerationModes.POISSON,
}

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
        mode_is_invalid = not isinstance(mode, str) or mode not in [m.value for m in ProfileGenerationModes]
        if mode_is_invalid:
            raise ValueError(f"Invalid profile generation mode: {mode}")
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
    def __init__(self, rate_type: str, rate: Optional[List[float]] = None, **kwargs):
        super().__init__(ProfileGenerationModes.FIXED)
        if rate_type == "synchronous" and rate and len(rate) > 0:
            raise ValueError("custom rates are not supported in synchronous mode")
        self._rates = rate
        self._rate_index = 0
        self._generated = False
        self._rate_type = rate_type

    def next_profile(
        self, current_report: TextGenerationBenchmarkReport
    ) -> Optional[Profile]:
        if self._rate_type == "synchronous":
            if self._generated:
                return None
        
            self._generated = True

            return Profile(
                load_gen_mode=LoadGenerationModes.SYNCHRONOUS, load_gen_rate=None
            )
        
        if self._rate_type in {"constant", "poisson"}:
            if self._rate_index >= len(self._rates):
                return None

            current_rate = self._rates[self._rate_index]
            self._rate_index += 1
        
            load_gen_mode = RateTypeLoadGenModeMap[self._rate_type]
            
            return Profile(
                load_gen_mode=load_gen_mode, load_gen_rate=current_rate
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