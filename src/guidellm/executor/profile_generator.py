from typing import Any, Dict, List, Literal, Optional, Sequence, Union, get_args

import numpy as np
from loguru import logger
from numpy._typing import NDArray
from pydantic import Field

from guidellm.config import settings
from guidellm.core import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.core.serializable import Serializable
from guidellm.scheduler import LoadGenerationMode

__all__ = [
    "Profile",
    "ProfileGenerationMode",
    "ProfileGenerator",
]

ProfileGenerationMode = Literal[
    "sweep", "synchronous", "throughput", "constant", "poisson"
]


class Profile(Serializable):
    """
    A data class representing a profile for load generation.

    :param load_gen_mode: The mode of load generation (e.g., constant, poisson).
    :type load_gen_mode: LoadGenerationMode
    :param load_gen_rate: The rate of load generation, if applicable.
    :type load_gen_rate: Optional[float]
    :param args: Additional arguments for the profile.
    :type args: Optional[Dict[str, Any]]
    """

    load_gen_mode: LoadGenerationMode
    load_gen_rate: Optional[float] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class ProfileGenerator:
    """
    Generates profiles based on different load generation modes.

    :param mode: The mode for profile generation (e.g., sweep, synchronous).
    :type mode: ProfileGenerationMode
    :param rate: The rate(s) for load generation; could be a float or list of floats.
    :type rate: Optional[Union[float, Sequence[float]]]
    """

    def __init__(
        self,
        mode: ProfileGenerationMode,
        rate: Optional[Union[float, Sequence[float]]] = None,
    ):
        if mode not in get_args(ProfileGenerationMode):
            err = ValueError(
                f"{mode} is not a valid Profile Generation Mode. "
                f"Valid options are {get_args(ProfileGenerationMode)}"
            )
            logger.error(err)
            raise err

        self._mode = mode

        if self._mode in ("sweep", "throughput", "synchronous"):
            if rate is not None:
                err = ValueError(f"Rates are not applicable for {self._mode} mode")
                logger.error(err)
                raise err
            self._rates = None
        else:
            if not rate:
                err = ValueError(f"Rates are required for {self._mode} mode")
                logger.error(err)
                raise err
            self._rates = rate if isinstance(rate, Sequence) else [rate]

            for rt in self._rates:
                if rt <= 0:
                    err = ValueError(
                        f"Rate must be > 0 for mode: {self._mode}. Given: {rt}"
                    )
                    logger.error(err)
                    raise err

        self._generated_count = 0

    def __len__(self) -> int:
        """
        Returns the number of profiles to generate based on the mode and rates.

        :return: The number of profiles.
        :rtype: int
        """
        if self._mode == "sweep":
            return settings.num_sweep_profiles + 2

        if self._mode in ("throughput", "synchronous"):
            return 1

        if not self._rates:
            raise ValueError(f"Rates are required for {self._mode} mode")

        return len(self._rates)

    @property
    def mode(self) -> ProfileGenerationMode:
        """
        Returns the current mode of profile generation.

        :return: The profile generation mode.
        :rtype: ProfileGenerationMode
        """
        return self._mode

    @property
    def rates(self) -> Optional[Sequence[float]]:
        """
        Returns the list of rates for load generation, if any.

        :return: Sequence of rates or None if not applicable.
        :rtype: Optional[Sequence[float]]
        """
        return self._rates

    @property
    def generated_count(self) -> int:
        """
        Returns the current count of generated profiles.

        :return: The current count of generated profiles.
        :rtype: int
        """
        return self._generated_count

    @property
    def profile_generation_modes(self) -> Sequence[ProfileGenerationMode]:
        """
        Return the list of profile modes to be run in the report.

        :return: Sequence of profile modes to be run in the report.
        :rtype: Sequence[ProfileGenerationMode]
        """
        if self._mode == "sweep":
            return ["synchronous", "throughput"] + ["constant"] * (  # type: ignore  # noqa: PGH003
                settings.num_sweep_profiles
            )

        if self._mode in ["throughput", "synchronous"]:
            return [self._mode]

        if self._rates is None:
            raise ValueError(f"Rates are required for {self._mode} mode")

        if self._mode in ["constant", "poisson"]:
            return [self._mode] * len(self._rates)

        raise ValueError(f"Invalid mode: {self._mode}")

    def next(self, current_report: TextGenerationBenchmarkReport) -> Optional[Profile]:
        """
        Generates the next profile based on the current mode and report.

        :param current_report: The current report report.
        :type current_report: TextGenerationBenchmarkReport
        :return: The generated profile or None if no more profiles.
        :rtype: Optional[Profile]
        """
        logger.debug(
            "Generating the next profile with mode: {}, current report: {}",
            self.mode,
            current_report,
        )

        if self.mode in ["constant", "poisson"]:
            if not self.rates:
                err = ValueError(f"Rates are required for {self.mode} mode")
                logger.error(err)
                raise err

            profile = self.create_fixed_rate_profile(
                self.generated_count,
                self.mode,
                self.rates,
            )
        elif self.mode == "synchronous":
            profile = self.create_synchronous_profile(self.generated_count)
        elif self.mode == "throughput":
            profile = self.create_throughput_profile(self.generated_count)
        elif self.mode == "sweep":
            profile = self.create_sweep_profile(
                self.generated_count,
                sync_benchmark=(
                    current_report.benchmarks[0] if current_report.benchmarks else None
                ),
                throughput_benchmark=(
                    current_report.benchmarks[1]
                    if len(current_report.benchmarks) > 1
                    else None
                ),
            )
        else:
            err = ValueError(f"Invalid mode: {self.mode}")
            logger.error(err)
            raise err

        self._generated_count += 1
        logger.info(
            "Generated profile: {}, total generated count: {}",
            profile,
            self._generated_count,
        )
        return profile

    @staticmethod
    def create_fixed_rate_profile(
        index: int, mode: ProfileGenerationMode, rates: Sequence[float]
    ) -> Optional[Profile]:
        """
        Creates a profile with a fixed rate.

        :param index: The index of the rate in the list.
        :type index: int
        :param mode: The mode for profile generation (e.g., constant, poisson).
        :type mode: ProfileGenerationMode
        :param rates: The list of rates for load generation.
        :type rates: Sequence[float]
        :return: The generated profile or None if index is out of range.
        :rtype: Optional[Profile]
        """
        modes_map: Dict[str, LoadGenerationMode] = {
            "constant": "constant",
            "poisson": "poisson",
        }

        if mode not in modes_map:
            err = ValueError(f"Invalid mode: {mode}")
            logger.error(err)
            raise err

        profile = (
            Profile(
                load_gen_mode=modes_map[mode],
                load_gen_rate=rates[index],
            )
            if index < len(rates)
            else None
        )
        logger.debug("Created fixed rate profile: {}", profile)
        return profile

    @staticmethod
    def create_synchronous_profile(index: int) -> Optional[Profile]:
        """
        Creates a profile with synchronous mode.

        :param index: The index of the profile to create.
        :type index: int
        :return: The generated profile or None if index is out of range.
        :rtype: Optional[Profile]
        """
        profile = (
            Profile(
                load_gen_mode="synchronous",
                load_gen_rate=None,
            )
            if index < 1
            else None
        )
        logger.debug("Created synchronous profile: {}", profile)
        return profile

    @staticmethod
    def create_throughput_profile(index: int) -> Optional[Profile]:
        """
        Creates a profile with throughput mode.

        :param index: The index of the profile to create.
        :type index: int
        :return: The generated profile or None if index is out of range.
        :rtype: Optional[Profile]
        """
        profile = (
            Profile(
                load_gen_mode="throughput",
                load_gen_rate=None,
            )
            if index < 1
            else None
        )
        logger.debug("Created throughput profile: {}", profile)
        return profile

    @staticmethod
    def create_sweep_profile(
        index: int,
        sync_benchmark: Optional[TextGenerationBenchmark],
        throughput_benchmark: Optional[TextGenerationBenchmark],
    ) -> Optional[Profile]:
        """
        Creates a profile with sweep mode, generating profiles between
        synchronous and throughput benchmarks.

        :param index: The index of the profile to create.
        :type index: int
        :param sync_benchmark: The synchronous report data.
        :type sync_benchmark: Optional[TextGenerationBenchmark]
        :param throughput_benchmark: The throughput report data.
        :type throughput_benchmark: Optional[TextGenerationBenchmark]
        :return: The generated profile or None if index is out of range.
        :rtype: Optional[Profile]
        """
        if index < 0 or index >= settings.num_sweep_profiles + 2:
            return None

        if index == 0:
            return ProfileGenerator.create_synchronous_profile(0)

        if not sync_benchmark:
            err = ValueError("Synchronous report is required for sweep mode")
            logger.error(err)
            raise err

        if index == 1:
            throughput_profile: Profile = ProfileGenerator.create_throughput_profile(0)  # type: ignore  # noqa: PGH003
            return throughput_profile

        if not throughput_benchmark:
            err = ValueError("Throughput report is required for sweep mode")
            logger.error(err)
            raise err

        min_rate = sync_benchmark.completed_request_rate
        max_rate = throughput_benchmark.completed_request_rate
        intermediate_rates: List[NDArray] = list(
            np.linspace(min_rate, max_rate, settings.num_sweep_profiles + 1)
        )[1:]

        return Profile(
            load_gen_mode="constant",
            load_gen_rate=(
                float(load_gen_rate)
                if (load_gen_rate := intermediate_rates[index - 2])
                else 1.0  # the fallback value
            ),
        )
