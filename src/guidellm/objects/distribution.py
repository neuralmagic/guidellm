import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from guidellm.objects import Serializable

__all__ = [
    "Percentiles",
    "DistributionSummary",
    "StatusDistributionSummary",
]


class Percentiles(Serializable):
    p001: float
    p01: float
    p05: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    p99: float
    p999: float

    @staticmethod
    def from_values(values: List[float]) -> "Percentiles":
        """
        Calculate percentiles from a list of values.

        :param values: A list of numerical values.
        :return: An instance of Percentiles with calculated percentiles.
        """
        if not values:
            return Percentiles(
                p001=0.0,
                p01=0.0,
                p05=0.0,
                p10=0.0,
                p25=0.0,
                p75=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                p999=0.0,
            )

        percentiles = np.percentile(values, [0.1, 1, 5, 10, 25, 75, 90, 95, 99, 99.9])
        return Percentiles(
            p001=percentiles[0],
            p01=percentiles[1],
            p05=percentiles[2],
            p10=percentiles[3],
            p25=percentiles[4],
            p75=percentiles[5],
            p90=percentiles[6],
            p95=percentiles[7],
            p99=percentiles[8],
            p999=percentiles[9],
        )


class DistributionSummary(Serializable):
    mean: float
    median: float
    variance: float
    std_dev: float
    min: float
    max: float
    count: int
    percentiles: Percentiles

    @staticmethod
    def from_values(values: List[float]) -> "DistributionSummary":
        """
        Create a DistributionSummary from a list of values.

        :param values: A list of numerical values.
        :return: An instance of DistributionSummary.
        """
        if not values:
            return DistributionSummary(
                mean=0.0,
                median=0.0,
                variance=0.0,
                std_dev=0.0,
                min=0.0,
                max=0.0,
                count=0,
                percentiles=Percentiles.from_values([]),
            )

        return DistributionSummary(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            variance=float(np.var(values)),
            std_dev=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            count=len(values),
            percentiles=Percentiles.from_values(values),
        )

    @staticmethod
    def from_time_measurements(
        measurements: List[Tuple[float, float]],
    ) -> "DistributionSummary":
        """
        Create a DistributionSummary from a list of time measurements of the form
        (time, value), where time is the timestamp and value is the measurement.

        :param measurements: A list of tuples containing (time, value) pairs.
        :return: An instance of DistributionSummary.
        """
        if not measurements:
            return DistributionSummary(
                mean=0.0,
                median=0.0,
                variance=0.0,
                std_dev=0.0,
                min=0.0,
                max=0.0,
                count=0,
                percentiles=Percentiles.from_values([]),
            )

        if len(measurements) == 1:
            return DistributionSummary(
                mean=measurements[0][1],
                median=measurements[0][1],
                variance=0.0,
                std_dev=0.0,
                min=measurements[0][1],
                max=measurements[0][1],
                count=1,
                percentiles=Percentiles.from_values([measurements[0][1]]),
            )

        measurements.sort(key=lambda x: x[0])
        integral = sum(
            (measurements[ind + 1][0] - measurements[ind][0]) * measurements[ind][1]
            for ind in range(len(measurements) - 1)
        )
        duration = measurements[-1][0] - measurements[0][0]
        mean = integral / duration if duration > 0 else 0.0
        variance = (
            sum(
                (measurements[ind + 1][0] - measurements[ind][0])
                * (measurements[ind][1] - mean) ** 2
                for ind in range(len(measurements) - 1)
            )
            / duration
            if duration > 0
            else 0.0
        )

        value_durations_dict = defaultdict(float)
        for ind in range(len(measurements) - 1):
            value_durations_dict[measurements[ind][1]] += (
                measurements[ind + 1][0] - measurements[ind][0]
            )
        value_durations = sorted(
            [(duration, value) for value, duration in value_durations_dict.items()],
            key=lambda x: x[0],
        )

        def _get_percentile(percentile: float) -> float:
            target_duration = percentile / 100 * duration
            cumulative_duration = 0.0
            for dur, val in value_durations:
                cumulative_duration += dur
                if cumulative_duration >= target_duration:
                    return val
            return value_durations[-1][1]

        return DistributionSummary(
            mean=mean,
            median=_get_percentile(50.0),
            variance=variance,
            std_dev=math.sqrt(variance),
            min=min([meas[1] for meas in measurements]),
            max=max([meas[1] for meas in measurements]),
            count=len(measurements),
            percentiles=Percentiles(
                p001=_get_percentile(0.1),
                p01=_get_percentile(1.0),
                p05=_get_percentile(5.0),
                p10=_get_percentile(10.0),
                p25=_get_percentile(25.0),
                p75=_get_percentile(75.0),
                p90=_get_percentile(90.0),
                p95=_get_percentile(95.0),
                p99=_get_percentile(99.0),
                p999=_get_percentile(99.9),
            ),
        )

    @staticmethod
    def from_time_measurements_with_sampling(
        measurements: List[Tuple[float, float]],
        sample_time: float,
    ) -> "DistributionSummary":
        """
        Create a DistributionSummary from a list of time measurements of the form
        (time, value), where time is the timestamp and value is the measurement.
        This method samples the measurements at regular intervals defined by
        sample_time.

        :param measurements: A list of tuples containing (time, value) pairs.
        :param sample_time: The time interval for sampling.
        :return: An instance of DistributionSummary.
        """
        measurements.sort(key=lambda x: x[0])
        samples = []
        min_time = measurements[0][0]
        max_time = measurements[-1][0] + sample_time

        for time_iter in np.arange(
            min_time,
            max_time,
            sample_time,
        ):
            count = 0
            while measurements and measurements[0][0] <= time_iter:
                count += measurements[0][1]
                measurements.pop(0)
            samples.append((time_iter, count))

        return DistributionSummary.from_time_measurements(samples)


class StatusDistributionSummary(Serializable):
    total: DistributionSummary
    completed: DistributionSummary
    errored: DistributionSummary

    @staticmethod
    def from_values(
        completed_values: List[float],
        errored_values: List[float],
    ) -> "StatusDistributionSummary":
        """
        Create a StatusDistributionSummary from completed and errored values.

        :param completed_values: A list of numerical values for completed requests.
        :param errored_values: A list of numerical values for errored requests.
        :return: An instance of StatusDistributionSummary.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_values(
                completed_values + errored_values,
            ),
            completed=DistributionSummary.from_values(completed_values),
            errored=DistributionSummary.from_values(errored_values),
        )

    @staticmethod
    def from_time_measurements(
        completed_measurements: List[Tuple[float, float]],
        errored_measurements: List[Tuple[float, float]],
    ) -> "StatusDistributionSummary":
        """
        Create a StatusDistributionSummary from completed and errored time measurements.

        :param completed_measurements: A list of tuples containing (time, value) pairs
            for completed requests.
        :param errored_measurements: A list of tuples containing (time, value) pairs
            for errored requests.
        :return: An instance of StatusDistributionSummary.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_time_measurements(
                completed_measurements + errored_measurements,
            ),
            completed=DistributionSummary.from_time_measurements(
                completed_measurements,
            ),
            errored=DistributionSummary.from_time_measurements(
                errored_measurements,
            ),
        )

    @staticmethod
    def from_time_measurements_with_sampling(
        completed_measurements: List[Tuple[float, float]],
        errored_measurements: List[Tuple[float, float]],
        sample_time: float,
    ) -> "StatusDistributionSummary":
        """
        Create a StatusDistributionSummary from completed and errored time measurements
        with sampling.

        :param completed_measurements: A list of tuples containing (time, value) pairs
            for completed requests.
        :param errored_measurements: A list of tuples containing (time, value) pairs
            for errored requests.
        :param sample_time: The time interval for sampling.
        :return: An instance of StatusDistributionSummary.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_time_measurements_with_sampling(
                completed_measurements + errored_measurements,
                sample_time,
            ),
            completed=DistributionSummary.from_time_measurements_with_sampling(
                completed_measurements,
                sample_time,
            ),
            errored=DistributionSummary.from_time_measurements_with_sampling(
                errored_measurements,
                sample_time,
            ),
        )
