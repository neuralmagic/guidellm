import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pydantic import Field

from guidellm.objects import Serializable

__all__ = [
    "Percentiles",
    "DistributionSummary",
    "StatusDistributionSummary",
]


class Percentiles(Serializable):
    """
    A serializable model representing percentiles of a distribution.
    """

    p001: float = Field(
        description="The 0.1th percentile of the distribution.",
    )
    p01: float = Field(
        description="The 1st percentile of the distribution.",
    )
    p05: float = Field(
        description="The 5th percentile of the distribution.",
    )
    p10: float = Field(
        description="The 10th percentile of the distribution.",
    )
    p25: float = Field(
        description="The 25th percentile of the distribution.",
    )
    p75: float = Field(
        description="The 75th percentile of the distribution.",
    )
    p90: float = Field(
        description="The 90th percentile of the distribution.",
    )
    p95: float = Field(
        description="The 95th percentile of the distribution.",
    )
    p99: float = Field(
        description="The 99th percentile of the distribution.",
    )
    p999: float = Field(
        description="The 99.9th percentile of the distribution.",
    )

    @staticmethod
    def from_values(values: List[float]) -> "Percentiles":
        """
        Calculate percentiles from a list of values.
        If the list is empty, all percentiles are set to 0.

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
    """
    A serializable model representing a statistical summary for a given
    distribution of numerical values.
    """

    mean: float = Field(
        description="The mean/average of the distribution.",
    )
    median: float = Field(
        description="The median of the distribution.",
    )
    variance: float = Field(
        description="The variance of the distribution.",
    )
    std_dev: float = Field(
        description="The standard deviation of the distribution.",
    )
    min: float = Field(
        description="The minimum value of the distribution.",
    )
    max: float = Field(
        description="The maximum value of the distribution.",
    )
    count: int = Field(
        description="The number of values in the distribution.",
    )
    percentiles: Percentiles = Field(
        description="The percentiles of the distribution.",
    )

    @staticmethod
    def from_values(values: List[float]) -> "DistributionSummary":
        """
        Calculate a distribution summary from a list of values.
        If the list is empty, all values are set to 0.

        :param values: A list of numerical values.
        :return: An instance of DistributionSummary with calculated values.
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
    def from_timestamped_values(
        values: List[Tuple[float, float]],
    ) -> "DistributionSummary":
        """
        Calculate a distribution summary from a list of timestamped values.
        Specifically, this calculates the statistics assuming a piecewise
        continuous distribution of values over time.
        For example, rather than finding the average concurrency of requests
        over a given time period, this will calculate that along with other
        statistics such as the variance and percentiles.
        If the list is empty, all values are set to 0.
        If the list contains only one value, all values are set to that value.
        Note, since this is calculating statistics over time, the values
        should contain the entire time range. Generally, this means the first
        value should be the start time with a measurement of 0.

        :param values: A list of timestamped numerical values of the form
            (timestamp, value).
        :return: An instance of DistributionSummary with calculated values.
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

        if len(values) == 1:
            return DistributionSummary(
                mean=values[0][1],
                median=values[0][1],
                variance=0.0,
                std_dev=0.0,
                min=values[0][1],
                max=values[0][1],
                count=1,
                percentiles=Percentiles.from_values([values[0][1]]),
            )

        # ensure values are sorted and piecewise continuous
        # (combine any values at the same time)
        tmp_values = sorted(values, key=lambda x: x[0])
        values = []
        epsilon = 1e-6

        for val in tmp_values:
            if values and abs(values[-1][0] - val[0]) < epsilon:
                values[-1] = (val[0], val[1] + values[-1][1])
            else:
                values.append(val)

        duration = values[-1][0] - values[0][0]

        # mean calculations
        integral = sum(
            (values[ind + 1][0] - values[ind][0]) * values[ind][1]
            for ind in range(len(values) - 1)
        )
        mean = integral / duration if duration > 0 else 0.0

        # variance calculations
        variance = (
            sum(
                (values[ind + 1][0] - values[ind][0]) * (values[ind][1] - mean) ** 2
                for ind in range(len(values) - 1)
            )
            / duration
            if duration > 0
            else 0.0
        )

        # percentile calculations
        value_durations_dict = defaultdict(float)
        for ind in range(len(values) - 1):
            value_durations_dict[values[ind][1]] += values[ind + 1][0] - values[ind][0]
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
            min=min([meas[1] for meas in values]),
            max=max([meas[1] for meas in values]),
            count=len(values),
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
    def from_timestamped_values_per_frequency(
        values: List[Tuple[float, float]],
        frequency: float,
    ) -> "DistributionSummary":
        """
        Calculate a distribution summary from a list of timestamped values
        at a given frequency.
        Specifically, this calculates the statistics assuming a piecewise
        continuous distribution of values over time and then samples at
        the given frequency from that distribution.
        For example, rather than finding the average requests per second
        over a given time period, this will calculate that along with other
        statistics such as the variance and percentiles.
        If the list is empty, all values are set to 0.
        If the list contains only one value, all values are set to that value.
        Note, since this is calculating statistics over time, the values
        should contain the entire time range. Generally, this means the first
        value should be the start time with a measurement of 0.

        :param values: A list of timestamped numerical values of the form
            (timestamp, value).
        :param frequency: The frequency to sample the distribution at
            represented in the same units as the timestamps.
        :return: An instance of DistributionSummary with calculated values.
        """
        values.sort(key=lambda x: x[0])
        samples = []
        min_time = values[0][0]
        max_time = values[-1][0] + frequency

        for time_iter in np.arange(
            min_time,
            max_time,
            frequency,
        ):
            count = 0
            while values and values[0][0] <= time_iter:
                count += values[0][1]
                values.pop(0)
            samples.append((time_iter, count))

        return DistributionSummary.from_timestamped_values(samples)

    @staticmethod
    def from_timestamped_interval_values(
        values: List[Tuple[float, float, float]],
    ) -> "DistributionSummary":
        """
        Calculate a distribution summary from a list of timestamped interval values,
        that may or may note be overlapping in ranges.
        Specifically, this calculates the statistics assuming a piecewise
        continuous distribution of values over time.
        For example, rather than finding the average concurrency of overlapping requests
        over a given time period, this will calculate that along with other
        statistics such as the variance and percentiles.
        If the list is empty, all values are set to 0.
        If the list contains only one value, all values are set to that value.
        Note, since this is calculating statistics over time, the values
        should contain the entire time range.

        :param values: A list of timestamped numerical values of the form
            (start_time, end_time, value).
        :return: An instance of DistributionSummary with calculated values.
        """
        events_dict = defaultdict(int)
        for start, end, count in values:
            events_dict[start] += count
            events_dict[end] -= count

        timestamped_values = []
        current_value = 0

        for time, delta in sorted(events_dict.items()):
            current_value += delta
            timestamped_values.append((time, current_value))

        return DistributionSummary.from_timestamped_values(
            timestamped_values,
        )


class StatusDistributionSummary(Serializable):
    """
    A serializable model representing distribution summary statistics
    based on groupings of status (e.g., completed, errored) for a given
    distribution of numerical values.
    Handles the total, completed, and errored distributions where the total
    is the combination of the completed and errored distributions.
    """

    total: DistributionSummary = Field(
        description="The distribution summary for all statuses (errored, completed).",
    )
    completed: DistributionSummary = Field(
        description=(
            "The distribution summary for completed statuses "
            "(e.g., successful requests)."
        )
    )
    errored: DistributionSummary = Field(
        description=(
            "The distribution summary for errored statuses " "(e.g., failed requests)."
        )
    )

    @staticmethod
    def from_values(
        completed_values: List[float],
        errored_values: List[float],
    ) -> "StatusDistributionSummary":
        """
        Calculate distribution summaries from a list of values for
        completed, errored, and the total combination of both.
        If the lists are empty, all values are set to 0.

        :param completed_values: A list of numerical values for completed statuses.
        :param errored_values: A list of numerical values for errored statuses.
        :return: An instance of StatusDistributionSummary with calculated values.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_values(
                completed_values + errored_values,
            ),
            completed=DistributionSummary.from_values(completed_values),
            errored=DistributionSummary.from_values(errored_values),
        )

    @staticmethod
    def from_timestamped_values(
        completed_values: List[Tuple[float, float]],
        errored_values: List[Tuple[float, float]],
    ) -> "StatusDistributionSummary":
        """
        Calculate distribution summaries from a list of timestamped values for
        completed, errored, and the total combination of both.
        Specifically, this calculates the statistics assuming a piecewise
        continuous distribution of values over time.
        For example, rather than finding the average concurrency of requests
        over a given time period, this will calculate that along with other
        statistics such as the variance and percentiles.
        If the lists are empty, all values are set to 0.
        If the lists contain only one value, all values are set to that value.
        Note, since this is calculating statistics over time, the values
        should contain the entire time range. Generally, this means the first
        value should be the start time with a measurement of 0.

        :param completed_values: A list of timestamped numerical values for
            completed statuses.
        :param errored_values: A list of timestamped numerical values for
            errored statuses.
        :return: An instance of StatusDistributionSummary with calculated values.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_timestamped_values(
                completed_values + errored_values,
            ),
            completed=DistributionSummary.from_timestamped_values(
                completed_values,
            ),
            errored=DistributionSummary.from_timestamped_values(
                errored_values,
            ),
        )

    @staticmethod
    def from_timestamped_values_per_frequency(
        completed_values: List[Tuple[float, float]],
        errored_values: List[Tuple[float, float]],
        frequency: float,
    ) -> "StatusDistributionSummary":
        """
        Calculate distribution summaries from a list of timestamped values for
        completed, errored, and the total combination of both at a given frequency.
        Specifically, this calculates the statistics assuming a piecewise
        continuous distribution of values over time and then samples at
        the given frequency from that distribution.
        For example, rather than finding the average requests per second
        over a given time period, this will calculate that along with other
        statistics such as the variance and percentiles.
        If the lists are empty, all values are set to 0.
        If the lists contain only one value, all values are set to that value.
        Note, since this is calculating statistics over time, the values
        should contain the entire time range. Generally, this means the first
        value should be the start time with a measurement of 0.

        :param completed_values: A list of timestamped numerical values for
            completed statuses.
        :param errored_values: A list of timestamped numerical values for
            errored statuses.
        :param frequency: The frequency to sample the distribution at
            represented in the same units as the timestamps.
        :return: An instance of StatusDistributionSummary with calculated values.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_timestamped_values_per_frequency(
                completed_values + errored_values,
                frequency,
            ),
            completed=DistributionSummary.from_timestamped_values_per_frequency(
                completed_values,
                frequency,
            ),
            errored=DistributionSummary.from_timestamped_values_per_frequency(
                errored_values,
                frequency,
            ),
        )

    @staticmethod
    def from_timestamped_interval_values(
        completed_values: List[Tuple[float, float, float]],
        errored_values: List[Tuple[float, float, float]],
    ) -> "StatusDistributionSummary":
        """
        Calculate distribution summaries from a list of timestamped interval values for
        completed, errored, and the total combination of both.
        Specifically, this calculates the statistics assuming a piecewise
        continuous distribution of values over time.
        For example, rather than finding the average concurrency of overlapping requests
        over a given time period, this will calculate that along with other
        statistics such as the variance and percentiles.
        If the lists are empty, all values are set to 0.
        If the lists contain only one value, all values are set to that value.
        Note, since this is calculating statistics over time, the values
        should contain the entire time range.

        :param completed_values: A list of timestamped numerical values for
            completed statuses.
        :param errored_values: A list of timestamped numerical values for
            errored statuses.
        :return: An instance of StatusDistributionSummary with calculated values.
        """
        return StatusDistributionSummary(
            total=DistributionSummary.from_timestamped_interval_values(
                completed_values + errored_values,
            ),
            completed=DistributionSummary.from_timestamped_interval_values(
                completed_values,
            ),
            errored=DistributionSummary.from_timestamped_interval_values(
                errored_values,
            ),
        )
