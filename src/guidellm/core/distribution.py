from typing import List, Sequence

import numpy as np
from loguru import logger
from pydantic import Field

from guidellm.core.serializable import Serializable

__all__ = ["Distribution"]


class Distribution(Serializable):
    """
    A class to represent a statistical distribution and perform various
    statistical analyses.
    """

    data: Sequence[float] = Field(
        default_factory=list,
        description="The data points of the distribution.",
    )

    def __str__(self):
        return f"Distribution({self.describe()})"

    @property
    def mean(self) -> float:
        """
        Calculate and return the mean of the distribution.
        :return: The mean of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate mean.")
            return 0.0

        mean_value = np.mean(self.data).item()
        logger.debug(f"Calculated mean: {mean_value}")
        return mean_value

    @property
    def median(self) -> float:
        """
        Calculate and return the median of the distribution.
        :return: The median of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate median.")
            return 0.0

        median_value = np.median(self.data).item()
        logger.debug(f"Calculated median: {median_value}")
        return median_value

    @property
    def variance(self) -> float:
        """
        Calculate and return the variance of the distribution.
        :return: The variance of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate variance.")
            return 0.0

        variance_value = np.var(self.data).item()
        logger.debug(f"Calculated variance: {variance_value}")
        return variance_value

    @property
    def std_deviation(self) -> float:
        """
        Calculate and return the standard deviation of the distribution.
        :return: The standard deviation of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate standard deviation.")
            return 0.0

        std_deviation_value = np.std(self.data).item()
        logger.debug(f"Calculated standard deviation: {std_deviation_value}")
        return std_deviation_value

    def percentile(self, percentile: float) -> float:
        """
        Calculate and return the specified percentile of the distribution.
        :param percentile: The desired percentile to calculate (0-100).
        :return: The specified percentile of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate percentile.")
            return 0.0

        percentile_value = np.percentile(self.data, percentile).item()
        logger.debug(f"Calculated {percentile}th percentile: {percentile_value}")
        return percentile_value

    def percentiles(self, percentiles: List[float]) -> List[float]:
        """
        Calculate and return the specified percentiles of the distribution.
        :param percentiles: A list of desired percentiles to calculate (0-100).
        :return: A list of the specified percentiles of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate percentiles.")
            return [0.0] * len(percentiles)

        percentiles_values = np.percentile(self.data, percentiles).tolist()
        logger.debug(f"Calculated percentiles {percentiles}: {percentiles_values}")
        return percentiles_values

    @property
    def min(self) -> float:
        """
        Return the minimum value of the distribution.
        :return: The minimum value of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate minimum.")
            return 0.0

        min_value = np.min(self.data)
        logger.debug(f"Calculated min: {min_value}")
        return min_value

    @property
    def max(self) -> float:
        """
        Return the maximum value of the distribution.
        :return: The maximum value of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate maximum.")
            return 0.0

        max_value = np.max(self.data)
        logger.debug(f"Calculated max: {max_value}")
        return max_value

    @property
    def range(self) -> float:
        """
        Calculate and return the range of the distribution (max - min).
        :return: The range of the distribution.
        """
        if not self.data:
            logger.warning("No data points available to calculate range.")
            return 0.0

        range_value = self.max - self.min
        logger.debug(f"Calculated range: {range_value}")
        return range_value

    def describe(self) -> dict:
        """
        Return a dictionary describing various statistics of the distribution.
        :return: A dictionary with statistical summaries of the distribution.
        """
        description = {
            "mean": self.mean,
            "median": self.median,
            "variance": self.variance,
            "std_deviation": self.std_deviation,
            "percentile_indices": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
            "percentile_values": self.percentiles(
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
            ),
            "min": self.min,
            "max": self.max,
            "range": self.range,
        }
        logger.debug(f"Generated description: {description}")
        return description

    def add_data(self, new_data: Sequence[float]):
        """
        Add new data points to the distribution.
        :param new_data: A list of new numerical data points to add.
        """
        self.data = list(self.data) + list(new_data)
        logger.debug(f"Added new data: {new_data}")

    def remove_data(self, remove_data: Sequence[float]):
        """
        Remove specified data points from the distribution.
        :param remove_data: A list of numerical data points to remove.
        """
        self.data = [item for item in self.data if item not in remove_data]
        logger.debug(f"Removed data: {remove_data}")
