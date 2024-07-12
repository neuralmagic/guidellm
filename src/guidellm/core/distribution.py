from typing import List, Optional, Union

import numpy as np
from loguru import logger

__all__ = ["Distribution"]


class Distribution:
    """
    A class to represent a statistical distribution and perform various statistical
    analyses.

    :param data: List of numerical data points (int or float) to initialize the
        distribution.
    :type data: List[Union[int, float]], optional
    """

    def __init__(self, data: Optional[List[Union[int, float]]] = None):
        """
        Initialize the Distribution with optional data.

        :param data: List of numerical data points to initialize the distribution,
            defaults to None.
        :type data: List[Union[int, float]], optional
        """
        self._data = list(data) if data else []
        logger.debug(f"Initialized Distribution with data: {self._data}")

    def __str__(self) -> str:
        """
        Return a string representation of the Distribution.

        :return: String representation of the Distribution.
        :rtype: str
        """
        return (
            f"Distribution(mean={self.mean:.2f}, median={self.median:.2f}, "
            f"min={self.min}, max={self.max}, count={len(self._data)})"
        )

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the Distribution for debugging.

        :return: Unambiguous string representation of the Distribution.
        :rtype: str
        """
        return f"Distribution(data={self._data})"

    @property
    def data(self) -> List[Union[int, float]]:
        """
        Return the data points of the distribution.

        :return: The data points of the distribution.
        :rtype: List[Union[int, float]]
        """
        return self._data

    @property
    def mean(self) -> float:
        """
        Calculate and return the mean of the distribution.

        :return: The mean of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate mean.")
            return 0.0

        mean_value = np.mean(self._data).item()
        logger.debug(f"Calculated mean: {mean_value}")
        return mean_value

    @property
    def median(self) -> float:
        """
        Calculate and return the median of the distribution.

        :return: The median of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate median.")
            return 0.0

        median_value = np.median(self._data).item()
        logger.debug(f"Calculated median: {median_value}")
        return median_value

    @property
    def variance(self) -> float:
        """
        Calculate and return the variance of the distribution.

        :return: The variance of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate variance.")
            return 0.0

        variance_value = np.var(self._data).item()
        logger.debug(f"Calculated variance: {variance_value}")
        return variance_value

    @property
    def std_deviation(self) -> float:
        """
        Calculate and return the standard deviation of the distribution.

        :return: The standard deviation of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate standard deviation.")
            return 0.0

        std_deviation_value = np.std(self._data).item()
        logger.debug(f"Calculated standard deviation: {std_deviation_value}")
        return std_deviation_value

    def percentile(self, percentile: float) -> float:
        """
        Calculate and return the specified percentile of the distribution.

        :param percentile: The desired percentile to calculate (0-100).
        :type percentile: float
        :return: The specified percentile of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate percentile.")
            return 0.0

        percentile_value = np.percentile(self._data, percentile).item()
        logger.debug(f"Calculated {percentile}th percentile: {percentile_value}")
        return percentile_value

    def percentiles(self, percentiles: List[float]) -> List[float]:
        """
        Calculate and return the specified percentiles of the distribution.

        :param percentiles: A list of desired percentiles to calculate (0-100).
        :type percentiles: List[float]
        :return: A list of the specified percentiles of the distribution.
        :rtype: List[float]
        """
        if not self._data:
            logger.warning("No data points available to calculate percentiles.")
            return [0.0] * len(percentiles)

        percentiles_values = np.percentile(self._data, percentiles).tolist()
        logger.debug(f"Calculated percentiles {percentiles}: {percentiles_values}")
        return percentiles_values

    @property
    def min(self) -> float:
        """
        Return the minimum value of the distribution.

        :return: The minimum value of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate minimum.")
            return 0.0

        min_value = np.min(self._data)
        logger.debug(f"Calculated min: {min_value}")
        return min_value

    @property
    def max(self) -> float:
        """
        Return the maximum value of the distribution.

        :return: The maximum value of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate maximum.")
            return 0.0

        max_value = np.max(self._data)
        logger.debug(f"Calculated max: {max_value}")
        return max_value

    @property
    def range(self) -> float:
        """
        Calculate and return the range of the distribution (max - min).

        :return: The range of the distribution.
        :rtype: float
        """
        if not self._data:
            logger.warning("No data points available to calculate range.")
            return 0.0

        range_value = self.max - self.min
        logger.debug(f"Calculated range: {range_value}")
        return range_value

    def describe(self) -> dict:
        """
        Return a dictionary describing various statistics of the distribution.

        :return: A dictionary with statistical summaries of the distribution.
        :rtype: dict
        """
        description = {
            "mean": self.mean,
            "median": self.median,
            "variance": self.variance,
            "std_deviation": self.std_deviation,
            "percentile_indices": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
            "percentile_values": self.percentiles(
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            ),
            "min": self.min,
            "max": self.max,
            "range": self.range,
        }
        logger.debug(f"Generated description: {description}")
        return description

    def add_data(self, new_data: List[Union[int, float]]):
        """
        Add new data points to the distribution.

        :param new_data: A list of new numerical data points to add.
        :type new_data: List[Union[int, float]]
        """
        self._data.extend(new_data)
        logger.debug(f"Added new data: {new_data}")

    def remove_data(self, remove_data: List[Union[int, float]]):
        """
        Remove specified data points from the distribution.

        :param remove_data: A list of numerical data points to remove.
        :type remove_data: List[Union[int, float]]
        """
        self._data = [item for item in self._data if item not in remove_data]
        logger.debug(f"Removed data: {remove_data}")
