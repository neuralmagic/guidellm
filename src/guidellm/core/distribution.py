import numpy as np
from typing import List, Union

__all__ = ["Distribution"]


class Distribution:
    """
    A class to represent a statistical distribution and perform various statistical analyses.

    :param data: List of numerical data points (int or float) to initialize the distribution.
    :type data: List[Union[int, float]], optional
    """

    def __init__(self, data: List[Union[int, float]] = None):
        """
        Initialize the Distribution with optional data.

        :param data: List of numerical data points to initialize the distribution, defaults to None
        :type data: List[Union[int, float]], optional
        """
        self._data = data or []

    def __str__(self) -> str:
        """
        Return a string representation of the Distribution.

        :return: String representation of the Distribution
        :rtype: str
        """
        return f"Distribution(mean={self.mean:.2f}, median={self.median:.2f}, min={self.min}, max={self.max}, count={len(self._data)})"

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the Distribution for debugging.

        :return: Unambiguous string representation of the Distribution
        :rtype: str
        """
        return f"Distribution(data={self._data})"

    @property
    def data(self):
        """
        Return the data points of the distribution.

        :return: The data points of the distribution
        :rtype: List[Union[int, float]]
        """
        return self._data

    @property
    def mean(self) -> float:
        """
        Calculate and return the mean of the distribution.

        :return: The mean of the distribution
        :rtype: float
        """
        return np.mean(self._data).item()

    @property
    def median(self) -> float:
        """
        Calculate and return the median of the distribution.

        :return: The median of the distribution
        :rtype: float
        """
        return np.median(self._data).item()

    @property
    def variance(self) -> float:
        """
        Calculate and return the variance of the distribution.

        :return: The variance of the distribution
        :rtype: float
        """
        return np.var(self._data).item()

    @property
    def std_deviation(self) -> float:
        """
        Calculate and return the standard deviation of the distribution.

        :return: The standard deviation of the distribution
        :rtype: float
        """
        return np.std(self._data).item()

    def percentile(self, percentile: float) -> float:
        """
        Calculate and return the specified percentile of the distribution.

        :param percentile: The desired percentile to calculate (0-100)
        :type percentile: float
        :return: The specified percentile of the distribution
        :rtype: float
        """
        return np.percentile(self._data, percentile)

    def percentiles(self, percentiles: List[float]) -> List[float]:
        """
        Calculate and return the specified percentiles of the distribution.

        :param percentiles: A list of desired percentiles to calculate (0-100)
        :type percentiles: List[float]
        :return: A list of the specified percentiles of the distribution
        :rtype: List[float]
        """
        return np.percentile(self._data, percentiles).tolist()

    @property
    def min(self) -> float:
        """
        Return the minimum value of the distribution.

        :return: The minimum value of the distribution
        :rtype: float
        """
        return np.min(self._data)

    @property
    def max(self) -> float:
        """
        Return the maximum value of the distribution.

        :return: The maximum value of the distribution
        :rtype: float
        """
        return np.max(self._data)

    @property
    def range(self) -> float:
        """
        Calculate and return the range of the distribution (max - min).

        :return: The range of the distribution
        :rtype: float
        """
        return self.max - self.min

    def describe(self) -> dict:
        """
        Return a dictionary describing various statistics of the distribution.

        :return: A dictionary with statistical summaries of the distribution
        :rtype: dict
        """
        return {
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

    def add_data(self, new_data: List[Union[int, float]]):
        """
        Add new data points to the distribution.

        :param new_data: A list of new numerical data points to add
        :type new_data: List[Union[int, float]]
        """
        self._data.extend(new_data)

    def remove_data(self, remove_data: List[Union[int, float]]):
        """
        Remove specified data points from the distribution.

        :param remove_data: A list of numerical data points to remove
        :type remove_data: List[Union[int, float]]
        """
        self._data = [item for item in self._data if item not in remove_data]
