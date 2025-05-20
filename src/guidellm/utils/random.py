from abc import ABC, abstractmethod
from collections.abc import Iterator
from random import Random
from typing import Generic, Optional, TypeVar

__all__ = [
    "DistributionSampler",
    "FloatDistributionSampler",
    "IntegerDistributionSampler",
]

T = TypeVar("T", int, float)


class DistributionSampler(ABC, Generic[T]):
    """
    Base class for sampling values from a distribution matching a given mean,
    variance, and constraints such as min and max values.
    Additionally, a random seed is provided to ensure reproducibility.
    """

    def __init__(
        self,
        mean: T,
        variance: Optional[T],
        min_value: Optional[T],
        max_value: Optional[T],
        random_seed: int = 42,
    ):
        """
        :param mean: The mean value of the distribution that the samples will be
            centered around.
        :param variance: The variance of the distribution. If None, the distribution
            will be uniform. If min or max are None, then mean is always returned.
        :param min_value: The minimum value that can be sampled. If None and variance
            is not None, then no constraint is applied. If None and variance is None,
            then mean is always returned.
        :param max_value: The maximum value that can be sampled. If None and variance
            is not None, then no constraint is applied. If None and variance is None,
            then mean is always returned.
        :param random_seed: A seed for the random number generator to ensure
            reproducibility of the samples. Defaults to 42.
        """
        self.mean = mean
        self.variance = variance
        self.min_value = min_value
        self.max_value = max_value
        self.seed = random_seed
        self.rand = Random(random_seed)  # noqa: S311

    def __iter__(self) -> Iterator[T]:
        """
        :return: An iterator that yields sampled values from the distribution.
        """
        sample_range = self.sample_range

        while True:
            yield self.sample(sample_range)

    @property
    def sample_range(self) -> tuple[T, T]:
        """
        :return: The range of values that can be sampled from the distribution.
        """
        minimum = self.min_value or (
            self.mean - 5 * self.variance if self.variance else self.mean
        )
        maximum = self.max_value or (
            self.mean + 5 * self.variance if self.variance else self.mean
        )

        return (minimum, maximum)

    @abstractmethod
    def sample(self, sample_range: tuple[T, T]) -> T:
        """
        :param sample_range: The range of values the sampler can sample from.
        :return: A sampled value from the distribution.
        """
        ...


class IntegerDistributionSampler(DistributionSampler[int]):
    """
    A sampler that yields integer values from a range with a given mean and variance.
    """

    def sample(self, sample_range: tuple[int, int]) -> int:
        """
        :param sample_range: The range of values the sampler can sample from.
        :return: A sampled integer value from the distribution.
        """
        minimum, maximum = sample_range

        if minimum == maximum:
            return self.mean

        if self.variance is None:
            # uniform distribution
            value = self.rand.randint(minimum, maximum)
        else:
            # normal distribution
            value = round(self.rand.gauss(self.mean, self.variance // 2))

        return max(minimum, min(maximum, value))


class FloatDistributionSampler(DistributionSampler[float]):
    """
    A sampler that yields float values from a range with a given mean and variance.
    """

    def sample(self, sample_range: tuple[float, float]) -> float:
        """
        :param sample_range: The range of values the sampler can sample from.
        :return: A sampled float value from the distribution.
        """
        minimum, maximum = sample_range

        if minimum == maximum:
            return self.mean

        if self.variance is None:
            # uniform distribution
            value = self.rand.uniform(minimum, maximum)
        else:
            # normal distribution
            value = self.rand.gauss(self.mean, self.variance)

        return max(minimum, min(maximum, value))
