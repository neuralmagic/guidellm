import math
import time as timer
from collections import defaultdict
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from pydantic import Field, computed_field

from guidellm.objects import Serializable

__all__ = [
    "Percentiles",
    "DistributionSummary",
    "StatusDistributionSummary",
    "RunningStats",
    "TimeRunningStats",
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
    mode: float = Field(
        description="The mode of the distribution.",
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
    total_sum: float = Field(
        description="The total sum of the values in the distribution.",
    )
    percentiles: Percentiles = Field(
        description="The percentiles of the distribution.",
    )
    cumulative_distribution_function: Optional[List[Tuple[float, float]]] = Field(
        description=("The cumulative distribution function (CDF) of the distribution."),
        default=None,
    )

    @staticmethod
    def from_distribution_function(
        distribution: List[Tuple[float, float]],
        include_cdf: bool = False,
    ) -> "DistributionSummary":
        """
        Calculate a distribution summary from a values or
        probability distribution function (PDF).
        For a PDF, it is expected to be a list of tuples where each tuple
        contains a value and its probability.
        The probabilities across all elements should be normalized (sum to 1).
        If the PDF is not normalized, it will be normalized.
        The values distribution function is a list of tuples where each tuple
        contains a value and some weighting for that value.
        The weightings will be normalized to a probability distribution function.

        :param pdf: A list of tuples representing the PDF.
            Each tuple contains a value and its probability.
        :param include_cdf: Whether to include the cumulative distribution function
            in the output DistributionSummary.
        :return: An instance of DistributionSummary with calculated values.
        """
        values, weights = zip(*distribution) if distribution else ([], [])
        values = np.array(values)
        weights = np.array(weights)

        # create the PDF
        probabilities = weights / np.sum(weights)
        pdf = np.column_stack((values, probabilities))
        pdf = pdf[np.argsort(pdf[:, 0])]
        values = pdf[:, 0]
        probabilities = pdf[:, 1]

        # calculate the CDF
        cumulative_probabilities = np.cumsum(probabilities)
        cdf = np.column_stack((values, cumulative_probabilities))

        # calculate statistics
        mean = np.sum(values * probabilities).item()
        median = cdf[np.argmax(cdf[:, 1] >= 0.5), 0].item() if len(cdf) > 0 else 0
        mode = values[np.argmax(probabilities)].item() if len(values) > 0 else 0
        variance = np.sum((values - mean) ** 2 * probabilities).item()
        std_dev = math.sqrt(variance)
        minimum = values[0].item() if len(values) > 0 else 0
        maximum = values[-1].item() if len(values) > 0 else 0
        count = len(values)
        total_sum = np.sum(values).item()

        return DistributionSummary(
            mean=mean,
            median=median,
            mode=mode,
            variance=variance,
            std_dev=std_dev,
            min=minimum,
            max=maximum,
            count=count,
            total_sum=total_sum,
            percentiles=(
                Percentiles(
                    p001=cdf[np.argmax(cdf[:, 1] >= 0.001), 0].item(),  # noqa: PLR2004
                    p01=cdf[np.argmax(cdf[:, 1] >= 0.01), 0].item(),  # noqa: PLR2004
                    p05=cdf[np.argmax(cdf[:, 1] >= 0.05), 0].item(),  # noqa: PLR2004
                    p10=cdf[np.argmax(cdf[:, 1] >= 0.1), 0].item(),  # noqa: PLR2004
                    p25=cdf[np.argmax(cdf[:, 1] >= 0.25), 0].item(),  # noqa: PLR2004
                    p75=cdf[np.argmax(cdf[:, 1] >= 0.75), 0].item(),  # noqa: PLR2004
                    p90=cdf[np.argmax(cdf[:, 1] >= 0.9), 0].item(),  # noqa: PLR2004
                    p95=cdf[np.argmax(cdf[:, 1] >= 0.95), 0].item(),  # noqa: PLR2004
                    p99=cdf[np.argmax(cdf[:, 1] >= 0.99), 0].item(),  # noqa: PLR2004
                    p999=cdf[np.argmax(cdf[:, 1] >= 0.999), 0].item(),  # noqa: PLR2004
                )
                if len(cdf) > 0
                else Percentiles(
                    p001=0,
                    p01=0,
                    p05=0,
                    p10=0,
                    p25=0,
                    p75=0,
                    p90=0,
                    p95=0,
                    p99=0,
                    p999=0,
                )
            ),
            cumulative_distribution_function=cdf.tolist() if include_cdf else None,
        )

    @staticmethod
    def from_values(
        values: List[float],
        weights: Optional[List[float]] = None,
        include_cdf: bool = False,
    ) -> "DistributionSummary":
        """
        Calculate a distribution summary from a list of values.
        If the list is empty, all stats are set to 0.
        If weights are provided, they are used to weight the values
        so that the probabilities are shifted accordingly and larger
        weights are given more importance / weight in the distribution.
        If the weights are not provided, all values are treated equally.

        :param values: A list of numerical values.
        :param weights: A list of weights for each value.
            If None, all values are treated equally.
        :param include_cdf: Whether to include the cumulative distribution function
            in the output DistributionSummary.
        :return: An instance of DistributionSummary with calculated values.
        """
        if weights is None:
            weights = [1.0] * len(values)

        if len(values) != len(weights):
            raise ValueError(
                "The length of values and weights must be the same.",
            )

        return DistributionSummary.from_distribution_function(
            distribution=list(zip(values, weights)),
            include_cdf=include_cdf,
        )

    @staticmethod
    def from_request_times(
        requests: List[Tuple[float, float]],
        distribution_type: Literal["concurrency", "rate"],
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> "DistributionSummary":
        if distribution_type == "concurrency":
            # convert to delta changes based on when requests were running
            time_deltas = defaultdict(int)
            for start, end in requests:
                time_deltas[start] += 1
                time_deltas[end] -= 1

            # convert to the events over time measuring concurrency changes
            events = []
            active = 0

            for time, delta in sorted(time_deltas.items()):
                active += delta
                events.append((time, active))
        elif distribution_type == "rate":
            # convert to events for when requests finished
            global_start = min(start for start, _ in requests) if requests else 0
            events = [(global_start, 1)] + [(end, 1) for _, end in requests]

        # combine any events that are very close together
        flattened_events = []
        for time, val in sorted(events):
            last_time, last_val = (
                flattened_events[-1] if flattened_events else (None, None)
            )

            if last_time is not None and abs(last_time - time) <= epsilon:
                flattened_events[-1] = (last_time, last_val + val)
            else:
                flattened_events.append((time, val))

        # convert to value distribution function
        distribution = defaultdict(float)

        for ind in range(len(flattened_events) - 1):
            start_time, value = flattened_events[ind]
            end_time, _ = flattened_events[ind + 1]
            duration = end_time - start_time

            if distribution_type == "concurrency":
                # weight the concurrency value by the duration
                distribution[value] += duration
            elif distribution_type == "rate":
                # weight the rate value by the duration
                rate = value / duration
                distribution[rate] += duration

        distribution = sorted(distribution.items())

        return DistributionSummary.from_distribution_function(
            distribution=distribution,
            include_cdf=include_cdf,
        )

    @staticmethod
    def from_iterable_request_times(
        requests: List[Tuple[float, float]],
        first_iter_times: List[float],
        iter_counts: List[int],
        first_iter_counts: Optional[List[int]] = None,
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> "DistributionSummary":
        if first_iter_counts is None:
            first_iter_counts = [1] * len(requests)

        if (
            len(requests) != len(first_iter_times)
            or len(requests) != len(iter_counts)
            or len(requests) != len(first_iter_counts)
        ):
            raise ValueError(
                "requests, first_iter_times, iter_counts, and first_iter_counts must"
                "be the same length."
                f"Given {len(requests)}, {len(first_iter_times)}, {len(iter_counts)}, "
                f"{len(first_iter_counts)}",
            )

        # first break up the requests into individual iterable events
        events = defaultdict(int)
        global_start = min(start for start, _ in requests) if requests else 0
        global_end = max(end for _, end in requests) if requests else 0
        events[global_start] = 0
        events[global_end] = 0

        for (_, end), first_iter, first_iter_count, total_count in zip(
            requests, first_iter_times, first_iter_counts, iter_counts
        ):
            events[first_iter] += first_iter_count

            if total_count > 1:
                iter_latency = (end - first_iter) / (total_count - 1)
                for ind in range(1, total_count):
                    events[first_iter + ind * iter_latency] += 1

        # combine any events that are very close together
        flattened_events = []

        for time, count in sorted(events.items()):
            last_time, last_count = (
                flattened_events[-1] if flattened_events else (None, None)
            )

            if last_time is not None and abs(last_time - time) <= epsilon:
                flattened_events[-1] = (last_time, last_count + count)
            else:
                flattened_events.append((time, count))

        # convert to value distribution function
        distribution = defaultdict(float)

        for ind in range(len(flattened_events) - 1):
            start_time, count = flattened_events[ind]
            end_time, _ = flattened_events[ind + 1]
            duration = end_time - start_time
            rate = count / duration
            distribution[rate] += duration

        distribution = sorted(distribution.items())

        return DistributionSummary.from_distribution_function(
            distribution=distribution,
            include_cdf=include_cdf,
        )


class StatusDistributionSummary(Serializable):
    """
    A serializable model representing distribution summary statistics
    based on groupings of status (e.g., successful, incomplete, error) for a given
    distribution of numerical values.
    Handles the total, successful, and errored dfistributions where the total
    is the combination of the successful and errored distributions.
    """

    total: DistributionSummary = Field(
        description=(
            "The dist summary for all statuses (successful, incomplete, error).",
        )
    )
    successful: DistributionSummary = Field(
        description=(
            "The distribution summary for successful statuses "
            "(e.g., successful requests)."
        )
    )
    incomplete: DistributionSummary = Field(
        description=(
            "The distribution summary for incomplete statuses "
            "(e.g., requests that hit a timeout error and were unable to complete)."
        ),
    )
    errored: DistributionSummary = Field(
        description=(
            "The distribution summary for errored statuses (e.g., failed requests)."
        )
    )

    @staticmethod
    def from_values(
        value_types: List[Literal["successful", "incomplete", "error"]],
        values: List[float],
        weights: Optional[List[float]] = None,
        include_cdf: bool = False,
    ) -> "StatusDistributionSummary":
        if any(
            type_ not in {"successful", "incomplete", "error"} for type_ in value_types
        ):
            raise ValueError(
                "value_types must be one of 'successful', 'incomplete', or 'error'. "
                f"Got {value_types} instead.",
            )

        if weights is None:
            weights = [1.0] * len(values)

        if len(value_types) != len(values) or len(value_types) != len(weights):
            raise ValueError(
                "The length of value_types, values, and weights must be the same.",
            )

        _, successful_values, successful_weights = (
            zip(*successful)
            if (
                successful := list(
                    filter(
                        lambda val: val[0] == "successful",
                        zip(value_types, values, weights),
                    )
                )
            )
            else ([], [], [])
        )
        _, incomplete_values, incomplete_weights = (
            zip(*incomplete)
            if (
                incomplete := list(
                    filter(
                        lambda val: val[0] == "incomplete",
                        zip(value_types, values, weights),
                    )
                )
            )
            else ([], [], [])
        )
        _, errored_values, errored_weights = (
            zip(*errored)
            if (
                errored := list(
                    filter(
                        lambda val: val[0] == "error",
                        zip(value_types, values, weights),
                    )
                )
            )
            else ([], [], [])
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_values(
                values=values,
                weights=weights,
                include_cdf=include_cdf,
            ),
            successful=DistributionSummary.from_values(
                values=successful_values,
                weights=successful_weights,
                include_cdf=include_cdf,
            ),
            incomplete=DistributionSummary.from_values(
                values=incomplete_values,
                weights=incomplete_weights,
                include_cdf=include_cdf,
            ),
            errored=DistributionSummary.from_values(
                values=errored_values,
                weights=errored_weights,
                include_cdf=include_cdf,
            ),
        )

    @staticmethod
    def from_request_times(
        request_types: List[Literal["successful", "incomplete", "error"]],
        requests: List[Tuple[float, float]],
        distribution_type: Literal["concurrency", "rate"],
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> "StatusDistributionSummary":
        if distribution_type not in {"concurrency", "rate"}:
            raise ValueError(
                f"Invalid distribution_type '{distribution_type}'. "
                "Must be 'concurrency' or 'rate'."
            )

        if any(
            type_ not in {"successful", "incomplete", "error"}
            for type_ in request_types
        ):
            raise ValueError(
                "request_types must be one of 'successful', 'incomplete', or 'error'. "
                f"Got {request_types} instead.",
            )

        if len(request_types) != len(requests):
            raise ValueError(
                "The length of request_types and requests must be the same. "
                f"Got {len(request_types)} and {len(requests)} instead.",
            )

        _, successful_requests = (
            zip(*successful)
            if (successful := list(zip(request_types, requests)))
            else ([], [])
        )
        _, incomplete_requests = (
            zip(*incomplete)
            if (incomplete := list(zip(request_types, requests)))
            else ([], [])
        )
        _, errored_requests = (
            zip(*errored)
            if (errored := list(zip(request_types, requests)))
            else ([], [])
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_request_times(
                requests=requests,
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            successful=DistributionSummary.from_request_times(
                requests=successful_requests,
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            incomplete=DistributionSummary.from_request_times(
                requests=incomplete_requests,
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            errored=DistributionSummary.from_request_times(
                requests=errored_requests,
                distribution_type=distribution_type,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
        )

    @staticmethod
    def from_iterable_request_times(
        request_types: List[Literal["successful", "incomplete", "error"]],
        requests: List[Tuple[float, float]],
        first_iter_times: List[float],
        iter_counts: Optional[List[int]] = None,
        first_iter_counts: Optional[List[int]] = None,
        include_cdf: bool = False,
        epsilon: float = 1e-6,
    ) -> "StatusDistributionSummary":
        if any(
            type_ not in {"successful", "incomplete", "error"}
            for type_ in request_types
        ):
            raise ValueError(
                "request_types must be one of 'successful', 'incomplete', or 'error'. "
                f"Got {request_types} instead.",
            )

        if iter_counts is None:
            iter_counts = [1] * len(requests)

        if first_iter_counts is None:
            first_iter_counts = [1] * len(requests)

        if (
            len(request_types) != len(requests)
            or len(requests) != len(first_iter_times)
            or len(requests) != len(iter_counts)
            or len(requests) != len(first_iter_counts)
        ):
            raise ValueError(
                "request_types, requests, first_iter_times, iter_counts, and "
                "first_iter_counts must be the same length."
                f"Given {len(request_types)}, {len(requests)}, "
                f"{len(first_iter_times)}, {len(iter_counts)}, "
                f"{len(first_iter_counts)}",
            )

        (
            _,
            successful_requests,
            successful_first_iter_times,
            successful_iter_counts,
            successful_first_iter_counts,
        ) = (
            zip(*successful)
            if (
                successful := list(
                    filter(
                        lambda val: val[0] == "successful",
                        zip(
                            request_types,
                            requests,
                            first_iter_times,
                            iter_counts,
                            first_iter_counts,
                        ),
                    )
                )
            )
            else ([], [], [], [], [])
        )
        (
            _,
            incomplete_requests,
            incomplete_first_iter_times,
            incomplete_iter_counts,
            incomplete_first_iter_counts,
        ) = (
            zip(*incomplete)
            if (
                incomplete := list(
                    filter(
                        lambda val: val[0] == "incomplete",
                        zip(
                            request_types,
                            requests,
                            first_iter_times,
                            iter_counts,
                            first_iter_counts,
                        ),
                    )
                )
            )
            else ([], [], [], [], [])
        )
        (
            _,
            errored_requests,
            errored_first_iter_times,
            errored_iter_counts,
            errored_first_iter_counts,
        ) = (
            zip(*errored)
            if (
                errored := list(
                    filter(
                        lambda val: val[0] == "error",
                        zip(
                            request_types,
                            requests,
                            first_iter_times,
                            iter_counts,
                            first_iter_counts,
                        ),
                    )
                )
            )
            else ([], [], [], [], [])
        )

        return StatusDistributionSummary(
            total=DistributionSummary.from_iterable_request_times(
                requests=requests,
                first_iter_times=first_iter_times,
                iter_counts=iter_counts,
                first_iter_counts=first_iter_counts,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            successful=DistributionSummary.from_iterable_request_times(
                requests=successful_requests,
                first_iter_times=successful_first_iter_times,
                iter_counts=successful_iter_counts,
                first_iter_counts=successful_first_iter_counts,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            incomplete=DistributionSummary.from_iterable_request_times(
                requests=incomplete_requests,
                first_iter_times=incomplete_first_iter_times,
                iter_counts=incomplete_iter_counts,
                first_iter_counts=incomplete_first_iter_counts,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
            errored=DistributionSummary.from_iterable_request_times(
                requests=errored_requests,
                first_iter_times=errored_first_iter_times,
                iter_counts=errored_iter_counts,
                first_iter_counts=errored_first_iter_counts,
                include_cdf=include_cdf,
                epsilon=epsilon,
            ),
        )


class RunningStats(Serializable):
    start_time: float = Field(
        default_factory=timer.time,
    )
    count: int = Field(
        default=0,
    )
    total: float = Field(
        default=0.0,
    )
    last: float = Field(
        default=0.0,
        description="The last value added to the running statistics.",
    )

    @computed_field
    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @computed_field
    @property
    def rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / (timer.time() - self.start_time)

    def __add__(self, value: Any) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Value must be an int or float, got {type(value)} instead.",
            )

        self.update(value)

        return self.mean

    def __iadd__(self, value: Any) -> "RunningStats":
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Value must be an int or float, got {type(value)} instead.",
            )

        self.update(value)

        return self

    def update(self, value: float, count: int = 1) -> None:
        """
        Update the running statistics with a new value.
        :param value: The new value to add to the running statistics.
        """
        self.count += count
        self.total += value
        self.last = value


class TimeRunningStats(RunningStats):
    @computed_field
    @property
    def total_ms(self) -> float:
        return self.total * 1000.0

    @computed_field
    @property
    def last_ms(self) -> float:
        return self.last * 1000.0

    @computed_field
    @property
    def mean_ms(self) -> float:
        return self.mean * 1000.0

    @computed_field
    @property
    def rate_ms(self) -> float:
        return self.rate * 1000.0
