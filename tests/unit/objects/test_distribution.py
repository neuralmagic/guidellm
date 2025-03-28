import math

import numpy as np
import pytest

from guidellm.objects import DistributionSummary, Percentiles


def create_default_percentiles() -> Percentiles:
    return Percentiles(
        p001=0.1,
        p01=1.0,
        p05=5.0,
        p10=10.0,
        p25=25.0,
        p75=75.0,
        p90=90.0,
        p95=95.0,
        p99=99.0,
        p999=99.9,
    )


def create_default_distribution_summary() -> DistributionSummary:
    return DistributionSummary(
        mean=50.0,
        median=50.0,
        variance=835,
        std_dev=math.sqrt(835),
        min=0.0,
        max=100.0,
        count=1001,
        percentiles=create_default_percentiles(),
    )


@pytest.mark.smoke()
def test_percentiles_initialization():
    percentiles = create_default_percentiles()
    assert percentiles.p001 == 0.1
    assert percentiles.p01 == 1.0
    assert percentiles.p05 == 5.0
    assert percentiles.p10 == 10.0
    assert percentiles.p25 == 25.0
    assert percentiles.p75 == 75.0
    assert percentiles.p90 == 90.0
    assert percentiles.p95 == 95.0
    assert percentiles.p99 == 99.0
    assert percentiles.p999 == 99.9


@pytest.mark.smoke()
def test_percentiles_invalid_initialization():
    test_kwargs = {
        "p001": 0.1,
        "p01": 1.0,
        "p05": 5.0,
        "p10": 10.0,
        "p25": 25.0,
        "p75": 75.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
        "p999": 99.9,
    }
    test_missing_keys = list(test_kwargs.keys())

    for missing_key in test_missing_keys:
        kwargs = {key: val for key, val in test_kwargs.items() if key != missing_key}
        with pytest.raises(ValueError):
            Percentiles(**kwargs)


@pytest.mark.smoke()
def test_percentiles_from_values():
    values = [val / 10 for val in range(1001)]
    percentiles = Percentiles.from_values(values)
    true_percentiles = np.percentile(
        values, [0.1, 1.0, 5.0, 10.0, 25.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    )
    assert percentiles.p001 == pytest.approx(true_percentiles[0])
    assert percentiles.p01 == pytest.approx(true_percentiles[1])
    assert percentiles.p05 == pytest.approx(true_percentiles[2])
    assert percentiles.p10 == pytest.approx(true_percentiles[3])
    assert percentiles.p25 == pytest.approx(true_percentiles[4])
    assert percentiles.p75 == pytest.approx(true_percentiles[5])
    assert percentiles.p90 == pytest.approx(true_percentiles[6])
    assert percentiles.p95 == pytest.approx(true_percentiles[7])
    assert percentiles.p99 == pytest.approx(true_percentiles[8])
    assert percentiles.p999 == pytest.approx(true_percentiles[9])


@pytest.mark.smoke()
def test_percentiles_marshalling():
    percentiles = create_default_percentiles()
    serialized = percentiles.model_dump()
    deserialized = Percentiles.model_validate(serialized)

    for key, value in vars(percentiles).items():
        assert getattr(deserialized, key) == value


@pytest.mark.smoke()
def test_distribution_summary_initialization():
    distribution_summary = DistributionSummary(
        mean=50.0,
        median=50.0,
        variance=835,
        std_dev=math.sqrt(835),
        min=0.0,
        max=100.0,
        count=1001,
        percentiles=create_default_percentiles(),
    )
    assert distribution_summary.mean == 50.0
    assert distribution_summary.median == 50.0
    assert distribution_summary.variance == 835
    assert distribution_summary.std_dev == math.sqrt(835)
    assert distribution_summary.min == 0.0
    assert distribution_summary.max == 100.0
    assert distribution_summary.count == 1001
    assert distribution_summary.percentiles.p001 == 0.1
    assert distribution_summary.percentiles.p01 == 1.0
    assert distribution_summary.percentiles.p05 == 5.0
    assert distribution_summary.percentiles.p10 == 10.0
    assert distribution_summary.percentiles.p25 == 25.0
    assert distribution_summary.percentiles.p75 == 75.0
    assert distribution_summary.percentiles.p90 == 90.0
    assert distribution_summary.percentiles.p95 == 95.0
    assert distribution_summary.percentiles.p99 == 99.0
    assert distribution_summary.percentiles.p999 == 99.9


@pytest.mark.smoke()
def test_distribution_summary_invalid_initialization():
    test_kwargs = {
        "mean": 50.0,
        "median": 50.0,
        "variance": 835,
        "std_dev": math.sqrt(835),
        "min": 0.0,
        "max": 100.0,
        "count": 1001,
        "percentiles": create_default_percentiles(),
    }
    test_missing_keys = list(test_kwargs.keys())
    for missing_key in test_missing_keys:
        kwargs = {key: val for key, val in test_kwargs.items() if key != missing_key}
        with pytest.raises(ValueError):
            DistributionSummary(**kwargs)


@pytest.mark.smoke()
def test_distribution_summary_from_values():
    values = [val / 10 for val in range(1001)]
    distribution_summary = DistributionSummary.from_values(values)
    assert distribution_summary.mean == pytest.approx(np.mean(values))
    assert distribution_summary.median == pytest.approx(np.median(values))
    assert distribution_summary.variance == pytest.approx(np.var(values, ddof=1))
    assert distribution_summary.std_dev == pytest.approx(np.std(values, ddof=1))
    assert distribution_summary.min == min(values)
    assert distribution_summary.max == max(values)
    assert distribution_summary.count == len(values)
    assert distribution_summary.percentiles.p001 == pytest.approx(
        np.percentile(values, 0.1)
    )
    assert distribution_summary.percentiles.p01 == pytest.approx(
        np.percentile(values, 1.0)
    )
    assert distribution_summary.percentiles.p05 == pytest.approx(
        np.percentile(values, 5.0)
    )
    assert distribution_summary.percentiles.p10 == pytest.approx(
        np.percentile(values, 10.0)
    )
    assert distribution_summary.percentiles.p25 == pytest.approx(
        np.percentile(values, 25.0)
    )
    assert distribution_summary.percentiles.p75 == pytest.approx(
        np.percentile(values, 75.0)
    )
    assert distribution_summary.percentiles.p90 == pytest.approx(
        np.percentile(values, 90.0)
    )
    assert distribution_summary.percentiles.p95 == pytest.approx(
        np.percentile(values, 95.0)
    )
    assert distribution_summary.percentiles.p99 == pytest.approx(
        np.percentile(values, 99.0)
    )
    assert distribution_summary.percentiles.p999 == pytest.approx(
        np.percentile(values, 99.9)
    )


@pytest.mark.smoke()
def test_distribution_summary_from_time_measurements_count():
    # create bimodal distribution to test count comes out to average
    # ie, 1 is active for 50 seconds, 2 is active for 100 seconds
    values = [(val / 10, 1) for val in range(500)]
    values += [(val / 5 + 50, 2) for val in range(500)]
    distribution_summary = DistributionSummary.from_time_measurements(
        values, time_weighting="count"
    )
    assert distribution_summary.mean == pytest.approx(
        (1 * 50 + 2 * 100) / 150, abs=0.001
    )
    assert distribution_summary.median == pytest.approx(2)
    assert distribution_summary.variance == pytest.approx(0.2223, abs=0.001)
    assert distribution_summary.std_dev == pytest.approx(0.4715, abs=0.001)
    assert distribution_summary.min == 1
    assert distribution_summary.max == 2
    assert distribution_summary.count == pytest.approx(100000, abs=1000)
    assert distribution_summary.percentiles.p001 == pytest.approx(1)
    assert distribution_summary.percentiles.p01 == pytest.approx(1)
    assert distribution_summary.percentiles.p05 == pytest.approx(1)
    assert distribution_summary.percentiles.p10 == pytest.approx(1)
    assert distribution_summary.percentiles.p25 == pytest.approx(1)
    assert distribution_summary.percentiles.p75 == pytest.approx(2)
    assert distribution_summary.percentiles.p90 == pytest.approx(2)
    assert distribution_summary.percentiles.p95 == pytest.approx(2)
    assert distribution_summary.percentiles.p99 == pytest.approx(2)
    assert distribution_summary.percentiles.p999 == pytest.approx(2)


@pytest.mark.smoke()
def test_distribution_summary_from_time_measurements_multiply():
    # create consistent timestamped values matching a rate of 10 per second
    values = [(val / 10, 1) for val in range(1001)]
    distribution_summary = DistributionSummary.from_time_measurements(
        values, time_weighting="multiply"
    )
    assert distribution_summary.mean == pytest.approx(0.1)
    assert distribution_summary.median == pytest.approx(0.1)
    assert distribution_summary.variance == pytest.approx(0)
    assert distribution_summary.std_dev == pytest.approx(0)
    assert distribution_summary.min == pytest.approx(0.1)
    assert distribution_summary.max == pytest.approx(0.1)
    assert distribution_summary.count == len(values) - 1
    assert distribution_summary.percentiles.p001 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p01 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p05 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p10 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p25 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p75 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p90 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p95 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p99 == pytest.approx(0.1)
    assert distribution_summary.percentiles.p999 == pytest.approx(0.1)


@pytest.mark.smoke()
def test_distribution_summary_from_time_measurements_divide():
    # create consistent timestamped values matching a rate of 10 per second
    values = [(val / 10, 1) for val in range(1001)]
    distribution_summary = DistributionSummary.from_time_measurements(
        values, time_weighting="divide"
    )
    assert distribution_summary.mean == pytest.approx(10.0)
    assert distribution_summary.median == pytest.approx(10.0)
    assert distribution_summary.variance == pytest.approx(0)
    assert distribution_summary.std_dev == pytest.approx(0)
    assert distribution_summary.min == pytest.approx(10.0)
    assert distribution_summary.max == pytest.approx(10.0)
    assert distribution_summary.count == len(values) - 1
    assert distribution_summary.percentiles.p001 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p01 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p05 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p10 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p25 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p75 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p90 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p95 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p99 == pytest.approx(10.0)
    assert distribution_summary.percentiles.p999 == pytest.approx(10.0)


@pytest.mark.smoke()
def test_distribution_summary_from_time_ranges_count():
    # create consistent time ranges representing 10 concurrent requests constant
    values = [(val / 10, val / 10 + 1, 1) for val in range(10001)]
    distribution_summary = DistributionSummary.from_time_ranges(
        values, time_weighting="count"
    )
    assert distribution_summary.mean == pytest.approx(10.0, abs=0.01)
    assert distribution_summary.median == pytest.approx(10.0)
    assert distribution_summary.variance == pytest.approx(0, abs=0.1)
    assert distribution_summary.std_dev == pytest.approx(0, abs=0.3)
    assert distribution_summary.min == pytest.approx(1)
    assert distribution_summary.max == pytest.approx(10.0)
    assert distribution_summary.count == pytest.approx(100000, abs=1000)
    assert distribution_summary.percentiles.p001 == pytest.approx(10, abs=5)
    assert distribution_summary.percentiles.p01 == pytest.approx(10)
    assert distribution_summary.percentiles.p05 == pytest.approx(10)
    assert distribution_summary.percentiles.p10 == pytest.approx(10)
    assert distribution_summary.percentiles.p25 == pytest.approx(10)
    assert distribution_summary.percentiles.p75 == pytest.approx(10)
    assert distribution_summary.percentiles.p90 == pytest.approx(10)
    assert distribution_summary.percentiles.p95 == pytest.approx(10)
    assert distribution_summary.percentiles.p99 == pytest.approx(10)
    assert distribution_summary.percentiles.p999 == pytest.approx(10)


@pytest.mark.smoke()
def test_distribution_summary_from_time_ranges_multiply():
    # create consistent time ranges representing 10 concurrent requests constant
    values = [(val / 10, val / 10 + 1, 1) for val in range(10001)]
    distribution_summary = DistributionSummary.from_time_ranges(
        values, time_weighting="multiply"
    )
    assert distribution_summary.mean == pytest.approx(1.0, abs=0.01)
    assert distribution_summary.median == pytest.approx(1.0)
    assert distribution_summary.variance == pytest.approx(0, abs=0.1)
    assert distribution_summary.std_dev == pytest.approx(0, abs=0.3)
    assert distribution_summary.min == pytest.approx(0.1)
    assert distribution_summary.max == pytest.approx(1.0)
    assert distribution_summary.count == pytest.approx(10000, abs=10)
    assert distribution_summary.percentiles.p001 == pytest.approx(1.0, abs=0.5)
    assert distribution_summary.percentiles.p01 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p05 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p10 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p25 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p75 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p90 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p95 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p99 == pytest.approx(1.0)
    assert distribution_summary.percentiles.p999 == pytest.approx(1.0)


@pytest.mark.smoke()
def test_distribution_summary_marshalling():
    distribution_summary = create_default_distribution_summary()
    serialized = distribution_summary.model_dump()
    deserialized = DistributionSummary.model_validate(serialized)

    for key, value in vars(distribution_summary).items():
        assert getattr(deserialized, key) == value
