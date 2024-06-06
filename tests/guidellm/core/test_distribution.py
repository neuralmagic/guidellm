import pytest
import numpy as np
from guidellm import Distribution


@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def dist(sample_data):
    return Distribution(sample_data)


@pytest.mark.unit
def test_mean(dist):
    assert dist.mean == np.mean(dist._data_)


@pytest.mark.unit
def test_median(dist):
    assert dist.median == np.median(dist._data_)


@pytest.mark.unit
def test_variance(dist):
    assert dist.variance == np.var(dist._data_)


@pytest.mark.unit
def test_std_deviation(dist):
    assert dist.std_deviation == np.std(dist._data_)


@pytest.mark.unit
def test_percentile(dist):
    assert dist.percentile(50) == np.percentile(dist._data_, 50)


@pytest.mark.unit
def test_percentiles(dist):
    percentiles = [10, 50, 90]
    assert (
        dist.percentiles(percentiles)
        == np.percentile(dist._data_, percentiles).tolist()
    )


@pytest.mark.unit
def test_min(dist):
    assert dist.min == np.min(dist._data_)


@pytest.mark.unit
def test_max(dist):
    assert dist.max == np.max(dist._data_)


@pytest.mark.unit
def test_range(dist):
    assert dist.range == np.max(dist._data_) - np.min(dist._data_)


@pytest.mark.unit
def test_describe(dist):
    description = dist.describe()
    assert description["mean"] == np.mean(dist._data_)
    assert description["median"] == np.median(dist._data_)
    assert description["variance"] == np.var(dist._data_)
    assert description["std_deviation"] == np.std(dist._data_)
    assert description["min"] == np.min(dist._data_)
    assert description["max"] == np.max(dist._data_)
    assert description["range"] == np.max(dist._data_) - np.min(dist._data_)
    assert (
        description["percentile_values"]
        == np.percentile(dist._data_, description["percentile_indices"]).tolist()
    )


@pytest.mark.regression
def test_add_data(dist):
    additional_data = [11, 12, 13]
    dist.add_data(additional_data)
    assert dist._data_[-3:] == additional_data
    assert len(dist._data_) == 13


@pytest.mark.regression
def test_remove_data(dist):
    remove_data = [1, 2, 3]
    dist.remove_data(remove_data)
    assert all(item not in dist._data_ for item in remove_data)
    assert len(dist._data_) == 7


@pytest.mark.end_to_end
def test_full_workflow():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    assert dist.mean == np.mean(data)

    dist.add_data([6, 7, 8])
    assert dist.mean == np.mean(dist._data_)

    dist.remove_data([1, 2])
    assert dist.mean == np.mean(dist._data_)
    assert len(dist._data_) == 6
