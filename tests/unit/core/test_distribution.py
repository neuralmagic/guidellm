import pytest

from domain.core import Distribution


@pytest.mark.smoke
def test_distribution_initialization():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    assert dist.data == data


@pytest.mark.sanity
def test_distribution_statistics():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    assert dist.mean == 3.0
    assert dist.median == 3.0
    assert dist.variance == 2.0
    assert dist.std_deviation == pytest.approx(1.414213, rel=1e-5)
    assert dist.min == 1
    assert dist.max == 5
    assert dist.range == 4


@pytest.mark.regression
def test_distribution_str():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    assert str(dist) == "Distribution(mean=3.00, median=3.00, min=1, max=5, count=5)"


@pytest.mark.regression
def test_distribution_repr():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    assert repr(dist) == f"Distribution(data={data})"


@pytest.mark.regression
def test_distribution_add_data():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    new_data = [6, 7, 8]
    dist.add_data(new_data)

    assert dist.data == data + new_data


@pytest.mark.regression
def test_distribution_remove_data():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data)
    remove_data = [2, 4]
    dist.remove_data(remove_data)
    assert dist.data == [1, 3, 5]
