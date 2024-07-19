import pytest

from guidellm.core import Distribution


@pytest.mark.smoke
def test_distribution_initialization():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    assert dist.data == data


@pytest.mark.smoke
def test_distribution_statistics():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    assert dist.mean == 3.0
    assert dist.median == 3.0
    assert dist.variance == 2.0
    assert dist.std_deviation == pytest.approx(1.414213, rel=1e-5)
    assert dist.min == 1
    assert dist.max == 5
    assert dist.range == 4


@pytest.mark.sanity
def test_distribution_add_data():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    new_data = [6, 7, 8]
    dist.add_data(new_data)

    assert dist.data == data + new_data


@pytest.mark.sanity
def test_distribution_remove_data():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    remove_data = [2, 4]
    dist.remove_data(remove_data)
    assert dist.data == [1, 3, 5]


@pytest.mark.regression
def test_distribution_str():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    assert str(dist) == (
        "Distribution({'mean': 3.0, 'median': 3.0, "
        "'variance': 2.0, 'std_deviation': 1.4142135623730951, "
        "'percentile_indices': "
        "[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99], "
        "'percentile_values': "
        "[1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 4.8, 4.96], "
        "'min': 1, 'max': 5, 'range': 4})"
    )


@pytest.mark.regression
def test_distribution_repr():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    assert repr(dist) == f"Distribution(data={data})"


@pytest.mark.regression
def test_distribution_json():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    json_str = dist.to_json()
    assert '"data":[1,2,3,4,5]' in json_str

    dist_restored = Distribution.from_json(json_str)
    assert dist_restored.data == data


@pytest.mark.regression
def test_distribution_yaml():
    data = [1, 2, 3, 4, 5]
    dist = Distribution(data=data)
    yaml_str = dist.to_yaml()
    assert "data:\n- 1\n- 2\n- 3\n- 4\n- 5\n" in yaml_str

    dist_restored = Distribution.from_yaml(yaml_str)
    assert dist_restored.data == data
