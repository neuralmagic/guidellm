import pytest
from click import BadParameter

from guidellm.utils import cli_params


@pytest.fixture
def max_requests_param_type():
    return cli_params.MaxRequestsType()


def test_valid_integer_input(max_requests_param_type):
    assert max_requests_param_type.convert(10, None, None) == 10
    assert max_requests_param_type.convert("42", None, None) == 42


def test_valid_dataset_input(max_requests_param_type):
    assert max_requests_param_type.convert("dataset", None, None) == "dataset"


def test_invalid_string_input(max_requests_param_type):
    with pytest.raises(BadParameter):
        max_requests_param_type.convert("invalid", None, None)


def test_invalid_float_input(max_requests_param_type):
    with pytest.raises(BadParameter):
        max_requests_param_type.convert("10.5", None, None)


def test_invalid_non_numeric_string_input(max_requests_param_type):
    with pytest.raises(BadParameter):
        max_requests_param_type.convert("abc", None, None)


def test_invalid_mixed_string_input(max_requests_param_type):
    with pytest.raises(BadParameter):
        max_requests_param_type.convert("123abc", None, None)
