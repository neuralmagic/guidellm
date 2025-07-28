import pytest

from guidellm.utils.text import camelize_str

@pytest.mark.smoke
def test_camelize_str():
  assert camelize_str("no_longer_snake_case") == "noLongerSnakeCase"