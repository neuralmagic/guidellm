import pytest

from guidellm.config import settings


@pytest.mark.smoke()
def test_import():
    assert settings
