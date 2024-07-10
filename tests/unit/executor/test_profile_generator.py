import pytest

from domain.executor import (
    ProfileGenerationMode,
    ProfileGenerator,
    SingleProfileGenerator,
    SweepProfileGenerator,
)


@pytest.mark.smoke
def test_profile_generator_registry():
    """
    Ensure that all registered classes exist in the Backend._registry.
    """

    assert ProfileGenerator._registry == {
        ProfileGenerationMode.SINGLE: SingleProfileGenerator,
        ProfileGenerationMode.SWEEP: SweepProfileGenerator,
    }
