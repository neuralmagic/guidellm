import pytest
import time
from guidellm.scheduler.load_generator import LoadGenerator, LoadGenerationModes


def test_load_generator_constant():
    rate = 2.0  # 2 requests per second
    generator = LoadGenerator(LoadGenerationModes.CONSTANT, rate)

    times = generator.times()
    start_time = next(times)
    time_increment = 1.0 / rate

    assert start_time > 0  # Check that start_time is a valid timestamp

    for i in range(1, 10):
        next_time = next(times)
        assert next_time == pytest.approx(start_time + i * time_increment, rel=1e-2)


def test_load_generator_poisson():
    rate = 2.0  # Average of 2 requests per second
    generator = LoadGenerator(LoadGenerationModes.POISSON, rate)

    times = generator.times()
    start_time = next(times)

    assert start_time > 0  # Check that start_time is a valid timestamp

    previous_time = start_time
    for _ in range(1, 10):
        next_time = next(times)
        interval = next_time - previous_time
        assert interval >= 0  # Ensure no negative intervals
        previous_time = next_time


def test_load_generator_invalid_mode():
    with pytest.raises(ValueError) as excinfo:
        LoadGenerator(LoadGenerationModes.SYNCHRONOUS, 1.0)
    assert "Synchronous mode not supported by LoadGenerator" in str(excinfo.value)


def test_load_generator_invalid_mode_times():
    generator = LoadGenerator(LoadGenerationModes.CONSTANT, 2.0)
    with pytest.raises(ValueError) as excinfo:
        generator.times()
    assert "Synchronous mode not supported by LoadGenerator" in str(excinfo.value)
