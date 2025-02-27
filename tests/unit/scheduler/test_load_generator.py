import time
from typing import get_args

import pytest
from scipy.stats import kstest  # type: ignore

from guidellm.scheduler import LoadGenerationMode, LoadGenerator


@pytest.mark.smoke()
def test_load_generator_mode():
    assert set(get_args(LoadGenerationMode)) == {
        "synchronous",
        "constant",
        "poisson",
        "throughput",
        "concurrent",
    }


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("mode", "rate"),
    [
        ("constant", 10),
        ("poisson", 5),
        ("throughput", None),
        ("synchronous", None),
        ("concurrent", 3),
    ],
)
def test_load_generator_instantiation(mode, rate):
    generator = LoadGenerator(mode=mode, rate=rate)
    assert generator.mode == mode
    assert generator.rate == rate


@pytest.mark.regression()
@pytest.mark.parametrize(
    ("mode", "rate", "expected_error"),
    [
        ("invalid_mode", None, ValueError),
        ("constant", 0, ValueError),
        ("poisson", -1, ValueError),
        ("concurrent", -1, ValueError),
        ("concurrent", 0, ValueError),
    ],
)
def test_load_generator_invalid_instantiation(mode, rate, expected_error):
    with pytest.raises(expected_error):
        LoadGenerator(mode=mode, rate=rate)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("mode", "rate"),
    [
        ("synchronous", None),
        ("throughput", None),
        ("constant", 1),
        ("poisson", 5),
    ],
)
def test_load_generator_times(mode, rate):
    # first check that the proper method is called
    generator = LoadGenerator(mode=mode, rate=rate)
    func_name = f"{mode}_times"
    assert hasattr(generator, func_name)
    assert callable(getattr(generator, func_name))

    call_count = 0

    def _increment_call_count():
        nonlocal call_count
        call_count += 1
        yield -1.0

    setattr(generator, func_name, _increment_call_count)
    for time_ in generator.times():
        assert time_ == -1.0
        break
    assert call_count == 1

    # now check that the method generates reasonable timestamps
    generator = LoadGenerator(mode=mode, rate=rate)
    start_time = time.time()
    for index, time_ in enumerate(generator.times()):
        if index > 10:
            break

        if mode == "synchronous":
            assert time_ == -1.0
        else:
            assert time_ >= start_time


@pytest.mark.smoke()
def test_load_generator_invalid_times():
    generator = LoadGenerator(mode="synchronous")

    for index, time_ in enumerate(generator.synchronous_times()):
        if index > 10:
            break

        assert time_ == -1.0


@pytest.mark.smoke()
def test_load_generator_throughput_times():
    generator = LoadGenerator(mode="throughput")

    for index, time_ in enumerate(generator.throughput_times()):
        if index > 10:
            break

        assert time_ <= time.time()


@pytest.mark.smoke()
@pytest.mark.parametrize("rate", [1, 10, 42])
def test_load_generator_constant_times(rate):
    generator = LoadGenerator(mode="constant", rate=rate)
    start_time = time.time()

    for index, time_ in enumerate(generator.constant_times()):
        if index > 10:
            break

        assert time_ == pytest.approx(start_time + index / rate, rel=1e-5)


@pytest.mark.smoke()
@pytest.mark.flaky(reruns=5)
def test_load_generator_poisson_times():
    rate = 5
    generator = LoadGenerator(mode="poisson", rate=rate)
    start_time = time.time()

    times = []
    prev_time = start_time

    for index, current_time in enumerate(generator.poisson_times()):
        if index > 100:
            break

        times.append(current_time - prev_time)
        prev_time = current_time

    mean_inter_arrival_time = 1 / rate

    # Perform Kolmogorov-Smirnov test to compare the sample distribution
    # to the expected exponential distribution
    ks_statistic, p_value = kstest(times, "expon", args=(0, mean_inter_arrival_time))
    assert p_value > 0.025, (
        f"Poisson-generated inter-arrival times do not "
        f"match the expected exponential distribution (p-value: {p_value})"
    )
