from typing import get_args
from unittest.mock import create_autospec

import pytest

from guidellm import settings
from guidellm.core import (
    TextGenerationBenchmark,
    TextGenerationBenchmarkReport,
)
from guidellm.executor import Profile, ProfileGenerationMode, ProfileGenerator


@pytest.mark.smoke()
def test_profile_generator_mode():
    assert set(get_args(ProfileGenerationMode)) == {
        "sweep",
        "synchronous",
        "throughput",
        "constant",
        "poisson",
        "concurrent",
    }


@pytest.mark.smoke()
def test_profile_instantiation():
    profile = Profile(load_gen_mode="constant", load_gen_rate=10)
    assert profile.load_gen_mode == "constant"
    assert profile.load_gen_rate == 10
    assert profile.args == {}


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("mode", "rate"),
    [
        ("sweep", None),
        ("synchronous", None),
        ("throughput", None),
        ("constant", 10),
        ("constant", [10, 20, 30]),
        ("poisson", 10),
        ("poisson", [10, 20, 30]),
        ("concurrent", 2),
    ],
)
def test_profile_generator_instantiation(mode, rate):
    generator = ProfileGenerator(mode=mode, rate=rate)
    assert generator.mode == mode

    if rate is None:
        assert generator.rates is None
    elif isinstance(rate, list):
        assert generator.rates == rate
    else:
        assert generator.rates == [rate]

    if mode == "sweep":
        assert len(generator) == settings.num_sweep_profiles + 2
        assert (
            generator.profile_generation_modes
            == ["synchronous", "throughput"]
            + ["constant"] * settings.num_sweep_profiles
        )
    elif mode in ("throughput", "synchronous"):
        assert len(generator) == 1
        assert generator.profile_generation_modes == [mode]
    else:
        assert len(generator) == len(rate) if isinstance(rate, list) else 1
        assert generator.profile_generation_modes == [mode] * (
            len(rate) if isinstance(rate, list) else 1
        )

    assert generator.generated_count == 0


@pytest.mark.sanity()
@pytest.mark.parametrize(
    ("mode", "rate"),
    [
        # invalid modes
        ("invalid_mode", None),
        # rates supplied for non-applicable modes
        ("sweep", 10),
        ("sweep", [10, 20, 30]),
        ("synchronous", 10),
        ("synchronous", [10, 20, 30]),
        ("throughput", 10),
        ("throughput", [10, 20, 30]),
        # invalid rates supplied for applicable modes
        ("constant", None),
        ("constant", -1),
        ("constant", 0),
        ("poisson", None),
        ("poisson", -1),
        ("poisson", 0),
        ("concurrent", 0),
        ("concurrent", -1),
    ],
)
def test_profile_generator_invalid_instantiation(mode, rate):
    with pytest.raises(ValueError):
        ProfileGenerator(mode=mode, rate=rate)


@pytest.mark.sanity()
def test_profile_generator_next_sweep():
    generator = ProfileGenerator(mode="sweep")
    current_report = TextGenerationBenchmarkReport()

    for index in range(settings.num_sweep_profiles + 2):
        profile: Profile = generator.next(current_report)  # type: ignore

        if index == 0:
            assert profile.load_gen_mode == "synchronous"
            assert profile.load_gen_rate is None
            mock_benchmark = create_autospec(TextGenerationBenchmark, instance=True)
            mock_benchmark.completed_request_rate = 1
            current_report.add_benchmark(mock_benchmark)
        elif index == 1:
            assert profile.load_gen_mode == "throughput"
            assert profile.load_gen_rate is None
            mock_benchmark = create_autospec(TextGenerationBenchmark, instance=True)
            mock_benchmark.completed_request_rate = 10
            current_report.add_benchmark(mock_benchmark)
        else:
            assert profile.load_gen_mode == "constant"
            assert profile.load_gen_rate == index

        assert generator.generated_count == index + 1

    for _ in range(3):
        assert generator.next(current_report) is None


@pytest.mark.sanity()
def test_profile_generator_next_synchronous():
    generator = ProfileGenerator(mode="synchronous")
    current_report = TextGenerationBenchmarkReport()

    profile: Profile = generator.next(current_report)  # type: ignore
    assert profile.load_gen_mode == "synchronous"
    assert profile.load_gen_rate is None
    assert generator.generated_count == 1

    for _ in range(3):
        assert generator.next(current_report) is None


@pytest.mark.sanity()
def test_profile_generator_next_throughput():
    generator = ProfileGenerator(mode="throughput")
    current_report = TextGenerationBenchmarkReport()

    profile: Profile = generator.next(current_report)  # type: ignore
    assert profile.load_gen_mode == "throughput"
    assert profile.load_gen_rate is None
    assert generator.generated_count == 1

    for _ in range(3):
        assert generator.next(current_report) is None


@pytest.mark.sanity()
def test_profile_generator_next_concurrent():
    generator = ProfileGenerator(mode="concurrent", rate=2.0)
    current_report = TextGenerationBenchmarkReport()

    profile: Profile = generator.next(current_report)  # type: ignore
    assert profile.load_gen_mode == "concurrent"
    assert profile.load_gen_rate == 2
    assert generator.generated_count == 1

    for _ in range(3):
        assert generator.next(current_report) is None


@pytest.mark.sanity()
@pytest.mark.parametrize(
    "rate",
    [
        10,
        [10, 20, 30],
    ],
)
def test_profile_generator_next_constant(rate):
    generator = ProfileGenerator(mode="constant", rate=rate)
    test_rates = rate if isinstance(rate, list) else [rate]
    current_report = TextGenerationBenchmarkReport()

    for index, test_rate in enumerate(test_rates):
        profile: Profile = generator.next(current_report)  # type: ignore
        assert profile.load_gen_mode == "constant"
        assert profile.load_gen_rate == test_rate
        assert generator.generated_count == index + 1

    for _ in range(3):
        assert generator.next(current_report) is None


@pytest.mark.sanity()
@pytest.mark.parametrize(
    "rate",
    [
        10,
        [10, 20, 30],
    ],
)
def test_profile_generator_next_poisson(rate):
    generator = ProfileGenerator(mode="poisson", rate=rate)
    test_rates = rate if isinstance(rate, list) else [rate]
    current_report = TextGenerationBenchmarkReport()

    for index, test_rate in enumerate(test_rates):
        profile: Profile = generator.next(current_report)  # type: ignore
        assert profile.load_gen_mode == "poisson"
        assert profile.load_gen_rate == test_rate
        assert generator.generated_count == index + 1

    for _ in range(3):
        assert generator.next(current_report) is None
