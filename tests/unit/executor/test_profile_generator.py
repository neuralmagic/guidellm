from unittest.mock import MagicMock

import numpy
import pytest

from guidellm.core import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.executor import (
    FixedRateProfileGenerator,
    ProfileGenerationMode,
    ProfileGenerator,
    SweepProfileGenerator,
)
from guidellm.scheduler import LoadGenerationMode

# Fixed Rate Profile Generator


def test_fixed_rate_profile_generator_creation():
    rates = [1.0]
    load_gen_mode = LoadGenerationMode.CONSTANT
    test_profile_generator = ProfileGenerator.create(
        ProfileGenerationMode.FIXED_RATE,
        **({"rates": rates, "load_gen_mode": load_gen_mode}),
    )
    assert isinstance(test_profile_generator, FixedRateProfileGenerator)
    assert test_profile_generator._rates == rates
    assert test_profile_generator._load_gen_mode == load_gen_mode
    assert test_profile_generator._rate_index == 0


def test_synchronous_mode_rate_list_error():
    rates = [1.0]
    load_gen_mode = LoadGenerationMode.SYNCHRONOUS
    with pytest.raises(
        ValueError, match="custom rates are not supported in synchronous mode"
    ):
        ProfileGenerator.create(
            ProfileGenerationMode.FIXED_RATE,
            **({"rates": rates, "load_gen_mode": load_gen_mode}),
        )


def test_next_with_multiple_rates():
    rates = [1.0, 2.0]
    load_gen_mode = LoadGenerationMode.CONSTANT
    test_profile_generator = ProfileGenerator.create(
        ProfileGenerationMode.FIXED_RATE,
        **({"rates": rates, "load_gen_mode": load_gen_mode}),
    )
    mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
    for rate in rates:
        current_profile = test_profile_generator.next(mock_report)
        assert current_profile is not None
        assert current_profile.load_gen_rate == rate
        assert current_profile.load_gen_mode == LoadGenerationMode.CONSTANT
    assert test_profile_generator.next(mock_report) is None


def test_next_with_sync_mode():
    load_gen_mode = LoadGenerationMode.SYNCHRONOUS
    test_profile_generator = ProfileGenerator.create(
        ProfileGenerationMode.FIXED_RATE, **({"load_gen_mode": load_gen_mode})
    )
    mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
    current_profile = test_profile_generator.next(mock_report)
    assert current_profile is not None
    assert current_profile.load_gen_rate is None
    assert current_profile.load_gen_mode == LoadGenerationMode.SYNCHRONOUS
    assert test_profile_generator.next(mock_report) is None


# Sweep Profile Generator


def test_sweep_profile_generator_creation():
    test_profile_generator = ProfileGenerator.create(
        ProfileGenerationMode.SWEEP, **({})
    )
    assert isinstance(test_profile_generator, SweepProfileGenerator)
    assert not test_profile_generator._sync_run
    assert not test_profile_generator._max_found
    assert test_profile_generator._pending_rates is None
    assert test_profile_generator._pending_rates is None


def test_first_profile_is_synchronous():
    test_profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
    mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
    profile = test_profile_generator.next(mock_report)
    assert profile is not None
    assert profile.load_gen_rate is None
    assert profile.load_gen_mode == LoadGenerationMode.SYNCHRONOUS


def test_rate_doubles():
    test_profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
    mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
    mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
    mock_benchmark.overloaded = False
    mock_benchmark.rate = 2.0
    mock_benchmark.request_rate = 2.0
    benchmarks = [mock_benchmark]
    mock_report.benchmarks = benchmarks
    test_profile_generator.next(mock_report)

    profile = test_profile_generator.next(mock_report)
    assert profile is not None
    assert profile.load_gen_rate == 4.0


def test_max_found():
    test_profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
    mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
    mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
    mock_benchmark.overloaded = False
    mock_benchmark.rate = 2.0
    mock_benchmark.request_rate = 2.0
    mock_overloaded_benchmark = MagicMock(spec=TextGenerationBenchmark)
    mock_overloaded_benchmark.overloaded = True
    mock_overloaded_benchmark.rate = 4.0
    mock_overloaded_benchmark.request_rate = 4.0
    benchmarks = [mock_benchmark, mock_overloaded_benchmark]
    mock_report.benchmarks = benchmarks

    test_profile_generator.next(mock_report)
    profile = test_profile_generator.next(mock_report)
    assert profile is not None

    # if benchmark wasn't overloaded, rates would have doubled to 8
    assert profile.load_gen_rate == 2.0


def test_pending_rates():
    test_profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
    mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
    mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
    mock_benchmark.overloaded = False
    mock_benchmark.rate = 2.0
    mock_benchmark.request_rate = 2.0
    mock_overloaded_benchmark = MagicMock(spec=TextGenerationBenchmark)
    mock_overloaded_benchmark.overloaded = True
    mock_overloaded_benchmark.rate = 8.0
    mock_overloaded_benchmark.request_rate = 8.0
    benchmarks = [mock_benchmark, mock_overloaded_benchmark]
    mock_report.benchmarks = benchmarks
    profile = test_profile_generator.next(mock_report)
    assert profile is not None
    for expected_rate in numpy.linspace(2.0, 8.0, 10):
        profile = test_profile_generator.next(mock_report)
        assert profile.load_gen_rate == expected_rate
