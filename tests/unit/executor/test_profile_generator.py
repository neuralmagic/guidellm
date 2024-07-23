import numpy
import pytest
from unittest.mock import MagicMock
from guidellm.executor import (ProfileGenerator, FixedRateProfileGenerator, SweepProfileGenerator)
from guidellm.core import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.executor import profile_generator, ProfileGenerationMode
from guidellm.scheduler import LoadGenerationMode

# Fixed Rate Profile Generator

def test_fixed_rate_profile_generator_creation():
  rates = [1]
  load_gen_mode = LoadGenerationMode.CONSTANT
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.FIXED_RATE, **({ "rates": rates, "load_gen_mode": load_gen_mode}))
  assert isinstance(profile_generator, FixedRateProfileGenerator)
  assert profile_generator._rates == rates
  assert profile_generator._load_gen_mode.name == load_gen_mode.name
  assert profile_generator._rate_index == 0

def test_synchronous_mode_rate_list_error():
  rates = [1]
  load_gen_mode = LoadGenerationMode.SYNCHRONOUS
  with pytest.raises(ValueError, match="custom rates are not supported in synchronous mode"):
    ProfileGenerator.create(ProfileGenerationMode.FIXED_RATE, **({ "rates": rates, "load_gen_mode": load_gen_mode}))

def test_next_with_multiple_rates():
  rates = [1, 2]
  load_gen_mode = LoadGenerationMode.CONSTANT
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.FIXED_RATE, **({ "rates": rates, "load_gen_mode": load_gen_mode}))
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  for rates in rates:
    current_profile = profile_generator.next(mock_report)
    assert current_profile.load_gen_rate == rates
    assert current_profile.load_gen_mode.name == LoadGenerationMode.CONSTANT.name
  assert profile_generator.next(mock_report) == None

def test_next_with_sync_mode():
  load_gen_mode = LoadGenerationMode.SYNCHRONOUS
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.FIXED_RATE, **({ "load_gen_mode": load_gen_mode}))
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  current_profile = profile_generator.next(mock_report)
  assert current_profile.load_gen_rate == None
  assert current_profile.load_gen_mode.name == LoadGenerationMode.SYNCHRONOUS.name
  assert profile_generator.next(mock_report) == None

# Sweep Profile Generator

def test_sweep_profile_generator_creation():
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP, **({}))
  assert isinstance(profile_generator, SweepProfileGenerator)
  assert profile_generator._sync_run == False
  assert profile_generator._max_found == False
  assert profile_generator._pending_rates == None
  assert profile_generator._pending_rates == None

def test_first_profile_is_synchronous():
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  profile = profile_generator.next(mock_report)
  assert profile.load_gen_rate == None
  assert profile.load_gen_mode.name == LoadGenerationMode.SYNCHRONOUS.name

def test_rate_doubles():
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_benchmark.overloaded = False
  mock_benchmark.rate = 2.0
  mock_benchmark.request_rate = 2.0
  benchmarks = [
        mock_benchmark
    ]
  mock_report.benchmarks = benchmarks
  profile = profile_generator.next(mock_report)

  profile = profile_generator.next(mock_report)
  assert profile.load_gen_rate == 4.0

def test_max_found():
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_benchmark.overloaded = False
  mock_benchmark.rate = 2.0
  mock_benchmark.request_rate = 2.0
  mock_overloaded_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_overloaded_benchmark.overloaded = True
  mock_overloaded_benchmark.rate = 4.0
  mock_overloaded_benchmark.request_rate = 4.0
  benchmarks = [
        mock_benchmark,
        mock_overloaded_benchmark
    ]
  mock_report.benchmarks = benchmarks

  profile_generator.next(mock_report)
  profile = profile_generator.next(mock_report)

  # if benchmark wasn't overloaded, rates would have doubled to 8
  assert profile.load_gen_rate == 2.0

def test_pending_rates():
  profile_generator = ProfileGenerator.create(ProfileGenerationMode.SWEEP)
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_benchmark.overloaded = False
  mock_benchmark.rate = 2.0
  mock_benchmark.request_rate = 2.0
  mock_overloaded_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_overloaded_benchmark.overloaded = True
  mock_overloaded_benchmark.rate = 8.0
  mock_overloaded_benchmark.request_rate = 8.0
  benchmarks = [
        mock_benchmark,
        mock_overloaded_benchmark
    ]
  mock_report.benchmarks = benchmarks
  profile = profile_generator.next(mock_report)
  for expected_rate in numpy.linspace(2.0, 8.0, 10):
        profile = profile_generator.next(mock_report)
        assert profile.load_gen_rate == expected_rate