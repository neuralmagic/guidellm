import numpy
import pytest
from unittest.mock import MagicMock
from guidellm.executor import (ProfileGenerator, FixedRateProfileGenerator, SweepProfileGenerator)
from src.guidellm.core.result import TextGenerationBenchmark, TextGenerationBenchmarkReport
from src.guidellm.executor import profile_generator
from src.guidellm.scheduler.load_generator import LoadGenerationModes

def test_invalid_profile_generation_mode_error():
  rate = [1]
  rate_type = "constant"
  profile_mode = "burst"
  with pytest.raises(ValueError, match=f"Invalid profile generation mode: {profile_mode}"):
    ProfileGenerator.create_generator(profile_mode, **({ "rate": rate, "rate_type": rate_type}))

# Fixed Rate Profile Generator

def test_fixed_rate_profile_generator_creation():
  rate = [1]
  rate_type = "constant"
  profile_generator = ProfileGenerator.create_generator("fixed_rate", **({ "rate": rate, "rate_type": rate_type}))
  assert isinstance(profile_generator, FixedRateProfileGenerator)
  assert profile_generator._rates == rate
  assert profile_generator._rate_type == rate_type
  assert profile_generator._rate_index == 0
  assert profile_generator._rate_index == 0

def test_synchronous_mode_rate_list_error():
  rate = [1]
  rate_type = "synchronous"
  with pytest.raises(ValueError, match="custom rates are not supported in synchronous mode"):
    ProfileGenerator.create_generator("fixed_rate", **({ "rate": rate, "rate_type": rate_type}))

def test_next_profile_with_multiple_rates():
  rates = [1, 2]
  rate_type = "constant"
  profile_generator = ProfileGenerator.create_generator("fixed_rate", **({ "rate": rates, "rate_type": rate_type}))
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  for rate in rates:
    current_profile = profile_generator.next_profile(mock_report)
    assert current_profile.load_gen_rate == rate
    assert current_profile.load_gen_mode.name == LoadGenerationModes.CONSTANT.name
  assert profile_generator.next_profile(mock_report) == None

def test_next_profile_with_sync_mode():
  rate_type = "synchronous"
  profile_generator = ProfileGenerator.create_generator("fixed_rate", **({ "rate_type": rate_type}))
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  current_profile = profile_generator.next_profile(mock_report)
  assert current_profile.load_gen_rate == None
  assert current_profile.load_gen_mode.name == LoadGenerationModes.SYNCHRONOUS.name
  assert profile_generator.next_profile(mock_report) == None

# Sweep Profile Generator

def test_sweep_profile_generator_creation():
  profile_generator = ProfileGenerator.create_generator("sweep", **({}))
  assert isinstance(profile_generator, SweepProfileGenerator)
  assert profile_generator._sync_run == False
  assert profile_generator._max_found == False
  assert profile_generator._pending_rates == None
  assert profile_generator._pending_rates == None

def test_first_profile_is_synchronous():
  profile_generator = ProfileGenerator.create_generator("sweep")
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  profile = profile_generator.next_profile(mock_report)
  assert profile.load_gen_rate == None
  assert profile.load_gen_mode.name == LoadGenerationModes.SYNCHRONOUS.name

def test_rate_doubles():
  profile_generator = ProfileGenerator.create_generator("sweep")
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_benchmark.overloaded = False
  mock_benchmark.args_rate = 2.0
  mock_benchmark.request_rate = 2.0
  benchmarks = [
        mock_benchmark
    ]
  mock_report.benchmarks = benchmarks
  profile = profile_generator.next_profile(mock_report)

  profile = profile_generator.next_profile(mock_report)
  assert profile.load_gen_rate == 4.0

def test_max_found():
  profile_generator = ProfileGenerator.create_generator("sweep")
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_benchmark.overloaded = False
  mock_benchmark.args_rate = 2.0
  mock_benchmark.request_rate = 2.0
  mock_overloaded_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_overloaded_benchmark.overloaded = True
  mock_overloaded_benchmark.args_rate = 4.0
  mock_overloaded_benchmark.request_rate = 4.0
  benchmarks = [
        mock_benchmark,
        mock_overloaded_benchmark
    ]
  mock_report.benchmarks = benchmarks

  profile_generator.next_profile(mock_report)
  profile = profile_generator.next_profile(mock_report)

  # if benchmark wasn't overloaded, rate would have doubled to 8
  assert profile.load_gen_rate == 2.0

def test_pending_rates():
  profile_generator = ProfileGenerator.create_generator("sweep")
  mock_report = MagicMock(spec=TextGenerationBenchmarkReport)
  mock_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_benchmark.overloaded = False
  mock_benchmark.args_rate = 2.0
  mock_benchmark.request_rate = 2.0
  mock_overloaded_benchmark = MagicMock(spec=TextGenerationBenchmark)
  mock_overloaded_benchmark.overloaded = True
  mock_overloaded_benchmark.args_rate = 8.0
  mock_overloaded_benchmark.request_rate = 8.0
  benchmarks = [
        mock_benchmark,
        mock_overloaded_benchmark
    ]
  mock_report.benchmarks = benchmarks
  profile = profile_generator.next_profile(mock_report)
  for expected_rate in numpy.linspace(2.0, 8.0, 10):
        profile = profile_generator.next_profile(mock_report)
        assert profile.load_gen_rate == expected_rate