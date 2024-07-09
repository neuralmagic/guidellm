import pytest
from unittest.mock import MagicMock
from guidellm.executor import (ProfileGenerator, FixedRateProfileGenerator, SweepProfileGenerator)
from src.guidellm.core.result import TextGenerationBenchmarkReport
from src.guidellm.scheduler.load_generator import LoadGenerationModes

def test_invalid_profile_generation_mode_error():
  rate = [1]
  rate_type = "constant"
  profile_mode = "burst"
  with pytest.raises(ValueError, match=f"Invalid profile generation mode: {profile_mode}"):
    ProfileGenerator.create_generator(profile_mode, **({ "rate": rate, "rate_type": rate_type}))

def test_sweep_profile_generator_creation():
  profile_generator = ProfileGenerator.create_generator("sweep", **({}))
  assert isinstance(profile_generator, SweepProfileGenerator)
  assert profile_generator._sync_run == False
  assert profile_generator._max_found == False
  assert profile_generator._pending_rates == None
  assert profile_generator._pending_rates == None

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