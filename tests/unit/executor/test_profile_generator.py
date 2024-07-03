import pytest

from guidellm.executor import (ProfileGenerator, FixedRateProfileGenerator, SweepProfileGenerator)

def test_invalid_profile_generation_mode_error():
  rate = [1]
  rate_type = "constant"
  profile_mode = "burst"
  with pytest.raises(ValueError, match=f"Invalid profile generation mode: {profile_mode}"):
    ProfileGenerator.create_generator(profile_mode, **({ "rate": rate, "rate_type": rate_type}))

def test_sweep_profile_generator_creation():
  profile = ProfileGenerator.create_generator("sweep", **({}))
  assert isinstance(profile, SweepProfileGenerator)
  assert profile._sync_run == False
  assert profile._max_found == False
  assert profile._pending_rates == None
  assert profile._pending_rates == None

def test_fixed_rate_profile_generator_creation():
  rate = [1]
  rate_type = "constant"
  profile = ProfileGenerator.create_generator("fixed_rate", **({ "rate": rate, "rate_type": rate_type}))
  assert isinstance(profile, FixedRateProfileGenerator)
  assert profile._rates == rate
  assert profile._rate_type == rate_type
  assert profile._rate_index == 0
  assert profile._rate_index == 0