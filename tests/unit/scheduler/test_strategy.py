import pytest
import asyncio
import math
from itertools import islice

from guidellm.scheduler.strategy import AsyncConstantStrategy

# -------------------------------------------------------------------
# Stub out asyncio.sleep so we don't actually wait
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fast_asyncio_sleep(monkeypatch):
    async def dummy_sleep(delay: float):
        return
    monkeypatch.setattr(asyncio, "sleep", dummy_sleep)

# -------------------------------------------------------------------
# Helper: pull N+1 timestamps after skipping the initial burst
# -------------------------------------------------------------------
def collect_times_after_burst(strategy: AsyncConstantStrategy, n: int):
    gen = strategy.request_times()
    # skip the initial burst
    # burst_count = math.floor(strategy.rate)
    # for _ in range(burst_count):
    #     next(gen)
    # now grab n+1 timestamps
    return list(islice(gen, n + 1))

# -------------------------------------------------------------------
# 1) Each inter-arrival ≈ 1/rate
# -------------------------------------------------------------------
def test_intervals_close_to_expected():
    rate = 5.0
    strategy = AsyncConstantStrategy(rate=rate)

    N = 20
    times = collect_times_after_burst(strategy, N)
    intervals = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    expected = 1.0 / rate

    for dt in intervals:
        assert dt == pytest.approx(expected, rel=1)

# -------------------------------------------------------------------
# 2) Over many events, average RPS ≈ target
# -------------------------------------------------------------------
def test_average_rate_over_many_requests():
    rate = 10.0
    strategy = AsyncConstantStrategy(rate=rate)

    M = 1000
    times = collect_times_after_burst(strategy, M)
    total_time = times[-1] - times[0]
    measured = M / total_time
    assert measured == pytest.approx(rate, rel=1)

# -------------------------------------------------------------------
# 3) Parameterized: various rates
# -------------------------------------------------------------------
@pytest.mark.parametrize("rate,tol", [
    (1.0, 0.02),
    (2.5, 0.02),
    (50.0, 0.05),
])
def test_various_rates(rate, tol):
    strategy = AsyncConstantStrategy(rate=rate)

    K = 200
    times = collect_times_after_burst(strategy, K)
    total_time = times[-1] - times[0]
    measured = K / total_time

    assert measured == pytest.approx(rate, rel=tol)

# -------------------------------------------------------------------
# 4) Invalid rates should raise immediately
# -------------------------------------------------------------------
def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        AsyncConstantStrategy(rate=0)
    with pytest.raises(ValueError):
        AsyncConstantStrategy(rate=-1)
