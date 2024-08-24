import pytest

from guidellm.utils import BenchmarkReportProgress


@pytest.fixture()
def benchmark_progress():
    return BenchmarkReportProgress()


@pytest.mark.smoke()
def test_initialization(benchmark_progress):
    assert benchmark_progress.report_task is None
    assert benchmark_progress.benchmark_tasks == []
    assert benchmark_progress.benchmark_tasks_started == []
    assert benchmark_progress.benchmark_tasks_completed == []
    assert benchmark_progress.benchmark_tasks_progress == []


@pytest.mark.smoke()
def test_start_method(benchmark_progress):
    descriptions = ["Benchmark 1", "Benchmark 2"]
    benchmark_progress.start(descriptions)

    assert len(benchmark_progress.benchmark_tasks) == 2
    assert benchmark_progress.report_task is not None

    benchmark_progress.finish()


@pytest.mark.sanity()
def test_update_benchmark(benchmark_progress):
    descriptions = ["Benchmark 1"]
    benchmark_progress.start(descriptions)

    benchmark_progress.update_benchmark(
        index=0,
        description="Updating Benchmark 1",
        completed=False,
        completed_count=50,
        completed_total=100,
        start_time=0,
        req_per_sec=10.5,
    )
    assert benchmark_progress.benchmark_tasks_progress[0] == 50.0

    benchmark_progress.finish()


@pytest.mark.sanity()
def test_finish_method(benchmark_progress):
    descriptions = ["Benchmark 1", "Benchmark 2"]
    benchmark_progress.start(descriptions)
    benchmark_progress.finish()

    assert benchmark_progress.report_progress.finished


@pytest.mark.regression()
def test_error_on_update_completed_benchmark(benchmark_progress):
    descriptions = ["Benchmark 1"]
    benchmark_progress.start(descriptions)
    benchmark_progress.update_benchmark(
        index=0,
        description="Benchmark 1",
        completed=True,
        completed_count=100,
        completed_total=100,
        start_time=0,
        req_per_sec=10.5,
    )

    with pytest.raises(ValueError, match="already completed"):
        benchmark_progress.update_benchmark(
            index=0,
            description="Benchmark 1",
            completed=False,
            completed_count=50,
            completed_total=100,
            start_time=0,
            req_per_sec=10.5,
        )

    benchmark_progress.finish()


@pytest.mark.regression()
def test_multiple_updates(benchmark_progress):
    descriptions = ["Benchmark 1", "Benchmark 2"]
    benchmark_progress.start(descriptions)

    # First update
    benchmark_progress.update_benchmark(
        index=0,
        description="Updating Benchmark 1",
        completed=False,
        completed_count=50,
        completed_total=100,
        start_time=0,
        req_per_sec=5.0,
    )
    assert benchmark_progress.benchmark_tasks_progress[0] == 50.0

    # Second update, same task
    benchmark_progress.update_benchmark(
        index=0,
        description="Updating Benchmark 1",
        completed=True,
        completed_count=100,
        completed_total=100,
        start_time=0,
        req_per_sec=5.0,
    )
    assert benchmark_progress.benchmark_tasks_progress[0] == 100.0

    benchmark_progress.finish()
