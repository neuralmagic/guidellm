# test_server_interaction.py

import json
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.vllm_sim_server import VllmSimServer


def get_guidellm_executable():
    """Get the path to the guidellm executable in the current environment."""
    # Get the directory where the current Python executable is located
    python_bin_dir = Path(sys.executable).parent
    guidellm_path = python_bin_dir / "guidellm"
    if guidellm_path.exists():
        return str(guidellm_path)
    else:
        # Fallback to just "guidellm" if not found
        return "guidellm"


@pytest.fixture(scope="module")
def server():
    """
    Pytest fixture to start and stop the server for the entire module
    using the TestServer class.
    """
    server = VllmSimServer(
        port=8000,
        model="databricks/dolly-v2-12b",
        mode="echo",
        time_to_first_token=1,  # 1ms TTFT
        inter_token_latency=1,  # 1ms ITL
    )
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(30)
def test_max_seconds_benchmark(server: VllmSimServer):
    """
    Another example test interacting with the server.
    """
    report_path = Path("tests/e2e/max_duration_benchmarks.json")
    rate = 10
    guidellm_exe = get_guidellm_executable()
    command = f"""
GUIDELLM__MAX_CONCURRENCY=10 GUIDELLM__MAX_WORKER_PROCESSES=10 {guidellm_exe} benchmark \
  --target "{server.get_url()}" \
  --rate-type constant \
  --rate {rate} \
  --max-seconds 1 \
  --data "prompt_tokens=256,output_tokens=128" \
  --processor "gpt2" \
  --output-path {report_path}
              """

    logger.info(f"Client command: {command}")
    os.system(command)  # noqa: S605

    assert report_path.exists()
    with report_path.open("r") as f:
        report = json.load(f)

    assert "benchmarks" in report
    benchmarks = report["benchmarks"]
    assert len(benchmarks) > 0
    benchmark = benchmarks[0]

    # Check that the max duration constraint was triggered
    assert "scheduler" in benchmark
    scheduler = benchmark["scheduler"]
    assert "state" in scheduler
    state = scheduler["state"]
    assert "end_processing_constraints" in state
    constraints = state["end_processing_constraints"]
    assert "max_duration" in constraints
    max_duration_constraint = constraints["max_duration"]
    assert "metadata" in max_duration_constraint
    metadata = max_duration_constraint["metadata"]
    assert "duration_exceeded" in metadata
    assert metadata["duration_exceeded"] is True

    assert "requests" in benchmark
    requests = benchmark["requests"]
    assert "successful" in requests
    successful = requests["successful"]
    assert len(successful) >= 1
    for request in successful:
        assert "request_latency" in request
        assert request["request_latency"] > 0
        # Streaming timing fields should now have proper values after fixing data transfer
        assert "time_to_first_token_ms" in request
        assert request["time_to_first_token_ms"] is not None
        assert request["time_to_first_token_ms"] > 0
        assert "time_per_output_token_ms" in request
        assert request["time_per_output_token_ms"] is not None
        assert request["time_per_output_token_ms"] > 0
        assert "inter_token_latency_ms" in request
        assert request["inter_token_latency_ms"] is not None
        assert request["inter_token_latency_ms"] > 0
        assert "tokens_per_second" in request
        assert request["tokens_per_second"] > 0
        assert "output_tokens_per_second" in request
        assert request["output_tokens_per_second"] > 0
        assert "total_tokens" in request
        assert request["total_tokens"] > 0
        assert "prompt_tokens" in request
        assert request["prompt_tokens"] > 0
        assert "output_tokens" in request
        assert request["output_tokens"] > 0

    if report_path.exists():
        report_path.unlink()


@pytest.mark.timeout(30)
def test_max_requests_benchmark(server: VllmSimServer):
    """
    Another example test interacting with the server.
    """
    report_path = Path("tests/e2e/max_number_benchmarks.json")
    rate = 10
    guidellm_exe = get_guidellm_executable()
    command = f"""
GUIDELLM__MAX_CONCURRENCY=10 GUIDELLM__MAX_WORKER_PROCESSES=10 {guidellm_exe} benchmark \
  --target "{server.get_url()}" \
  --rate-type constant \
  --rate {rate} \
  --max-requests {rate} \
  --data "prompt_tokens=256,output_tokens=128" \
  --processor "gpt2" \
  --output-path {report_path}
              """

    logger.info(f"Client command: {command}")
    os.system(command)  # noqa: S605

    assert report_path.exists()
    with report_path.open("r") as f:
        report = json.load(f)

    assert "benchmarks" in report
    benchmarks = report["benchmarks"]
    assert len(benchmarks) > 0
    benchmark = benchmarks[0]

    # Check that the max number constraint was triggered
    assert "scheduler" in benchmark
    scheduler = benchmark["scheduler"]
    assert "state" in scheduler
    state = scheduler["state"]
    assert "end_processing_constraints" in state
    constraints = state["end_processing_constraints"]
    assert "max_number" in constraints
    max_number_constraint = constraints["max_number"]
    assert "metadata" in max_number_constraint
    metadata = max_number_constraint["metadata"]
    assert "processed_exceeded" in metadata
    assert metadata["processed_exceeded"] is True

    assert "requests" in benchmark
    requests = benchmark["requests"]
    assert "successful" in requests
    successful = requests["successful"]
    assert len(successful) == rate
    for request in successful:
        assert "request_latency" in request
        assert request["request_latency"] > 0
        # Streaming timing fields should now have proper values after fixing data transfer
        assert "time_to_first_token_ms" in request
        assert request["time_to_first_token_ms"] is not None
        assert request["time_to_first_token_ms"] > 0
        assert "time_per_output_token_ms" in request
        assert request["time_per_output_token_ms"] is not None
        assert request["time_per_output_token_ms"] > 0
        assert "inter_token_latency_ms" in request
        assert request["inter_token_latency_ms"] is not None
        assert request["inter_token_latency_ms"] > 0
        assert "tokens_per_second" in request
        assert request["tokens_per_second"] > 0
        assert "output_tokens_per_second" in request
        assert request["output_tokens_per_second"] > 0
        assert "total_tokens" in request
        assert request["total_tokens"] > 0
        assert "prompt_tokens" in request
        assert request["prompt_tokens"] > 0
        assert "output_tokens" in request
        assert request["output_tokens"] > 0

    if report_path.exists():
        report_path.unlink()
