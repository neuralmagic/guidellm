import pytest
from time import sleep
from guidellm.core.distribution import Distribution
from guidellm.core.result import BenchmarkResult


@pytest.mark.unit
def test_benchmark_result_initialization():
    result = BenchmarkResult("test_id")
    assert result.id == "test_id"
    assert result.prompt == ""
    assert result.output == ""
    assert result.prompt_word_count == 0
    assert result.prompt_token_count == 0
    assert result.output_word_count == 0
    assert result.output_token_count == 0
    assert result.start_time is None
    assert result.end_time is None
    assert result.first_token_time is None
    assert isinstance(result.decode_times, Distribution)


@pytest.mark.unit
def test_benchmark_result_start():
    result = BenchmarkResult("test_id")
    result.start("Test prompt.")
    assert result.prompt == "Test prompt."
    assert result.prompt_word_count == 2
    assert result.prompt_token_count == 12  # Placeholder for token count
    assert result.start_time is not None


@pytest.mark.unit
def test_benchmark_result_add_token():
    result = BenchmarkResult("test_id")
    result.start("Test prompt.")
    sleep(0.1)  # Simulate delay
    result.add_token("Token1")
    assert result.output == "Token1"
    assert result.output_word_count == 1
    assert result.output_token_count == 6  # Placeholder for token count
    assert result.first_token_time is not None
    assert len(result.decode_times.data_) == 1


@pytest.mark.unit
def test_benchmark_result_end_with_default_token_counts():
    result = BenchmarkResult("test_id")
    result.start("Test prompt.")
    sleep(0.1)  # Simulate delay
    result.add_token("Token1")
    sleep(0.1)  # Simulate delay
    result.end("Final output.")
    assert result.output == "Final output."
    assert result.output_word_count == 2
    assert result.output_token_count == 2  # Defaults to word count
    assert result.prompt_token_count == 2  # Defaults to word count
    assert result.end_time is not None


@pytest.mark.unit
def test_benchmark_result_end_with_supplied_token_counts():
    result = BenchmarkResult("test_id")
    result.start("Test prompt.")
    sleep(0.1)  # Simulate delay
    result.add_token("Token1")
    sleep(0.1)  # Simulate delay
    result.end("Final output.", prompt_token_count=5, output_token_count=7)
    assert result.output == "Final output."
    assert result.output_word_count == 2
    assert result.output_token_count == 7
    assert result.prompt_token_count == 5
    assert result.end_time is not None


@pytest.mark.unit
def test_benchmark_result_str():
    result = BenchmarkResult("test_id")
    result.start("Test prompt.")
    sleep(0.1)  # Simulate delay
    result.add_token("Token1")
    sleep(0.1)  # Simulate delay
    result.end("Final output.")
    expected_str = (
        f"BenchmarkResult(id=test_id, prompt='Test prompt.', "
        f"output='Final output.', start_time={result.start_time}, "
        f"end_time={result.end_time}, first_token_time={result.first_token_time})"
    )
    assert str(result) == expected_str


@pytest.mark.unit
def test_benchmark_result_repr():
    result = BenchmarkResult("test_id")
    result.start("Test prompt.")
    sleep(0.1)  # Simulate delay
    result.add_token("Token1")
    sleep(0.1)  # Simulate delay
    result.end("Final output.")
    expected_repr = (
        f"BenchmarkResult(id=test_id, prompt='Test prompt.', "
        f"prompt_word_count=2, prompt_token_count=2, "
        f"output='Final output.', output_word_count=2, "
        f"output_token_count=2, start_time={result.start_time}, "
        f"end_time={result.end_time}, first_token_time={result.first_token_time}, "
        f"decode_times={result.decode_times})"
    )
    assert repr(result) == expected_repr


@pytest.mark.end_to_end
def test_benchmark_result_full_workflow():
    result = BenchmarkResult("test_id")

    # Start the benchmark
    result.start("Test prompt.")
    assert result.prompt == "Test prompt."
    assert result.prompt_word_count == 2
    assert result.prompt_token_count == 12  # Placeholder for token count
    assert result.start_time is not None

    # Add tokens
    sleep(0.1)  # Simulate delay
    result.add_token("Token1")
    assert result.output == "Token1"
    assert result.output_word_count == 1
    assert result.output_token_count == 6  # Placeholder for token count
    assert result.first_token_time is not None
    assert len(result.decode_times.data_) == 1

    sleep(0.1)  # Simulate delay
    result.add_token("Token2")
    assert result.output == "Token1Token2"
    assert result.output_word_count == 1  # Placeholder for word count update
    assert result.output_token_count == 12  # Placeholder for token count
    assert len(result.decode_times.data_) == 2

    # End the benchmark with supplied token counts
    sleep(0.1)  # Simulate delay
    result.end("Final output.", prompt_token_count=5, output_token_count=7)
    assert result.output == "Final output."
    assert result.output_word_count == 2
    assert result.output_token_count == 7
    assert result.prompt_token_count == 5
    assert result.end_time is not None
