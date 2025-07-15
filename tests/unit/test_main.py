import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli


@pytest.mark.smoke
def test_benchmark_run_invalid_header_format():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target-header",
            "invalid-header",
            "--target",
            "http://localhost:8000",
            "--data",
            "prompt_tokens=1,output_tokens=1",
            "--rate-type",
            "constant",
            "--rate",
            "1",
            "--max-requests",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid header format" in result.output


@pytest.mark.smoke
def test_benchmark_run_valid_header_format():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target-header",
            "Authorization: Bearer my-token",
            "--target",
            "http://localhost:8000",
            "--data",
            "prompt_tokens=1,output_tokens=1",
            "--rate-type",
            "constant",
            "--rate",
            "1",
            "--max-requests",
            "1",
        ],
    )
    # This will fail because it can't connect to the server,
    # but it will pass the header parsing, which is what we want to test.
    assert "Invalid header format" not in result.output
