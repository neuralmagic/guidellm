from typing import List
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from guidellm.main import main


def test_main_defaults_from_cli(
    patch_main: MagicMock, cli_runner: CliRunner, default_main_kwargs
):
    cli_runner.invoke(main)

    assert patch_main.call_count == 1
    assert patch_main.call_args.kwargs == default_main_kwargs


def test_main_cli_overrided(
    patch_main: MagicMock, cli_runner: CliRunner, default_main_kwargs
):
    cli_runner.invoke(
        main,
        ["--target", "localhost:9000", "--backend", "test", "--rate-type", "sweep"],
    )
    default_main_kwargs.update(
        {"target": "localhost:9000", "backend": "test", "rate_type": "sweep"}
    )

    assert patch_main.call_count == 1
    assert patch_main.call_args.kwargs == default_main_kwargs


@pytest.mark.parametrize(
    ("args", "expected_stdout"),
    [
        (
            ["--backend", "invalid", "--rate-type", "sweep"],
            (
                b"Usage: main [OPTIONS]\nTry 'main --help' for help.\n\n"
                b"Error: Invalid value for '--backend': "
                b"'invalid' is not one of 'test', 'openai_server'.\n"
            ),
        ),
        (
            ["--max-requests", "str instead of int"],
            (
                b"Usage: main [OPTIONS]\nTry 'main --help' for help.\n\n"
                b"Error: Invalid value for '--max-requests': "
                b"'str instead of int' is not a valid integer.\n"
            ),
        ),
    ],
)
def test_main_cli_validation_error(
    patch_main: MagicMock,
    cli_runner: CliRunner,
    args: List[str],
    expected_stdout: bytes,
):
    result = cli_runner.invoke(main, args)

    assert patch_main.call_count == 0
    assert result.stdout_bytes == expected_stdout
