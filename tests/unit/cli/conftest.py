from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def patch_main(mocker) -> MagicMock:
    return mocker.patch("guidellm.main.main.callback")


@pytest.fixture
def default_main_kwargs() -> Dict[str, Any]:
    """
    All the defaults come from the `guidellm.main` function.
    """

    return {
        "target": "http://localhost:8000",
        "host": None,
        "port": None,
        "backend": "openai_server",
        "model": None,
        "task": None,
        "data": None,
        "data_type": "transformers",
        "tokenizer": None,
        "rate_type": "synchronous",
        "rate": (),
        "max_seconds": 120,
        "max_requests": None,
        "output_path": "benchmark_report.json",
    }
