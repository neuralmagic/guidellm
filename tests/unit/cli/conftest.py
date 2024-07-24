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
        "target": "localhost:8000/completions",
        "host": None,
        "port": None,
        "path": None,
        "backend": "openai_server",
        "model": None,
        "task": None,
        "data": None,
        "data_type": "transformers",
        "tokenizer": None,
        "rate_type": "synchronous",
        "rate": (1.0,),
        "num_seconds": 120,
        "num_requests": None,
    }
