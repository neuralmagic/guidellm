"""
Guidellm is a package that provides an easy and intuitive interface for
evaluating and benchmarking large language models (LLMs).
"""

import contextlib
import logging
import os

with (
    open(os.devnull, "w") as devnull,  # noqa: PTH123
    contextlib.redirect_stderr(devnull),
    contextlib.redirect_stdout(devnull),
):
    from transformers.utils import logging as hf_logging  # type: ignore[import]

    # Set the log level for the transformers library to ERROR
    # to ignore None of PyTorch, TensorFlow found
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence warnings for tokenizers
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)

from .config import (
    DatasetSettings,
    Environment,
    LoggingSettings,
    OpenAISettings,
    Settings,
    print_config,
    reload_settings,
    settings,
)
from .logger import configure_logger, logger

__all__ = [
    "DatasetSettings",
    "Environment",
    "LoggingSettings",
    "OpenAISettings",
    "Settings",
    "configure_logger",
    "logger",
    "print_config",
    "reload_settings",
    "settings",
]
