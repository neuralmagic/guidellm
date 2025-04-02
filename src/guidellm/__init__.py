"""
Guidellm is a package that provides an easy and intuitive interface for
evaluating and benchmarking large language models (LLMs).
"""

# flake8: noqa

import os
import logging
import contextlib


with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(
    devnull
), contextlib.redirect_stdout(devnull):
    from transformers.utils import logging as hf_logging

    # Set the log level for the transformers library to ERROR
    # to ignore None of PyTorch, TensorFlow found
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence warnings for tokenizers
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)

from .config import settings
from .logger import configure_logger, logger

__all__ = ["configure_logger", "logger", "settings", "generate_benchmark_report"]
