"""
Guidellm is a package that provides an easy and intuitive interface for
evaluating and benchmarking large language models (LLMs).
"""

import os

from .config import settings
from .logger import configure_logger, logger

__all__ = ["configure_logger", "logger", "settings"]


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence warnings for tokenizers
