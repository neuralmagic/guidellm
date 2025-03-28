"""
Guidellm is a package that provides an easy and intuitive interface for
evaluating and benchmarking large language models (LLMs).
"""

# flake8: noqa

import os
import transformers  # type: ignore

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence warnings for tokenizers
transformers.logging.set_verbosity_error()  # Silence warnings for transformers


from .config import settings
from .logger import configure_logger, logger
# from .main import generate_benchmark_report

__all__ = ["configure_logger", "logger", "settings", "generate_benchmark_report"]
