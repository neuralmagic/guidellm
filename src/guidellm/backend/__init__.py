"""
Backend infrastructure for GuideLLM language model interactions.

Provides abstract base classes, implemented backends, request/response objects,
and timing utilities for standardized communication with LLM providers.
"""

# Import backend implementations to trigger registration
from . import openai  # noqa: F401
from .backend import (
    Backend,
    BackendType,
)
from .objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)

__all__ = [
    "Backend",
    "BackendType",
    "GenerationRequest",
    "GenerationRequestTimings",
    "GenerationResponse",
]
