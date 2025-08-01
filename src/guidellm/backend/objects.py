"""
Backend object models for request and response handling in the GuideLLM toolkit.

This module provides standardized models for generation requests, responses,
and timing information to ensure consistent data handling across different
backend implementations.

Classes:
    GenerationRequest: Request model for generation operations with content,
        parameters, statistics, and constraints.
    GenerationResponse: Response model containing generation results, token
        counts, timing information, and error details.
    GenerationRequestTimings: Timing model for tracking request lifecycle
        events and performance metrics.
"""

import uuid
from typing import Any, Literal, Optional

from pydantic import Field

from guidellm.objects.pydantic import StandardBaseModel
from guidellm.scheduler import RequestTimings

__all__ = [
    "GenerationRequest",
    "GenerationRequestTimings",
    "GenerationResponse",
]


class GenerationRequest(StandardBaseModel):
    """
    Request model for backend generation operations.

    Encapsulates all necessary information for performing text or chat completion
    requests through backend systems.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the request.",
    )
    request_type: Literal["text_completions", "chat_completions"] = Field(
        default="text_completions",
        description=(
            "The type of request (e.g., text, chat). "
            "If request_type='text_completions', resolved by backend.text_completions. "
            "If request_typ='chat_completions', resolved by backend.chat_completions."
        ),
    )
    content: Any = Field(
        description=(
            "The content for the request to send to the backend. "
            "For request_type='text_completions', this should be a string or list "
            "of strings which will be resolved by backend.text_completions(). "
            "For request_type='chat_completions', this should be a string, "
            "a list of (str, Dict[str, Union[str, Dict[str, str]]], Path, Image), "
            "or raw content which will be resolved by backend.chat_completions(). "
            "For raw content, set raw_content=True in the params field."
        )
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters passed as kwargs to the backend methods. "
            "For HTTP backends, these are included in the request body. "
            "Common parameters include max_tokens, temperature, and stream."
        ),
    )
    stats: dict[Literal["prompt_tokens"], int] = Field(
        default_factory=dict,
        description=(
            "Request statistics including prompt token count. "
            "Used for tracking resource usage and performance analysis."
        ),
    )
    constraints: dict[Literal["output_tokens"], int] = Field(
        default_factory=dict,
        description=(
            "Request constraints such as maximum output tokens. "
            "Used to control backend generation behavior and resource limits."
        ),
    )


class GenerationResponse(StandardBaseModel):
    """
    Response model for backend generation operations.

    Contains the results of a generation request including the generated content,
    token usage statistics, iteration counts, and any errors encountered during
    processing. Supports both complete responses and streaming delta updates.
    """

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    request_args: dict[str, Any] = Field(
        description="Arguments that were passed to the backend for this request."
    )
    value: Optional[str] = Field(
        default=None,
        description="Complete generated text content. None for streaming responses.",
    )
    delta: Optional[str] = Field(
        default=None, description="Incremental text content for streaming responses."
    )
    iterations: int = Field(
        default=0, description="Number of generation iterations completed."
    )
    request_prompt_tokens: Optional[int] = Field(
        default=None, description="Token count from the original request prompt."
    )
    request_output_tokens: Optional[int] = Field(
        default=None,
        description="Expected output token count from the original request.",
    )
    response_prompt_tokens: Optional[int] = Field(
        default=None, description="Actual prompt token count reported by the backend."
    )
    response_output_tokens: Optional[int] = Field(
        default=None, description="Actual output token count reported by the backend."
    )


class GenerationRequestTimings(RequestTimings):
    """
    Timing model for tracking generation request lifecycle events.

    Extends the base RequestTimings with generation-specific timing points
    including first and last iteration timestamps.
    """

    first_iteration: Optional[float] = Field(
        default=None,
        description="Unix timestamp when the first generation iteration began.",
    )
    last_iteration: Optional[float] = Field(
        default=None,
        description="Unix timestamp when the last generation iteration completed.",
    )
