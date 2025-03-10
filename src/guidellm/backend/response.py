from typing import Any, Dict, Literal, Optional

from loguru import logger
from pydantic import BaseModel, computed_field

from guidellm.config import settings

__all__ = [
    "StreamingResponseType",
    "StreamingTextResponse",
    "RequestArgs",
    "ResponseSummary",
]


StreamingResponseType = Literal["start", "iter"]


class StreamingTextResponse(BaseModel):
    """
    A model representing the response content for a streaming text request.

    :param type_: The type of the response; either 'start' or 'iter'.
    :param value: The value of the response up to this iteration.
    :param start_time: The time.time() the request started.
    :param iter_count: The iteration count for the response. For 'start' this is 0
        and for the first 'iter' it is 1.
    :param delta: The text delta added to the response for this stream iteration.
    :param time: If 'start', the time.time() the request started.
        If 'iter', the time.time() the iteration was received.
    :param request_id: The unique identifier for the request, if any.
    """

    type_: StreamingResponseType
    value: str
    start_time: float
    iter_count: int
    delta: str
    time: float
    request_id: Optional[str] = None


class RequestArgs(BaseModel):
    """
    A model representing the arguments for a request to a backend.
    Biases towards an HTTP request, but can be used for other types of backends.

    :param target: The target URL or function for the request.
    :param headers: The headers, if any, included in the request such as authorization.
    :param payload: The payload / arguments for the request including the prompt /
        content and other configurations.
    :param timeout: The timeout for the request in seconds, if any.
    :param http2: Whether HTTP/2 was used for the request, if applicable.
    """

    target: str
    headers: Dict[str, str]
    payload: Dict[str, Any]
    timeout: Optional[float] = None
    http2: Optional[bool] = None


class ResponseSummary(BaseModel):
    """
    A model representing a summary of a backend request.
    Always returned as the final iteration of a streaming request.

    :param value: The final value returned from the request.
    :param request_args: The arguments used to make the request.
    :param start_time: The time the request started.
    :param end_time: The time the request ended.
    :param iterations: The number of iterations in the request.
    :param prompt_tokens: The number of tokens in the prompt, if any usage was returned.
    :param output_tokens: The number of tokens in the output, if any usage was returned.
    :param request_id: The unique identifier for the request, if any.
    :param error: The error message, if any, returned from making the request.
    """

    value: str
    request_args: RequestArgs
    iterations: int = 0
    start_time: float
    end_time: float
    request_prompt_tokens: Optional[int] = None
    request_output_tokens: Optional[int] = None
    response_prompt_tokens: Optional[int] = None
    response_output_tokens: Optional[int] = None
    request_id: Optional[str] = None
    error: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def prompt_tokens(self) -> Optional[int]:
        """
        The number of tokens measured in the prompt based on preferences
        for trusting the input or response.

        :return: The number of tokens in the prompt, if any.
        """
        if settings.preferred_prompt_tokens_source == "backend":
            if self.response_prompt_tokens is None:
                logger.warning(
                    "Preferred prompt tokens source is backend, but no prompt token "
                    f"values were returned with the response for {self}. "
                    "Defulating to request_prompt_tokens (if available)."
                )
            return self.response_prompt_tokens or self.request_prompt_tokens
        elif settings.preferred_prompt_tokens_source == "request":
            if self.request_prompt_tokens is None:
                logger.warning(
                    "Preferred prompt tokens source is request, but no prompt token "
                    f"values were returned with the request for {self}. "
                    "Defulating to response_prompt_tokens (if available)."
                )
            return self.request_prompt_tokens or self.response_prompt_tokens

        return self.response_prompt_tokens or self.request_prompt_tokens

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens(self) -> Optional[int]:
        """
        The number of tokens measured in the output based on preferences
        for trusting the input or response.

        :return: The number of tokens in the output, if any.
        """
        if settings.preferred_output_tokens_source == "backend":
            if self.response_output_tokens is None:
                logger.warning(
                    "Preferred output tokens source is backend, but no output token "
                    f"values were returned with the response for {self}. "
                    "Defulating to request_output_tokens (if available)."
                )
            return self.response_output_tokens or self.request_output_tokens
        elif settings.preferred_output_tokens_source == "request":
            if self.request_output_tokens is None:
                logger.warning(
                    "Preferred output tokens source is request, but no output token "
                    f"values were returned with the request for {self}. "
                    "Defulating to response_output_tokens (if available)."
                )
            return self.request_output_tokens or self.response_output_tokens

        return self.response_output_tokens or self.request_output_tokens
