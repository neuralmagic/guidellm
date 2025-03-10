import asyncio
import math
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field

from guidellm.backend import (
    Backend,
    RequestArgs,
    ResponseSummary,
    StreamingTextResponse,
)
from guidellm.scheduler.scheduler import RequestsWorker

__all__ = ["GenerationRequest", "BackendRequestsWorker"]


class GenerationRequest(BaseModel):
    """
    A class representing a request for generation.
    This class is used to encapsulate the details of a generation request,
    including the request ID, type, content, parameters, statistics, and constraints.
    It is designed to be used with the BackendRequestsWorker class to handle
    the generation process.

    :param request_id: The unique identifier for the request.
    :param request_type: The type of request (e.g., text, chat).
    :param content: The content for the request to send to the backend.
        If request_type is 'text', this should be a string or list of strings
        which will be resolved by backend.text_completions.
        If request_type is 'chat', this should be a string,
        a list of (str, Dict[str, Union[str, Dict[str, str]], Path, Image]),
        or Any raw content which will be resolved by backend.chat_completions.
        If raw content, raw_content=True must be passed in the params.
    :param params: Additional parameters for the request passed in as kwargs.
        For an http backend, these are passed into the body of the request.
    :param stats: Statistics for the request, such as the number of prompt tokens.
        Used for tracking and reporting purposes.
    :param constraints: Constraints for the request, such as the maximum number
        of output tokens. Used for controlling the behavior of the backend.
    """

    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the request.",
    )
    request_type: Literal["text", "chat"] = Field(
        default="text",
        description=(
            "The type of request (e.g., text, chat). "
            "If request_type is 'text', resolved by backend.text_completions. "
            "If request_type is 'chat', resolved by backend.chat_completions."
        ),
    )
    content: Any = Field(
        description=(
            "The content for the request to send to the backend. "
            "If request_type is 'text', this should be a string or list of strings "
            "which will be resolved by backend.text_completions. "
            "If request_type is 'chat', this should be a string, "
            "a list of (str, Dict[str, Union[str, Dict[str, str]], Path, Image]), "
            "or Any raw content which will be resolved by backend.chat_completions. "
            "If raw content, raw_content=True must be passed in the params."
        )
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters for the request that will be passed in as kwargs. "
            "For an http backend, these are passed into the body of the request. "
        ),
    )
    stats: Dict[Literal["prompt_tokens"], int] = Field(
        default_factory=dict,
        description=(
            "Statistics for the request, such as the number of prompt tokens. "
            "Used for tracking and reporting purposes."
        ),
    )
    constraints: Dict[Literal["output_tokens"], int] = Field(
        default_factory=dict,
        description=(
            "Constraints for the request, such as the maximum number of output tokens. "
            "Used for controlling the behavior of the backend."
        ),
    )


class BackendRequestsWorker(RequestsWorker):
    """
    A class that handles the execution of requests using a backend.
    This class is responsible for sending requests to the backend,
    handling responses, and managing errors.

    :param backend: The backend to use for handling requests.
        This should be an instance of Backend such as an OpenAIHTTPBackend.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

    async def resolve(
        self,
        request: GenerationRequest,
        start_time: float,
        timeout_time: float,
    ) -> ResponseSummary:
        """
        Resolve a request by sending it to the backend and handling the response.
        This method sends the request to the backend, waits for a response,
        and handles any errors that may occur during the process.

        :param request: The request to resolve.
        :param start_time: The time to start the request.
        :param timeout_time: The time to wait for a response before timing out.
            If timeout_time is math.inf, the request will not timeout.
        :return: A ResponseSummary object containing the response from the backend.
            If an error occurs, the ResponseSummary will contain the error message.
        """
        response = None
        error: Optional[str] = None

        try:
            request_func, request_kwargs = self._create_request_func_kwargs(request)

            async def _runner():
                # wrap function so we can enforce timeout and
                # still return the latest state from the backend
                async for resp in request_func(**request_kwargs):
                    nonlocal response
                    response = resp

            if (wait_time := start_time - time.time()) > 0:
                await asyncio.sleep(wait_time)

            start_time = time.time()
            await asyncio.wait_for(
                _runner(),
                timeout=timeout_time - time.time() if timeout_time < math.inf else None,
            )

            if not response:
                raise ValueError(
                    f"No response received for request: {request} "
                    f"and backend: {self.backend}"
                )
            if not isinstance(response, ResponseSummary):
                raise ValueError(
                    f"Received no ResponseSummary for request: {request} "
                    f"and backend: {self.backend}, received: {response}"
                )
        except asyncio.TimeoutError as texc:
            error = str(texc)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        return self._handle_response(request, response, error, start_time)

    def _create_request_func_kwargs(
        self,
        request: GenerationRequest,
    ) -> Tuple[
        AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None],
        Dict[str, Any],
    ]:
        request_func: AsyncGenerator[
            Union[StreamingTextResponse, ResponseSummary], None
        ]
        request_kwargs: Dict[str, Any]

        if request.request_type == "text":
            request_func = self.backend.text_completions
            request_kwargs = {
                "prompt": request.content,
                "request_id": request.request_id,
                "prompt_token_count": request.stats.get("prompt_tokens", None),
                "output_token_count": request.constraints.get("output_tokens", None),
                **request.params,
            }
        elif request.request_type == "chat":
            request_func = self.backend.chat_completions
            request_kwargs = {
                "content": request.content,
                "request_id": request.request_id,
                "prompt_token_count": request.stats.get("prompt_tokens", None),
                "output_token_count": request.constraints.get("output_tokens", None),
                **request.params,
            }
        else:
            raise ValueError(
                f"Invalid request type: {request.request_type} for {request}"
            )

        return request_func, request_kwargs

    def _handle_response(
        self,
        request: GenerationRequest,
        response: Any,
        error: Optional[str],
        start_time: float,
    ) -> ResponseSummary:
        if response is None or not isinstance(
            response, (ResponseSummary, StreamingTextResponse)
        ):
            # nothing received or invalid response, fill in defaults for error
            if response:
                error = str(
                    ValueError(
                        f"Invalid response: {type(response)} for request: {request}; "
                    )
                ) + (error or "")

            return ResponseSummary(
                value="",
                request_args=RequestArgs(
                    target=self.backend.target,
                    headers={},
                    payload={},
                ),
                start_time=start_time,
                end_time=time.time(),
                request_id=request.request_id,
                error=error or "Unknown error",
            )

        if isinstance(response, StreamingTextResponse):
            return ResponseSummary(
                value=response.value,
                request_args=RequestArgs(
                    target=self.backend.target,
                    headers={},
                    payload={},
                ),
                start_time=response.start_time,
                end_time=time.time(),
                request_prompt_tokens=request.stats.get("prompt_tokens", None),
                request_output_tokens=None,
                response_prompt_tokens=None,
                response_output_tokens=response.iter_count,
                request_id=request.request_id,
                error=error or "Unknown error",
            )

        response.error = error

        return response
