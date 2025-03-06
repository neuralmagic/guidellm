import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field, computed_field

from guidellm.config import settings

__all__ = [
    "Backend",
    "BackendType",
    "StreamingResponseType",
    "StreamingRequestArgs",
    "StreamingResponseTimings",
    "StreamingTextResponseStats",
    "StreamingResponse",
]


BackendType = Literal["openai_http"]

StreamingResponseType = Literal["start", "iter", "final"]


class StreamingRequestArgs(BaseModel):
    """
    A model representing the arguments for a streaming request to a backend.
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


class StreamingResponseTimings(BaseModel):
    """
    A model representing the performance timings for a streaming response
    from a backend. Includes the start time of the request, the end time of
    the request if completed, the delta time for the latest iteration,
    and the list of timing values for each iteration.

    :param request_start: The absolute start time of the request in seconds.
    :param values: The list of absolute timing values for each iteration in seconds,
        if any have occurred so far.
        The first value is the time the first token was received.
        The last value is the time the last token was received.
        All values in between are the times each iteration was received, which
        may or may not correspond to a token depending on the backend's implementation.
    :param request_end: The absolute end time of the request in seconds, if completed.
    :param delta: The time in seconds for the latest iteration, if any.
    """

    request_start: Optional[float] = None
    values: List[float] = Field(default_factory=list)
    request_end: Optional[float] = None
    delta: Optional[float] = None


class StreamingTextResponseStats(BaseModel):
    """
    A model representing the statistics for a streaming text response from a backend.
    request_* values are the numbers passed in to the backend's request implementation,
    including any measured prompt_tokens along with the number of output_tokens that
    were requested. response_* values are the numbers returned from the backend's
    response implementation, if any, including any measured prompt_tokens along with
    the number of output_tokens that were returned.

    :param request_prompt_tokens: The number of prompt tokens requested for the request.
    :param request_output_tokens: The number of output tokens requested for the request.
    :param response_prompt_tokens: The number of prompt tokens returned in the response.
    :param response_output_tokens: The number of output tokens returned in the response.
    :param response_stream_iterations: The number of iterations that have been returned
        from the backend so far, or if at the end, the total number of iterations that
        were returned.
    """

    request_prompt_tokens: Optional[int] = None
    request_output_tokens: Optional[int] = None
    response_prompt_tokens: Optional[int] = None
    response_output_tokens: Optional[int] = None
    response_stream_iterations: int = 0

    @computed_field  # type: ignore[misc]
    @property
    def prompt_tokens_count(self) -> Optional[int]:
        if settings.preferred_prompt_tokens_source == "backend":
            if not self.response_prompt_tokens:
                logger.warning(
                    "preferred_prompt_tokens_source is set to 'backend', "
                    " but no prompt tokens were returned by the backend. "
                    "Falling back to local, if available."
                )
            return self.response_prompt_tokens or self.request_prompt_tokens

        if settings.preferred_prompt_tokens_source == "local":
            if not self.request_prompt_tokens:
                logger.warning(
                    "preferred_prompt_tokens_source is set to 'local', "
                    "but no prompt tokens were provided in the request. "
                    "Falling back to backend, if available."
                )
            return self.request_prompt_tokens or self.response_prompt_tokens

        return self.response_prompt_tokens or self.request_prompt_tokens

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_count(self) -> Optional[int]:
        if settings.preferred_output_tokens_source == "backend":
            if not self.response_output_tokens:
                logger.warning(
                    "preferred_output_tokens_source is set to 'backend', "
                    "but no output tokens were returned by the backend. "
                    "Falling back to local, if available."
                )
            return self.response_output_tokens or self.request_output_tokens

        if settings.preferred_output_tokens_source == "local":
            if not self.request_output_tokens:
                logger.warning(
                    "preferred_output_tokens_source is set to 'local', "
                    "but no output tokens were provided in the request. "
                    "Falling back to backend, if available."
                )
            return self.request_output_tokens or self.response_output_tokens

        return self.response_output_tokens or self.request_output_tokens


class StreamingResponse(BaseModel):
    """
    A model representing a response from a streaming request to a backend.
    Includes the type of response, the request arguments, the performance timings,
    the statistics, the delta time for the latest iteration,
    and the content of the response.

    :param type_: The type of response, either 'start' for the initial response,
        'iter' for intermediate streaming output, or 'final' for the final result.
        The response cycle from a backend will always start with a 'start' response,
        followed by zero or more 'iter' responses, and ending with a 'final' response.
    :param id_: The unique identifier for the request, if any.
        Used for tracking purposes.
    :param request_args: The arguments for the request that generated this response.
    :param timings: The performance timings for the response.
    :param stats: The statistics for the response.
    :param delta: The delta content for the latest iteration, if any.
    :param content: The returned content for the response, continuously appended to for
        each iteration.
    """

    type_: StreamingResponseType = "start"
    id_: Optional[str] = None
    request_args: StreamingRequestArgs
    timings: StreamingResponseTimings = Field(default_factory=StreamingResponseTimings)
    stats: StreamingTextResponseStats = Field(
        default_factory=StreamingTextResponseStats
    )
    delta: Any = None
    content: Any = ""


class Backend(ABC):
    """
    Abstract base class for generative AI backends.

    This class provides a common interface for creating and interacting with different
    generative AI backends. Subclasses should implement the abstract methods to
    define specific backend behavior.

    :cvar _registry: A registration dictionary that maps BackendType to backend classes.
    :param type_: The type of the backend.
    """

    _registry: Dict[BackendType, "Type[Backend]"] = {}

    @classmethod
    def register(cls, backend_type: BackendType):
        """
        A decorator to register a backend class in the backend registry.

        :param backend_type: The type of backend to register.
        :type backend_type: BackendType
        :return: The decorated backend class.
        :rtype: Type[Backend]
        """

        def inner_wrapper(wrapped_class: Type["Backend"]):
            cls._registry[backend_type] = wrapped_class
            logger.info("Registered backend type: {}", backend_type)
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, type_: BackendType, **kwargs) -> "Backend":
        """
        Factory method to create a backend instance based on the backend type.

        :param type_: The type of backend to create.
        :type type_: BackendType
        :param kwargs: Additional arguments for backend initialization.
        :return: An instance of a subclass of Backend.
        :rtype: Backend
        :raises ValueError: If the backend type is not registered.
        """

        logger.info("Creating backend of type {}", type_)

        if type_ not in cls._registry:
            err = ValueError(f"Unsupported backend type: {type_}")
            logger.error("{}", err)
            raise err

        return Backend._registry[type_](**kwargs)

    def __init__(self, type_: BackendType):
        self._type = type_

    @property
    def type_(self) -> BackendType:
        """
        :return: The type of the backend.
        """
        return self._type

    @property
    @abstractmethod
    def target(self) -> str:
        """
        :return: The target location for the backend.
        """
        ...

    @property
    @abstractmethod
    def model(self) -> Optional[str]:
        """
        :return: The model used for the backend requests.
        """
        ...

    def validate(self):
        """
        Handle final setup and validate the backend is ready for use.
        If not successful, raises the appropriate exception.
        """
        logger.info("{} validating backend {}", self.__class__.__name__, self.type_)
        self.check_setup()
        models = self.available_models()
        if not models:
            raise ValueError("No models available for the backend")

        async def _test_request():
            async for _ in self.text_completions(
                prompt="Test connection", output_token_count=1
            ):  # type: ignore[attr-defined]
                pass

        asyncio.run(_test_request())

    @abstractmethod
    def check_setup(self):
        """
        Check the setup for the backend.
        If unsuccessful, raises the appropriate exception.

        :raises ValueError: If the setup check fails.
        """
        ...

    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Get the list of available models for the backend.

        :return: The list of available models.
        :rtype: List[str]
        """
        ...

    @abstractmethod
    async def text_completions(
        self,
        prompt: Union[str, List[str]],
        id_: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Generate text only completions for the given prompt.
        Does not support multiple modalities, complicated chat interfaces,
        or chat templates. Specifically, it requests with only the prompt.

        :param prompt: The prompt (or list of prompts) to generate a completion for.
            If a list is supplied, these are concatenated and run through the model
            for a single prompt.
        :param id_: The unique identifier for the request, if any.
            Added to logging statements and the response for tracking purposes.
        :param prompt_token_count: The number of tokens measured in the prompt, if any.
            Returned in the response stats for later analysis, if applicable.
        :param output_token_count: If supplied, the number of tokens to enforce
            generation of for the output for this request.
        :param kwargs: Additional keyword arguments to pass with the request.
        :return: An async generator that yields StreamingResponse objects containing the
            response content. Will always start with a 'start' response,
            followed by 0 or more 'iter' responses, and ending with a 'final' response.
        """
        ...

    @abstractmethod
    async def chat_completions(
        self,
        content: Union[
            str,
            List[Union[str, Dict[str, Union[str, Dict[str, str]]], Path, Image.Image]],
            Any,
        ],
        id_: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        raw_content: bool = False,
        **kwargs,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Generate chat completions for the given content.
        Supports multiple modalities, complicated chat interfaces, and chat templates.
        Specifically, it requests with the content, which can be any combination of
        text, images, and audio provided the target model supports it,
        and returns the output text. Additionally, any chat templates
        for the model are applied within the backend.

        :param content: The content (or list of content) to generate a completion for.
            This supports any combination of text, images, and audio (model dependent).
            Supported text only request examples:
                content="Sample prompt", content=["Sample prompt", "Second prompt"],
                content=[{"type": "text", "value": "Sample prompt"}.
            Supported text and image request examples:
                content=["Describe the image", PIL.Image.open("image.jpg")],
                content=["Describe the image", Path("image.jpg")],
                content=["Describe the image", {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}].
            Supported text and audio request examples:
                content=["Transcribe the audio", Path("audio.wav")],
                content=["Transcribe the audio", {"type": "input_audio",
                "input_audio": {"data": f"{base64_bytes}", "format": "wav}].
            Additionally, if raw_content=True then the content is passed directly to the
            backend without any processing.
        :param id_: The unique identifier for the request, if any.
            Added to logging statements and the response for tracking purposes.
        :param prompt_token_count: The number of tokens measured in the prompt, if any.
            Returned in the response stats for later analysis, if applicable.
        :param output_token_count: If supplied, the number of tokens to enforce
            generation of for the output for this request.
        :param kwargs: Additional keyword arguments to pass with the request.
        :return: An async generator that yields StreamingResponse objects containing the
            response content. Will always start with a 'start' response,
            followed by 0 or more 'iter' responses, and ending with a 'final' response.
        """
        ...
