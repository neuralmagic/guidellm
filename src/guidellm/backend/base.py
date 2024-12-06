import asyncio
import functools
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from loguru import logger
from pydantic import BaseModel
from transformers import (  # type: ignore  # noqa: PGH003
    AutoTokenizer,
    PreTrainedTokenizer,
)

from guidellm.core import TextGenerationRequest, TextGenerationResult

__all__ = ["Backend", "BackendEngine", "BackendEnginePublic", "GenerativeResponse"]


BackendEnginePublic = Literal["openai_server", "aiohttp_server"]
BackendEngine = Union[BackendEnginePublic, Literal["test"]]


class GenerativeResponse(BaseModel):
    """
    A model representing a response from a generative AI backend.

    :param type_: The type of response, either 'token_iter' for intermediate
        token output or 'final' for the final result.
    :type type_: Literal["token_iter", "final"]
    :param add_token: The token to add to the output
        (only applicable if type_ is 'token_iter').
    :type add_token: Optional[str]
    :param prompt: The original prompt sent to the backend.
    :type prompt: Optional[str]
    :param output: The final generated output (only applicable if type_ is 'final').
    :type output: Optional[str]
    :param prompt_token_count: The number of tokens in the prompt.
    :type prompt_token_count: Optional[int]
    :param output_token_count: The number of tokens in the output.
    :type output_token_count: Optional[int]
    """

    type_: Literal["token_iter", "final"]
    add_token: Optional[str] = None
    prompt: Optional[str] = None
    output: Optional[str] = None
    prompt_token_count: Optional[int] = None
    output_token_count: Optional[int] = None


class Backend(ABC):
    """
    Abstract base class for generative AI backends.

    This class provides a common interface for creating and interacting with different
    generative AI backends. Subclasses should implement the abstract methods to
    define specific backend behavior.

    :cvar _registry: A dictionary that maps BackendEngine types to backend classes.
    :type _registry: Dict[BackendEngine, Type[Backend]]
    :param type_: The type of the backend.
    :type type_: BackendEngine
    :param target: The target URL for the backend.
    :type target: str
    :param model: The model used by the backend.
    :type model: str
    """

    _registry: Dict[BackendEngine, "Type[Backend]"] = {}

    @classmethod
    def register(cls, backend_type: BackendEngine):
        """
        A decorator to register a backend class in the backend registry.

        :param backend_type: The type of backend to register.
        :type backend_type: BackendEngine
        :return: The decorated backend class.
        :rtype: Type[Backend]
        """

        def inner_wrapper(wrapped_class: Type["Backend"]):
            cls._registry[backend_type] = wrapped_class
            logger.info("Registered backend type: {}", backend_type)
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, backend_type: BackendEngine, **kwargs) -> "Backend":
        """
        Factory method to create a backend instance based on the backend type.

        :param backend_type: The type of backend to create.
        :type backend_type: BackendEngine
        :param kwargs: Additional arguments for backend initialization.
        :return: An instance of a subclass of Backend.
        :rtype: Backend
        :raises ValueError: If the backend type is not registered.
        """

        logger.info("Creating backend of type {}", backend_type)

        if backend_type not in cls._registry:
            err = ValueError(f"Unsupported backend type: {backend_type}")
            logger.error("{}", err)
            raise err

        return Backend._registry[backend_type](**kwargs)

    def __init__(self, type_: BackendEngine, target: str, model: str):
        """
        Base constructor for the Backend class.
        Calls into test_connection to ensure the backend is reachable.
        Ensure all setup is done in the subclass constructor before calling super.

        :param type_: The type of the backend.
        :param target: The target URL for the backend.
        :param model: The model used by the backend.
        """
        self._type = type_
        self._target = target
        self._model = model

        self.test_connection()

    @property
    def default_model(self) -> str:
        """
        Get the default model for the backend.

        :return: The default model.
        :rtype: str
        :raises ValueError: If no models are available.
        """
        return _cachable_default_model(self)

    @property
    def type_(self) -> BackendEngine:
        """
        Get the type of the backend.

        :return: The type of the backend.
        :rtype: BackendEngine
        """
        return self._type

    @property
    def target(self) -> str:
        """
        Get the target URL for the backend.

        :return: The target URL.
        :rtype: str
        """
        return self._target

    @property
    def model(self) -> str:
        """
        Get the model used by the backend.

        :return: The model name.
        :rtype: str
        """
        return self._model

    def model_tokenizer(self) -> PreTrainedTokenizer:
        """
        Get the tokenizer for the backend model.

        :return: The tokenizer instance.
        """
        return AutoTokenizer.from_pretrained(self.model)

    def test_connection(self) -> bool:
        """
        Test the connection to the backend by running a short text generation request.
        If successful, returns True, otherwise raises an exception.

        :return: True if the connection is successful.
        :rtype: bool
        :raises ValueError: If the connection test fails.
        """
        try:
            asyncio.get_running_loop()
            is_async = True
        except RuntimeError:
            is_async = False

        if is_async:
            logger.warning("Running in async mode, cannot test connection")
            return True

        try:
            request = TextGenerationRequest(
                prompt="Test connection", output_token_count=5
            )

            asyncio.run(self.submit(request))
            return True
        except Exception as err:
            raise_err = RuntimeError(
                f"Backend connection test failed for backend type={self.type_} "
                f"with target={self.target} and model={self.model} with error: {err}"
            )
            logger.error(raise_err)
            raise raise_err from err

    async def submit(self, request: TextGenerationRequest) -> TextGenerationResult:
        """
        Submit a text generation request and return the result.

        This method handles the request submission to the backend and processes
        the response in a streaming fashion if applicable.

        :param request: The request object containing the prompt
            and other configurations.
        :type request: TextGenerationRequest
        :return: The result of the text generation request.
        :rtype: TextGenerationResult
        :raises ValueError: If no response is received from the backend.
        """

        logger.debug("Submitting request with prompt: {}", request.prompt)

        result = TextGenerationResult(request=request)
        result.start(request.prompt)
        received_final = False

        async for response in self.make_request(request):
            logger.debug("Received response: {}", response)
            if response.type_ == "token_iter":
                result.output_token(response.add_token if response.add_token else "")
            elif response.type_ == "final":
                if received_final:
                    err = ValueError(
                        "Received multiple final responses from the backend."
                    )
                    logger.error(err)
                    raise err

                result.end(
                    output=response.output,
                    prompt_token_count=response.prompt_token_count,
                    output_token_count=response.output_token_count,
                )
                received_final = True
            else:
                err = ValueError(
                    f"Invalid response received from the backend of type: "
                    f"{response.type_} for {response}"
                )
                logger.error(err)
                raise err

        if not received_final:
            err = ValueError("No final response received from the backend.")
            logger.error(err)
            raise err

        logger.info("Request completed with output: {}", result.output)

        return result

    @abstractmethod
    async def make_request(
        self,
        request: TextGenerationRequest,
    ) -> AsyncGenerator[GenerativeResponse, None]:
        """
        Abstract method to make a request to the backend.

        Subclasses must implement this method to define how requests are handled
        by the backend.

        :param request: The request object containing the prompt and
            other configurations.
        :type request: TextGenerationRequest
        :yield: A generator yielding responses from the backend.
        :rtype: AsyncGenerator[GenerativeResponse, None]
        """
        yield None  # type: ignore  # noqa: PGH003

    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Abstract method to get the available models for the backend.

        Subclasses must implement this method to provide the list of models
        supported by the backend.

        :return: A list of available models.
        :rtype: List[str]
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


@functools.lru_cache(maxsize=1)
def _cachable_default_model(backend: Backend) -> str:
    """
    Get the default model for a backend using LRU caching.

    This function caches the default model to optimize repeated lookups.

    :param backend: The backend instance for which to get the default model.
    :type backend: Backend
    :return: The default model.
    :rtype: str
    :raises ValueError: If no models are available.
    """
    logger.debug("Getting default model for backend: {}", backend)
    models = backend.available_models()
    if models:
        logger.debug("Default model: {}", models[0])
        return models[0]

    err = ValueError("No models available.")
    logger.error(err)
    raise err
