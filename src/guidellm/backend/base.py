from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Type

from loguru import logger

from guidellm.core import TextGenerationRequest, TextGenerationResult

__all__ = ["Backend", "BackendEngine", "GenerativeResponse"]


class BackendEngine(str, Enum):
    """
    Determines the Engine of the LLM Backend.
    All the implemented backends in the project have the engine.

    NOTE: the `TEST` engine has to be used only for testing purposes.
    """

    TEST = "test"
    OPENAI_SERVER = "openai_server"


@dataclass
class GenerativeResponse:
    """
    A dataclass to represent a response from a generative AI backend.
    """

    type_: str  # One of 'token_iter', 'final'
    add_token: Optional[str] = None
    prompt: Optional[str] = None
    output: Optional[str] = None
    prompt_token_count: Optional[int] = None
    output_token_count: Optional[int] = None


class Backend(ABC):
    """
    An abstract base class with template methods for generative AI backends.
    """

    _registry: Dict[BackendEngine, "Type[Backend]"] = {}

    @classmethod
    def register(cls, backend_type: BackendEngine):
        """
        A decorator to register a backend class in the backend registry.

        :param backend_type: The type of backend to register.
        :type backend_type: BackendType
        """

        def inner_wrapper(wrapped_class: Type["Backend"]):
            cls._registry[backend_type] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, backend_type: BackendEngine, **kwargs) -> "Backend":
        """
        Factory method to create a backend based on the backend type.

        :param backend_type: The type of backend to create.
        :type backend_type: BackendType
        :param kwargs: Additional arguments for backend initialization.
        :type kwargs: dict
        :return: An instance of a subclass of Backend.
        :rtype: Backend
        """

        logger.info(f"Creating backend of type {backend_type}")

        if backend_type not in cls._registry:
            logger.error(f"Unsupported backend type: {backend_type}")
            raise ValueError(f"Unsupported backend type: {backend_type}")

        return Backend._registry[backend_type](**kwargs)

    def submit(self, request: TextGenerationRequest) -> TextGenerationResult:
        """
        Submit a result request and populate the BenchmarkResult.

        :param request: The result request to submit.
        :type request: TextGenerationRequest
        :return: The populated result result.
        :rtype: TextGenerationResult
        """

        logger.info(f"Submitting request with prompt: {request.prompt}")

        # TODO: Doublecheck why do we need the result id
        # result_id = str(uuid.uuid4())

        result = TextGenerationResult(TextGenerationRequest(prompt=request.prompt))
        result.start(request.prompt)

        for response in self.make_request(request):  # GenerativeResponse
            if response.type_ == "token_iter" and response.add_token:
                result.output_token(response.add_token)
            elif response.type_ == "final":
                result.end(
                    response.prompt_token_count,
                    response.output_token_count,
                )

        logger.info(f"Request completed with output: {result.output}")

        return result

    @abstractmethod
    def make_request(
        self, request: TextGenerationRequest
    ) -> Iterator[GenerativeResponse]:
        """
        Abstract method to make a request to the backend.

        :param request: The result request to submit.
        :type request: TextGenerationRequest
        :return: An iterator over the generative responses.
        :rtype: Iterator[GenerativeResponse]
        """

        pass

    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Abstract method to get the available models for the backend.

        :return: A list of available models.
        :rtype: List[str]
        """

        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """
        Abstract method to get the default model for the backend.

        :return: The default model.
        :rtype: str
        """

        pass

    @abstractmethod
    def model_tokenizer(self, model: str) -> Optional[str]:
        """
        Abstract method to get the tokenizer for a model.

        :param model: The model to get the tokenizer for.
        :type model: str
        :return: The tokenizer for the model, or None if it cannot be created.
        :rtype: Optional[str]
        """

        pass
