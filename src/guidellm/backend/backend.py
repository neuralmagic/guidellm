"""
Backend interface and registry for generative AI model interactions.

This module provides the abstract base class and interface for implementing
backends that communicate with generative AI models. Backends handle the
lifecycle of generation requests, including startup, validation, request
processing, and shutdown phases.

Classes:
    Backend: Abstract base class for generative AI backends with registry support.

Type Aliases:
    BackendType: Literal type defining supported backend implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, Optional

from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import BackendInterface, ScheduledRequestInfo
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "Backend",
    "BackendType",
]


BackendType = Literal["openai_http"]


class Backend(
    ABC,
    RegistryMixin["type[Backend]"],
    BackendInterface[GenerationRequest, GenerationRequestTimings, GenerationResponse],
):
    """
    Abstract base class for generative AI backends with registry and lifecycle.

    This class defines the interface for implementing backends that communicate with
    generative AI models. It combines the registry pattern for automatic discovery
    with a well-defined lifecycle for process-based distributed execution.

    The backend lifecycle consists of four main phases:
    1. Creation and initial configuration (constructor and factory methods)
    2. Process startup - Initialize resources within a worker process
    3. Validation - Verify backend readiness and configuration
    4. Request resolution - Process generation requests iteratively
    5. Process shutdown - Clean up resources when process terminates

    All backend implementations must ensure that their state (excluding resources
    created during process_startup) is pickleable to support transfer across
    process boundaries in distributed execution environments.

    Example:
    ::
        # Register a custom backend implementation
        @Backend.register("my_backend")
        class MyBackend(Backend):
            def __init__(self, api_key: str):
                super().__init__("my_backend")
                self.api_key = api_key

            async def process_startup(self):
                # Initialize process-specific resources
                self.client = MyAPIClient(self.api_key)

            ...

        # Create backend instance using factory method
        backend = Backend.create("my_backend", api_key="secret")
    """

    @classmethod
    def create(cls, type_: BackendType, **kwargs) -> "Backend":
        """
        Factory method to create a backend instance based on the backend type.

        :param type_: The type of backend to create.
        :param kwargs: Additional arguments for backend initialization.
        :return: An instance of a subclass of Backend.
        :raises ValueError: If the backend type is not registered.
        """

        backend = cls.get_registered_object(type_)

        return backend(**kwargs)

    def __init__(self, type_: BackendType):
        """
        Initialize a backend instance with the specified type.

        :param type_: The backend type identifier for this instance.
        """
        self.type_ = type_

    @property
    def processes_limit(self) -> Optional[int]:
        """
        :return: The maximum number of worker processes supported by the
            backend. None if not limited.
        """
        return None

    @property
    def requests_limit(self) -> Optional[int]:
        """
        :return: The maximum number of concurrent requests that can be processed
            at once globally by the backend. None if not limited.
        """
        return None

    @abstractmethod
    async def process_startup(self):
        """
        Initialize process-specific resources and connections.

        This method is called when a backend instance is transferred to a worker
        process and needs to establish connections, initialize clients, or set up
        any other resources required for request processing. All resources created
        here are process-local and do not need to be pickleable.
        If there are any errors during startup, this method should raise an
        appropriate exception.

        Must be called before validate() or resolve() can be used.
        """
        ...

    @abstractmethod
    async def validate(self):
        """
        Validate backend configuration and readiness for request processing.

        This method verifies that the backend is properly configured and can
        successfully communicate with the target model service. It should be
        called after process_startup() and before resolve() to ensure the
        backend is ready to handle generation requests.
        If the backend cannot connect to the service or is not ready,
        this method should raise an appropriate exception.
        """

    @abstractmethod
    async def process_shutdown(self):
        """
        Clean up process-specific resources and connections.

        This method is called when the worker process is shutting down and
        should clean up any resources created during process_startup(). After
        this method is called, validate() and resolve() should not be used.
        """
        ...

    @abstractmethod
    async def resolve(
        self,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        history: Optional[list[tuple[GenerationRequest, GenerationResponse]]] = None,
    ) -> AsyncIterator[
        tuple[GenerationResponse, ScheduledRequestInfo[GenerationRequestTimings]]
    ]:
        """
        Process a generation request and yield progressive responses.

        This method processes a generation request through the backend's model
        service, yielding intermediate responses as the generation progresses.
        The final yielded item contains the complete response and timing data.

        The request_info parameter is updated with timing metadata and other
        tracking information throughout the request processing lifecycle.

        :param request: The generation request containing content and parameters.
        :param request_info: Request tracking information to be updated with
            timing and progress metadata during processing.
        :param history: Optional conversation history for multi-turn requests.
            Each tuple contains a previous request-response pair that provides
            context for the current generation.
        :yields: Tuples of (response, updated_request_info) as the generation
            progresses. The final tuple contains the complete response.
        """
        ...

    @abstractmethod
    async def info(self) -> dict[str, Any]:
        """
        :return: Dictionary containing backend metadata such as model
            information, service endpoints, version details, and other
            configuration data useful for reporting and diagnostics.
        """
        ...

    @abstractmethod
    async def default_model(self) -> str:
        """
        :return: The model name or identifier that this backend is
            configured to use by default for generation requests.
        """
        ...
