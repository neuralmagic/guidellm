from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
)

from pydantic import Field

from guidellm.objects import StandardBaseModel
from guidellm.scheduler import (
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
)


class BackendInterface(ABC, Generic[RequestT, RequestTimingsT, ResponseT]):
    """
    Abstract interface for request processing backends. Note: before process_startup
    is invoked, the implementation must ensure all properties are pickleable.
    """

    @property
    @abstractmethod
    def processes_limit(self) -> Optional[int]:
        """Maximum worker processes supported, or None if unlimited."""

    @property
    @abstractmethod
    def requests_limit(self) -> Optional[int]:
        """Maximum concurrent requests supported, or None if unlimited."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """
        :return: Backend metadata including model any initializaiton and
            configuration information.
        """
        ...

    @abstractmethod
    async def process_startup(self) -> None:
        """
        Perform backend initialization and startup procedures.

        :raises: Implementation-specific exceptions for startup failures.
        """

    @abstractmethod
    async def validate(self) -> None:
        """
        Validate backend configuration and operational status.

        :raises: Implementation-specific exceptions for validation failures.
        """

    @abstractmethod
    async def process_shutdown(self) -> None:
        """
        Perform backend cleanup and shutdown procedures.

        :raises: Implementation-specific exceptions for shutdown failures.
        """

    @abstractmethod
    async def resolve(
        self,
        request: RequestT,
        request_info: ScheduledRequestInfo[RequestTimingsT],
        history: Optional[list[tuple[RequestT, ResponseT]]] = None,
    ) -> AsyncIterator[tuple[ResponseT, ScheduledRequestInfo[RequestTimingsT]]]:
        """
        Process a request and yield incremental response updates.

        :param request: The request object to process.
        :param request_info: Scheduling metadata and timing information.
        :param history: Optional conversation history for multi-turn requests.
        :yield: Tuples of (response, updated_request_info) for each response chunk.
        :raises: Implementation-specific exceptions for processing failures.
        """


BackendT = TypeVar("BackendT", bound="BackendInterface")
