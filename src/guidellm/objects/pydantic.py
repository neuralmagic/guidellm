from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["StandardBaseModel", "StatusBreakdown"]


class StandardBaseModel(BaseModel):
    """
    A base class for Pydantic models throughout GuideLLM enabling standard
    configuration and logging.
    """

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
        validate_assignment=True,
        from_attributes=True,
    )

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        logger.debug(
            "Initialized new instance of {} with data: {}",
            self.__class__.__name__,
            data,
        )


SuccessfulT = TypeVar("SuccessfulT")
ErroredT = TypeVar("ErroredT")
IncompleteT = TypeVar("IncompleteT")
TotalT = TypeVar("TotalT")


class StatusBreakdown(BaseModel, Generic[SuccessfulT, ErroredT, IncompleteT, TotalT]):
    """
    A base class for Pydantic models that are separated by statuses including
    successful, incomplete, and errored. It additionally enables the inclusion
    of total, which is intended as the combination of all statuses.
    Total may or may not be used depending on if it duplicates information.
    """

    successful: SuccessfulT = Field(
        description="The results with a successful status.",
        default=None,  # type: ignore[assignment]
    )
    errored: ErroredT = Field(
        description="The results with an errored status.",
        default=None,  # type: ignore[assignment]
    )
    incomplete: IncompleteT = Field(
        description="The results with an incomplete status.",
        default=None,  # type: ignore[assignment]
    )
    total: TotalT = Field(
        description="The combination of all statuses.",
        default=None,  # type: ignore[assignment]
    )
