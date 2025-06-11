import json
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["StandardBaseModel", "StatusBreakdown"]

T = TypeVar("T", bound="StandardBaseModel")


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

    @classmethod
    def get_default(cls: type[T], field: str) -> Any:
        """Get default values for model fields"""
        return cls.model_fields[field].default

    @classmethod
    def from_file(cls: type[T], filename: Path, overrides: Optional[dict] = None) -> T:
        """
        Attempt to create a new instance of the model using
        data loaded from json or yaml file.
        """
        try:
            with filename.open() as f:
                if str(filename).endswith(".json"):
                    data = json.load(f)
                else:  # Assume everything else is yaml
                    data = yaml.safe_load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Failed to parse {filename} as type {cls.__name__}")
            raise ValueError(f"Error when parsing file: {filename}") from e

        data.update(overrides)
        return cls.model_validate(data)


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
