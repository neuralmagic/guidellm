from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

__all__ = ["StandardBaseModel"]


class StandardBaseModel(BaseModel):
    """
    A base class for models that require YAML and JSON serialization and
    deserialization.
    """

    model_config = ConfigDict(
        extra="allow",
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
