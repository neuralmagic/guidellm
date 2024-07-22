from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict


class Serializable(BaseModel):
    """
    A base class for models that require YAML and JSON serialization and
    deserialization.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_encoders={
            BaseException: lambda value: str(value),
        },
    )

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        logger.debug(
            "Initialized new instance of {} with data: {}",
            self.__class__.__name__,
            data,
        )

    def to_yaml(self) -> str:
        """
        Serialize the model to a YAML string.

        :return: YAML string representation of the model.
        """
        logger.debug("Serializing to YAML... {}", self)
        yaml_str = yaml.dump(self.model_dump())

        return yaml_str

    @classmethod
    def from_yaml(cls, data: str):
        """
        Deserialize a YAML string to a model instance.

        :param data: YAML string to deserialize.
        :return: An instance of the model.
        """
        logger.debug("Deserializing from YAML... {}", data)
        obj = cls.model_validate(yaml.safe_load(data))

        return obj

    def to_json(self) -> str:
        """
        Serialize the model to a JSON string.

        :return: JSON string representation of the model.
        """
        logger.debug("Serializing to JSON... {}", self)
        json_str = self.model_dump_json()

        return json_str

    @classmethod
    def from_json(cls, data: str):
        """
        Deserialize a JSON string to a model instance.

        :param data: JSON string to deserialize.
        :return: An instance of the model.
        """
        logger.debug("Deserializing from JSON... {}", data)
        obj = cls.model_validate_json(data)

        return obj
