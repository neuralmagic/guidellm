from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel


class Serializable(BaseModel):
    """
    A base class for models that require YAML and JSON serialization and
    deserialization.
    """

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
        logger.debug("Serialized to YAML: {}", yaml_str)

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
        logger.debug("Deserialized from YAML: {}", obj)

        return obj

    def to_json(self) -> str:
        """
        Serialize the model to a JSON string.

        :return: JSON string representation of the model.
        """
        logger.debug("Serializing to JSON... {}", self)
        json_str = self.model_dump_json()
        logger.debug("Serialized to JSON: {}", json_str)

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
        logger.debug("Deserialized from JSON: {}", obj)

        return obj
