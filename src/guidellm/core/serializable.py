from typing import Any, Optional

import os
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict
from enum import Enum

from guidellm.utils import is_file_name


__all__ = ["Serializable", "SerializableFileType"]


class SerializableFileType(Enum):
    """
    Enum class for file types supported by Serializable.
    """

    YAML = "yaml"
    JSON = "json"


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

    def save_file(self, path: str, type_: Optional[SerializableFileType] = None) -> str:
        """
        Save the model to a file in either YAML or JSON format.

        :param path: Path to the exact file or the containing directory.
            If it is a directory, the file name will be inferred from the class name.
        :param type_: Optional type to save ('yaml' or 'json').
            If not provided and the path has an extension,
            it will be inferred to save in that format.
            If not provided and the path does not have an extension,
            it will save in YAML format.
        :return: The path to the saved file.
        """
        logger.debug("Saving to file... {} with format: {}", path, type_)

        if not is_file_name(path):
            file_name = f"{self.__class__.__name__.lower()}"
            if type_:
                file_name += f".{type_.value.lower()}"
            else:
                file_name += ".yaml"
                type_ = SerializableFileType.YAML
            path = os.path.join(path, file_name)

        if not type_:
            extension = path.split(".")[-1].upper()

            if extension not in SerializableFileType.__members__:
                raise ValueError(
                    f"Unsupported file extension: {extension}. "
                    f"Expected one of {', '.join(SerializableFileType.__members__)}) "
                    f"for {path}"
                )

            type_ = SerializableFileType[extension]

        if type_.name not in SerializableFileType.__members__:
            raise ValueError(
                f"Unsupported file format: {type_} "
                f"(expected 'yaml' or 'json') for {path}"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as file:
            if type_ == SerializableFileType.YAML:
                file.write(self.to_yaml())
            elif type_ == SerializableFileType.JSON:
                file.write(self.to_json())
            else:
                raise ValueError(f"Unsupported file format: {type_}")

        logger.info("Successfully saved {} to {}", self.__class__.__name__, path)

        return path

    @classmethod
    def load_file(cls, path: str):
        """
        Load a model from a file in either YAML or JSON format.

        :param path: Path to the file.
        :return: An instance of the model.
        """
        logger.debug("Loading from file... {}", path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        elif not os.path.isfile(path):
            raise ValueError(f"Path is not a file: {path}")

        extension = path.split(".")[-1].upper()

        if extension not in SerializableFileType.__members__:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Expected one of {', '.join(SerializableFileType.__members__)}) "
                f"for {path}"
            )

        type_ = SerializableFileType[extension]

        with open(path, "r") as file:
            data = file.read()

            if type_ == SerializableFileType.YAML:
                obj = cls.from_yaml(data)
            elif type_ == SerializableFileType.JSON:
                obj = cls.from_json(data)
            else:
                raise ValueError(f"Unsupported file format: {type_}")

        return obj
