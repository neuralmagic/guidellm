import functools
import os
from enum import Enum
from typing import Any, List, Union

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict

from guidellm.utils import is_directory_name, is_file_name

__all__ = ["Serializable", "SerializableFileExtensions"]


class SerializableFileExtensions(str, Enum):
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

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def available_file_extensions() -> List[str]:
        """
        Returns string representation of available filetypes that
        are supported by the Serializable class.
        """

        return [item.value for item in SerializableFileExtensions]

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

    def save_file(
        self,
        path: str,
        extension: Union[
            SerializableFileExtensions, str
        ] = SerializableFileExtensions.YAML,
    ) -> str:
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

        if is_file_name(path):
            extension = path.split(".")[-1].lower()
            if extension not in self.available_file_extensions():
                raise ValueError(
                    f"Unsupported file extension: .{str(extension)}. "
                    f"Expected one of {', '.join(self.available_file_extensions())})."
                )
        elif is_directory_name(path):
            file_name = f"{self.__class__.__name__.lower()}.{extension}"
            path = os.path.join(path, file_name)
        else:
            raise ValueError("Output path must be a either directory or file path")

        with open(path, "w") as file:
            if extension == SerializableFileExtensions.YAML:
                file.write(self.to_yaml())
            elif extension == SerializableFileExtensions.JSON:
                file.write(self.to_json())
            else:
                raise ValueError(f"Unsupported file format: {extension}")

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

        extension = path.split(".")[-1].lower()

        if extension not in cls.available_file_extensions():
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Expected one of {cls.available_file_extensions()}) "
                f"for {path}"
            )

        with open(path, "r") as file:
            data = file.read()

            if extension == SerializableFileExtensions.YAML:
                obj = cls.from_yaml(data)
            elif extension == SerializableFileExtensions.JSON:
                obj = cls.from_json(data)
            else:
                raise ValueError(f"Unsupported file format: {extension}")

        return obj
