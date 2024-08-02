import os
from typing import Any, Literal, Tuple, Union

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict

from guidellm.utils import is_directory_name, is_file_name

__all__ = ["Serializable", "_Extension"]


_Extension = Union[Literal["yaml"], Literal["json"]]

AVAILABLE_FILE_EXTENSIONS: Tuple[_Extension, ...] = ("yaml", "json")


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

    def save_file(self, path: str, extension: _Extension = "yaml") -> str:
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
            requested_extension = path.split(".")[-1].lower()
            if requested_extension not in AVAILABLE_FILE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension: .{extension}. "
                    f"Expected one of {', '.join(AVAILABLE_FILE_EXTENSIONS)})."
                )

        elif is_directory_name(path):
            file_name = f"{self.__class__.__name__.lower()}.{extension}"
            path = os.path.join(path, file_name)
        else:
            raise ValueError("Output path must be a either directory or file path")

        with open(path, "w") as file:
            if extension == "yaml":
                file.write(self.to_yaml())
            elif extension == "json":
                file.write(self.to_json())
            else:
                raise ValueError(f"Unsupported file format: {extension}")

        logger.info("Successfully saved {} to {}", self.__class__.__name__, path)

        return path

    @classmethod
    def load_file(cls, path: str) -> "Serializable":
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

        if extension not in AVAILABLE_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Expected one of {AVAILABLE_FILE_EXTENSIONS}) "
                f"for {path}"
            )

        with open(path, "r") as file:
            data = file.read()

            if extension == "yaml":
                obj = cls.from_yaml(data)
            elif extension == "json":
                obj = cls.from_json(data)
            else:
                raise ValueError(f"Unsupported file format: {extension}")

        return obj
