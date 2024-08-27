from pathlib import Path
from typing import Any, Literal, Union, get_args

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict

__all__ = ["Serializable", "SerializableFileType"]


SerializableFileType = Literal["yaml", "json"]


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

        return yaml.dump(self.model_dump())

    @classmethod
    def from_yaml(cls, data: str):
        """
        Deserialize a YAML string to a model instance.

        :param data: YAML string to deserialize.
        :return: An instance of the model.
        """
        logger.debug("Deserializing from YAML... {}", data)

        return cls.model_validate(yaml.safe_load(data))

    def to_json(self) -> str:
        """
        Serialize the model to a JSON string.

        :return: JSON string representation of the model.
        """
        logger.debug("Serializing to JSON... {}", self)

        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str):
        """
        Deserialize a JSON string to a model instance.

        :param data: JSON string to deserialize.
        :return: An instance of the model.
        """
        logger.debug("Deserializing from JSON... {}", data)

        return cls.model_validate_json(data)

    def save_file(
        self,
        path: Union[str, Path],
        type_: SerializableFileType = "yaml",
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
        logger.debug("Saving to file... {} with format: {}", path, type_)

        if isinstance(path, str):
            path = Path(path)

        if path.suffix:
            # is a file
            ext = path.suffix[1:].lower()
            if type_ not in get_args(SerializableFileType):
                raise ValueError(
                    f"Unsupported file extension: {type_}. "
                    f"Expected one of {SerializableFileType} "
                    f"for {path}"
                )
            type_ = ext  # type: ignore # noqa: PGH003
        else:
            # is a directory
            file_name = f"{self.__class__.__name__.lower()}.{type_}"
            path = path / file_name

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as file:
            if type_ == "yaml":
                file.write(self.to_yaml())
            elif type_ == "json":
                file.write(self.to_json())
            else:
                raise ValueError(
                    f"Unsupported file extension: {type_}"
                    f"Expected one of {SerializableFileType} "
                    f"for {path}"
                )

        logger.info("Successfully saved {} to {}", self.__class__.__name__, path)

        return str(path)

    @classmethod
    def load_file(cls, path: Union[str, Path]):
        """
        Load a model from a file in either YAML or JSON format.

        :param path: Path to the file.
        :return: An instance of the model.
        """
        logger.debug("Loading from file... {}", path)

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        extension = path.suffix[1:].lower()

        with path.open() as file:
            data = file.read()

            if extension == "yaml":
                obj = cls.from_yaml(data)
            elif extension == "json":
                obj = cls.from_json(data)
            else:
                raise ValueError(
                    f"Unsupported file extension: {extension}"
                    f"Expected one of {SerializableFileType} "
                    f"for {path}"
                )

        return obj
