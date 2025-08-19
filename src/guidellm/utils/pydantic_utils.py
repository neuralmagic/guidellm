"""
Pydantic utilities for polymorphic model serialization and registry integration.

Provides integration between Pydantic and the registry system, enabling
polymorphic serialization and deserialization of Pydantic models using
a discriminator field and dynamic class registry.

Classes:
    ReloadableBaseModel: Base model with schema reloading capabilities.
    PydanticClassRegistryMixin: Polymorphic Pydantic models with registry support.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from guidellm.utils.registry import RegistryMixin

__all__ = [
    "PydanticClassRegistryMixin",
    "ReloadableBaseModel",
    "StandardBaseDict",
    "StandardBaseModel",
    "StatusBreakdown",
]


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
T = TypeVar("T", bound=BaseModel)


class ReloadableBaseModel(BaseModel):
    """Base Pydantic model with schema reloading capabilities."""

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
        validate_assignment=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def reload_schema(cls):
        """
        Reload the class schema with updated registry information.

        :return: None
        """
        cls.model_rebuild(force=True)


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

    @classmethod
    def get_default(cls: type[T], field: str) -> Any:
        """Get default values for model fields"""
        return cls.model_fields[field].default


class StandardBaseDict(StandardBaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        validate_assignment=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
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


class PydanticClassRegistryMixin(
    ReloadableBaseModel, ABC, RegistryMixin[BaseModelT], Generic[BaseModelT]
):
    """
    Polymorphic Pydantic models with registry-based dynamic instantiation.

    Integrates Pydantic validation with the registry system to enable polymorphic
    serialization and deserialization based on a discriminator field. Automatically
    instantiates the correct subclass during validation based on registry mappings.

    Example:
    ::
        class BaseConfig(PydanticClassRegistryMixin["BaseConfig"]):
            schema_discriminator: ClassVar[str] = "config_type"
            config_type: str = Field(description="Configuration type identifier")

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type["BaseConfig"]:
                return BaseConfig

        @BaseConfig.register("type_a")
        class ConfigA(BaseConfig):
            config_type: str = "type_a"
            value: str = Field(description="Configuration value")

        # Dynamic instantiation
        config = BaseConfig.model_validate({"config_type": "type_a", "value": "test"})
    """

    schema_discriminator: ClassVar[str] = "model_type"

    @classmethod
    def register_decorator(
        cls, clazz: type[BaseModel], name: Optional[str] = None
    ) -> type[BaseModel]:
        """
        Register a Pydantic model class with type validation.

        :param clazz: The Pydantic model class to register.
        :param name: Optional registry name. Defaults to class name if None.
        :return: The registered class.
        :raises TypeError: If clazz is not a Pydantic BaseModel subclass.
        """
        if not issubclass(clazz, BaseModel):
            raise TypeError(
                f"Cannot register {clazz.__name__} as it is not a subclass of "
                "Pydantic BaseModel"
            )

        dec_clazz = super().register_decorator(clazz, name=name)
        cls.reload_schema()

        return dec_clazz

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generate polymorphic validation schema for dynamic instantiation.

        :param source_type: The type for schema generation.
        :param handler: Core schema generation handler.
        :return: Tagged union schema for polymorphic validation.
        """
        if source_type == cls.__pydantic_schema_base_type__():
            if not cls.registry:
                return cls.__pydantic_generate_base_schema__(handler)

            choices = {
                name: handler(model_class) for name, model_class in cls.registry.items()
            }

            return core_schema.tagged_union_schema(
                choices=choices,
                discriminator=cls.schema_discriminator,
            )

        return handler(cls)

    @classmethod
    @abstractmethod
    def __pydantic_schema_base_type__(cls) -> type[BaseModelT]:
        """
        Define the base type for polymorphic validation.

        :return: The base class type for the polymorphic hierarchy.
        """
        ...

    @classmethod
    def __pydantic_generate_base_schema__(
        cls, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generate base schema for polymorphic models without registry.

        :param handler: Core schema generation handler.
        :return: Base CoreSchema accepting any valid input.
        """
        return core_schema.any_schema()

    @classmethod
    def auto_populate_registry(cls) -> bool:
        """
        Initialize registry and reload schema for validation readiness.

        :return: True if registry was populated, False if already populated.
        :raises ValueError: If called when registry_auto_discovery is False.
        """
        populated = super().auto_populate_registry()
        cls.reload_schema()

        return populated
