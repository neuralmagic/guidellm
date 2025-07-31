"""
Registry system for classes in the GuideLLM toolkit.

This module provides a flexible class registration and discovery system used
throughout the GuideLLM toolkit. It enables automatic registration of classes
and discovery of implementations through class decorators and module imports.

The registry system is used to track different implementations of token proposal
methods, speculative decoding algorithms, and speculator models, allowing for
dynamic discovery and instantiation based on configuration parameters.

Classes:
    ClassRegistryMixin: Base mixin for creating class registries with decorators
        and optional auto-discovery capabilities through registry_auto_discovery flag.
    AutoClassRegistryMixin: A backward-compatible version of ClassRegistryMixin with
        auto-discovery enabled by default
"""

from typing import Any, Callable, ClassVar, Optional

__all__ = ["ClassRegistryMixin"]


class ClassRegistryMixin:
    """
    A mixin class that provides a registration system for tracking class
    implementations with optional auto-discovery capabilities.

    This mixin allows classes to maintain a registry of subclasses that can be
    dynamically discovered and instantiated. Classes that inherit from this mixin
    can use the @register decorator to add themselves to the registry.

    The registry is class-specific, meaning each class that inherits from this mixin
    will have its own separate registry of implementations.

    The mixin can also be configured to automatically discover and register classes
    from specified packages by setting registry_auto_discovery=True and defining
    an auto_package class variable to specify which package(s) should be automatically
    imported to discover implementations.

    Example:
    ::
        class BaseAlgorithm(ClassRegistryMixin):
            pass

        @BaseAlgorithm.register()
        class ConcreteAlgorithm(BaseAlgorithm):
            pass

        @BaseAlgorithm.register("custom_name")
        class AnotherAlgorithm(BaseAlgorithm):
            pass

        # Get all registered algorithm implementations
        algorithms = BaseAlgorithm.registered_classes()

    :cvar registry: A dictionary mapping class names to classes that have been
        registered to the extending subclass through the @subclass.register() decorator
    :cvar registry_auto_discovery: A flag that enables automatic discovery and import of
        modules from the auto_package when set to True. Default is False.
    :cvar registry_populated: A flag that tracks whether the registry has been
        populated with classes from the specified package(s).
    """

    registry: ClassVar[Optional[dict[str, type[Any]]]] = None
    registry_auto_discovery: ClassVar[bool] = False
    registry_populated: ClassVar[bool] = False

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[type[Any]], type[Any]]:
        """
        An invoked class decorator that registers that class with the registry under
        either the provided name or the class name if no name is provided.

        Example:
        ```python
        @ClassRegistryMixin.register()
        class ExampleClass:
            ...

        @ClassRegistryMixin.register("custom_name")
        class AnotherExampleClass:
            ...
        ```

        :param name: Optional name to register the class under. If None, the class name
            is used as the registry key.
        :return: A decorator function that registers the decorated class.
        :raises ValueError: If name is provided but is not a string.
        """
        if name is not None and not isinstance(name, str):
            raise ValueError(
                "ClassRegistryMixin.register() name must be a string or None. "
                f"Got {name}."
            )

        return lambda subclass: cls.register_decorator(subclass, name=name)

    @classmethod
    def register_decorator(
        cls, clazz: type[Any], name: Optional[str] = None
    ) -> type[Any]:
        """
        A non-invoked class decorator that registers the class with the registry.
        If passed through a lambda, then name can be passed in as well.
        Otherwise, the only argument is the decorated class.

        Example:
        ```python
        @ClassRegistryMixin.register_decorator
        class ExampleClass:
            ...
        ```

        :param clazz: The class to register
        :param name: Optional name to register the class under. If None, the class name
            is used as the registry key.
        :return: The registered class.
        :raises TypeError: If the decorator is used incorrectly or if the class is not
            a type.
        :raises ValueError: If the class is already registered or if name is provided
            but is not a string.
        """

        if not isinstance(clazz, type):
            raise TypeError(
                "ClassRegistryMixin.register_decorator must be used as a class "
                "decorator and without invocation."
                f"Got improper clazz arg {clazz}."
            )

        if not name:
            name = clazz.__name__
        elif not isinstance(name, str):
            raise ValueError(
                "ClassRegistryMixin.register_decorator must be used as a class "
                "decorator and without invocation. "
                f"Got imporoper name arg {name}."
            )

        if cls.registry is None:
            cls.registry = {}

        if name in cls.registry:
            raise ValueError(
                f"ClassRegistryMixin.register_decorator cannot register a class "
                f"{clazz} with the name {name} because it is already registered."
            )

        cls.registry[name] = clazz

        return clazz
