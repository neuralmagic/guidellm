"""
Automatic module importing utilities for dynamic class discovery.

This module provides a mixin class for automatic module importing within a package,
enabling dynamic discovery of classes and implementations without explicit imports.
It is particularly useful for auto-registering classes in a registry pattern where
subclasses need to be discoverable at runtime.

The AutoImporterMixin can be combined with registration mechanisms to create
extensible systems where new implementations are automatically discovered and
registered when they are placed in the correct package structure.

Classes:
    - AutoImporterMixin: A mixin class that provides functionality to automatically
        import all modules within a specified package or list of packa
"""

import importlib
import pkgutil
import sys
from typing import ClassVar, Optional, Union

__all__ = ["AutoImporterMixin"]


class AutoImporterMixin:
    """
    A mixin class that provides functionality to automatically import all modules
    within a specified package or list of packages.

    This mixin is designed to be used with class registration mechanisms to enable
    automatic discovery and registration of classes without explicit imports. When
    a class inherits from AutoImporterMixin, it can define the package(s) to scan
    for modules by setting the `auto_package` class variable.

    Usage Example:
    ```python
    from speculators.utils import AutoImporterMixin
    class MyRegistry(AutoImporterMixin):
        auto_package = "my_package.implementations"

    MyRegistry.auto_import_package_modules()
    ```

    :cvar auto_package: The package name or tuple of names to import modules from.
    :cvar auto_ignore_modules: Optional tuple of module names to ignore during import.
    :cvar auto_imported_modules: List tracking which modules have been imported.
    """

    auto_package: ClassVar[Optional[Union[str, tuple[str, ...]]]] = None
    auto_ignore_modules: ClassVar[Optional[tuple[str, ...]]] = None
    auto_imported_modules: ClassVar[Optional[list]] = None

    @classmethod
    def auto_import_package_modules(cls):
        """
        Automatically imports all modules within the specified package(s).

        This method scans the package(s) defined in the `auto_package` class variable
        and imports all modules found, tracking them in `auto_imported_modules`. It
        skips packages (directories) and any modules listed in `auto_ignore_modules`.

        :raises ValueError: If the `auto_package` class variable is not set
        """
        if cls.auto_package is None:
            raise ValueError(
                "The class variable 'auto_package' must be set to the package name to "
                "import modules from."
            )

        cls.auto_imported_modules = []
        packages = (
            cls.auto_package
            if isinstance(cls.auto_package, tuple)
            else (cls.auto_package,)
        )

        for package_name in packages:
            package = importlib.import_module(package_name)

            for _, module_name, is_pkg in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            ):
                if (
                    is_pkg
                    or (
                        cls.auto_ignore_modules is not None
                        and module_name in cls.auto_ignore_modules
                    )
                    or module_name in cls.auto_imported_modules
                ):
                    # Skip packages and ignored modules
                    continue

                if module_name in sys.modules:
                    # Avoid circular imports
                    cls.auto_imported_modules.append(module_name)
                else:
                    importlib.import_module(module_name)
                    cls.auto_imported_modules.append(module_name)
