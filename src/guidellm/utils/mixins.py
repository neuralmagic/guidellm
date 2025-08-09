"""
Mixin classes for common metadata extraction and object introspection.

Provides reusable mixins for extracting structured metadata from objects,
enabling consistent information exposure across different class hierarchies.

Classes:
    InfoMixin: Mixin providing standardized metadata extraction capabilities.
"""

from typing import Any

__all__ = ["InfoMixin"]


class InfoMixin:
    """Mixin class providing standardized metadata extraction for introspection."""

    @classmethod
    def extract_from_obj(cls, obj: Any) -> dict[str, Any]:
        """
        Extract structured metadata from any object.

        Attempts to use the object's own `info` method or property if available,
        otherwise constructs metadata from object attributes and type information.

        :param obj: Object to extract metadata from.
        :return: Dictionary containing object metadata including type, class,
            module, and public attributes.
        """
        if hasattr(obj, "info"):
            return obj.info() if callable(obj.info) else obj.info

        return {
            "str": str(obj),
            "type": type(obj).__name__,
            "class": obj.__class__.__name__ if hasattr(obj, "__class__") else None,
            "module": obj.__class__.__module__ if hasattr(obj, "__class__") else None,
            "attributes": (
                {
                    key: val
                    if isinstance(val, (str, int, float, bool, list, dict))
                    else str(val)
                    for key, val in obj.__dict__.items()
                    if not key.startswith("_")
                }
                if hasattr(obj, "__dict__")
                else {}
            ),
        }

    @property
    def info(self) -> dict[str, Any]:
        """
        Return structured metadata about this instance.

        :return: Dictionary containing class name, module, and public attributes.
        """
        return self.extract_from_obj(self)
