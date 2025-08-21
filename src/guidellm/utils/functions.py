"""
Utility functions for safe operations and value handling.

Provides defensive programming utilities for common operations that may encounter
None values, invalid inputs, or edge cases. Includes safe arithmetic operations,
attribute access, and timestamp formatting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

__all__ = [
    "all_defined",
    "safe_add",
    "safe_divide",
    "safe_format_timestamp",
    "safe_getattr",
    "safe_multiply",
]


def safe_getattr(obj: Any | None, attr: str, default: Any = None) -> Any:
    """
    Safely get an attribute from an object with None handling.

    :param obj: Object to get the attribute from, or None
    :param attr: Name of the attribute to retrieve
    :param default: Value to return if object is None or attribute doesn't exist
    :return: Attribute value or default if not found or object is None
    """
    if obj is None:
        return default

    return getattr(obj, attr, default)


def all_defined(*values: Any | None) -> bool:
    """
    Check if all provided values are defined (not None).

    :param values: Variable number of values to check for None
    :return: True if all values are not None, False otherwise
    """
    return all(value is not None for value in values)


def safe_divide(
    numerator: int | float | None,
    denominator: int | float | None,
    num_default: float = 0.0,
    den_default: float = 1.0,
) -> float:
    """
    Safely divide two numbers with None handling and zero protection.

    :param numerator: Number to divide, or None to use num_default
    :param denominator: Number to divide by, or None to use den_default
    :param num_default: Default value for numerator if None
    :param den_default: Default value for denominator if None
    :return: Division result with protection against division by zero
    """
    numerator = numerator if numerator is not None else num_default
    denominator = denominator if denominator is not None else den_default

    return numerator / (denominator or 1e-10)


def safe_multiply(*values: int | float | None, default: float = 1.0) -> float:
    """
    Safely multiply multiple numbers with None handling.

    :param values: Variable number of values to multiply, None values treated as 1.0
    :param default: Starting value for multiplication
    :return: Product of all non-None values multiplied by default
    """
    result = default
    for val in values:
        result *= val if val is not None else 1.0
    return result


def safe_add(
    *values: int | float | None, signs: list[int] | None = None, default: float = 0.0
) -> float:
    """
    Safely add multiple numbers with None handling and optional signs.

    :param values: Variable number of values to add, None values use default
    :param signs: Optional list of 1 (add) or -1 (subtract) for each value.
        If None, all values are added. Must match length of values.
    :param default: Value to substitute for None values
    :return: Result of adding all values safely (default used when value is None)
    """
    if not values:
        return default

    values = list(values)

    if signs is None:
        signs = [1] * len(values)

    if len(signs) != len(values):
        raise ValueError("Length of signs must match length of values")

    result = values[0] if values[0] is not None else default

    for ind in range(1, len(values)):
        val = values[ind] if values[ind] is not None else default
        result += signs[ind] * val

    return result


def safe_format_timestamp(
    timestamp: float | None, format_: str = "%H:%M:%S", default: str = "N/A"
) -> str:
    """
    Safely format a timestamp with error handling and validation.

    :param timestamp: Unix timestamp to format, or None
    :param format_: Strftime format string for timestamp formatting
    :param default: Value to return if timestamp is invalid or None
    :return: Formatted timestamp string or default value
    """
    if timestamp is None or timestamp < 0 or timestamp > 2**31:
        return default

    try:
        return datetime.fromtimestamp(timestamp).strftime(format_)
    except (ValueError, OverflowError, OSError):
        return default
