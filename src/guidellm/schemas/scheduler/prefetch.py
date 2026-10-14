"""Shared prefetch argument type for profiles and scheduling strategies."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, NonNegativeInt

__all__ = ["Prefetch"]


def normalize_prefetch(value: Any) -> Any:
    """
    Map the load-everything aliases onto ``"all"``.

    ``-1``, ``"-1"``, and ``"all"`` are the same input. Other values are left
    for the field type to validate.

    :param value: Raw prefetch argument
    :return: ``"all"`` for the load-everything aliases, otherwise ``value``
    """
    if value in (-1, "-1", "all"):
        return "all"
    return value


Prefetch = Annotated[
    NonNegativeInt | Literal["start", "all"],
    BeforeValidator(normalize_prefetch),
]
