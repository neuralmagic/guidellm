import importlib
import sys
from typing import Tuple


def _extract_python_version(data: str) -> Tuple[int, ...]:
    """Extract '3.12' -> (3, 12)."""

    if len(items := data.split(".")) > 2:
        raise ValueError("Python version format: MAJOR.MINOR")

    if not all(item.isnumeric() for item in items):
        raise ValueError("Python version must include only numbers")

    return tuple(int(item) for item in items)


def check_python_version(min_version: str, max_version: str) -> None:
    """Raises the error if the current version is not in the range."""

    min_version_info: Tuple[int, ...] = _extract_python_version(min_version)
    max_version_info: Tuple[int, ...] = _extract_python_version(max_version)
    current_version_info: Tuple[int, int] = (
        sys.version_info.major,
        sys.version_info.minor,
    )

    if not (min_version_info <= current_version_info <= max_version_info):
        raise RuntimeError(
            "This feature requires Python version "
            f"to be in range: {min_version}..{max_version}."
            "You are using Python {}.{}.{}".format(
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            )
        )


def module_is_available(module: str, helper: str):
    """Ensure that the module is available for other project components."""

    try:
        importlib.import_module(module)
    except ImportError:
        raise RuntimeError(f"Module '{module}' is not available. {helper}") from None
