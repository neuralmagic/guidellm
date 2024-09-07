"""
This module includes custom CLI parameters for the `click` package.
"""

from typing import Any, Optional

from click import Context, Parameter, ParamType

__all__ = ["MAX_REQUESTS"]


class MaxRequestsType(ParamType):
    """
    Catch the `dataset` string parameter to determine the behavior of the Scheduler.
    """

    name = "max_requests"

    def convert(
        self, value: Any, param: Optional[Parameter], ctx: Optional[Context]
    ) -> Any:
        if isinstance(value, int):
            return value

        try:
            return int(value)
        except ValueError:
            if value == "dataset":
                return value
            else:
                self.fail(f"{value} is not a valid integer or 'dataset'", param, ctx)


MAX_REQUESTS = MaxRequestsType()
