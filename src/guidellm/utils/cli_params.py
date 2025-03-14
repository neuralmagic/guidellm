"""
This module includes custom CLI parameters for the `click` package.
"""

from typing import Any, Optional

from click import BadParameter, Context, Parameter, ParamType

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


class Union(ParamType):
    """
    A custom click parameter type that allows for multiple types to be accepted.
    """

    def __init__(self, *types: ParamType):
        self.types = types
        self.name = "".join(t.name for t in types)

    def convert(self, value, param, ctx):
        fails = []
        for t in self.types:
            try:
                return t.convert(value, param, ctx)
            except BadParameter as e:
                fails.append(str(e))
                continue

        self.fail("; ".join(fails) or f"Invalid value: {value}")


    def get_metavar(self, param: Parameter) -> str:
        def get_choices(t: ParamType) -> str:
            meta = t.get_metavar(param)
            return meta if meta is not None else t.name

        # Get the choices for each type in the union.
        choices_str = "|".join(map(get_choices, self.types))

        # Use curly braces to indicate a required argument.
        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"

        # Use square braces to indicate an option or optional argument.
        return f"[{choices_str}]"
