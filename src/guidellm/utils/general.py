from typing import Final

__all__ = ["UNSET", "UnsetType"]


class UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = UnsetType()
