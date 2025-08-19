"""
MessagePack encoding utilities with Pydantic model support.

Provides binary serialization and deserialization of Python objects using MessagePack,
with special handling for Pydantic models to preserve type information and generic
parameters for accurate reconstruction.

Classes:
    MsgpackEncoding: MessagePack encoder/decoder with Pydantic support.
"""

import importlib
from typing import Any

import msgpack
from pydantic import BaseModel

__all__ = ["MsgpackEncoding"]


class MsgpackEncoding:
    """
    MessagePack encoder/decoder with Pydantic model support.

    Provides binary serialization of Python objects with special handling
    for Pydantic models to preserve type information and generic parameters.
    """

    PYDANTIC_TAG = "__pydantic__"
    PYDANTIC_DATA = "data"
    PYDANTIC_ARGS = "args"

    @classmethod
    def encode(cls, obj: Any) -> bytes:
        """
        Encode a Python object to MessagePack binary format.

        :param obj: The object to encode (supports Pydantic models, dicts, lists, etc.).
        :return: Binary MessagePack representation.
        """
        return msgpack.packb(cls.to_primitive(obj), use_bin_type=True)

    @classmethod
    def decode(cls, data: bytes) -> Any:
        """
        Decode MessagePack binary data back to Python objects.

        :param data: Binary MessagePack data to decode.
        :return: Reconstructed Python object with original types preserved.
        """
        return cls.from_primitive(msgpack.unpackb(data, raw=False))

    @classmethod
    def to_primitive(cls, obj: Any) -> Any:
        """
        Convert objects to primitive types for MessagePack serialization.

        Recursively converts complex objects to primitives. Pydantic models are
        converted to tagged dictionaries with type metadata for reconstruction.

        :param obj: The object to convert.
        :return: Primitive representation suitable for MessagePack.
        """
        if isinstance(obj, BaseModel):
            model_cls = obj.__class__

            origin = getattr(model_cls, "__origin__", None)
            if origin is None and hasattr(model_cls, "__pydantic_generic_metadata__"):
                origin = model_cls.__pydantic_generic_metadata__.get("origin", None)
            if origin is None:
                origin = model_cls

            args = getattr(model_cls, "__args__", ())
            if not args and hasattr(model_cls, "__pydantic_generic_metadata__"):
                args = model_cls.__pydantic_generic_metadata__.get("args", ())

            encoded = {
                cls.PYDANTIC_TAG: f"{origin.__module__}.{origin.__name__}",
                # TODO: Review Cursor generated code (start)
                cls.PYDANTIC_DATA: obj.model_dump(exclude_none=False),
                # TODO: Review Cursor generated code (end)
            }

            if args:
                encoded[cls.PYDANTIC_ARGS] = [
                    f"{arg.__module__}.{arg.__name__}" for arg in args
                ]

            return encoded

        if isinstance(obj, dict):
            return {
                cls.to_primitive(key): cls.to_primitive(val) for key, val in obj.items()
            }

        if isinstance(obj, list):
            return [cls.to_primitive(val) for val in obj]

        if isinstance(obj, tuple):
            return tuple(cls.to_primitive(val) for val in obj)

        return obj

    @classmethod
    def from_primitive(cls, obj: Any) -> Any:
        """
        Reconstruct objects from their primitive MessagePack representation.

        Recursively converts primitives back to original objects. Tagged dictionaries
        are restored to Pydantic models with proper types and generic parameters.

        :param obj: The primitive representation to convert.
        :return: Reconstructed object with original types.
        :raises ImportError: If a Pydantic model's module cannot be imported.
        :raises AttributeError: If a class reference cannot be found.
        """
        if isinstance(obj, dict) and cls.PYDANTIC_TAG in obj:
            origin_path = obj[cls.PYDANTIC_TAG]
            module_name, class_name = origin_path.rsplit(".", 1)
            origin_cls = getattr(importlib.import_module(module_name), class_name)

            type_args = []
            if cls.PYDANTIC_ARGS in obj:
                for arg_path in obj[cls.PYDANTIC_ARGS]:
                    mod, clazz = arg_path.rsplit(".", 1)
                    type_args.append(getattr(importlib.import_module(mod), clazz))

            model_cls = origin_cls[tuple(type_args)] if type_args else origin_cls

            return model_cls.model_validate(obj[cls.PYDANTIC_DATA])

        if isinstance(obj, dict):
            return {
                cls.from_primitive(k): cls.from_primitive(v) for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [cls.from_primitive(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(cls.from_primitive(v) for v in obj)

        return obj
