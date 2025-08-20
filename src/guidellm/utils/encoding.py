"""
MessagePack encoding utilities with Pydantic model support.

Provides binary serialization and deserialization of Python objects using MessagePack,
with special handling for Pydantic models to preserve type information and generic
parameters for accurate reconstruction.

Classes:
    MsgpackEncoding: MessagePack encoder/decoder with Pydantic support.
"""

import importlib
from typing import Any, get_args, get_origin

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
            # Get the module, class, and any generics for reconstruction later
            model_cls = obj.__class__
            origin = get_origin(model_cls) or model_cls
            args = tuple(get_args(model_cls))
            if not args and hasattr(model_cls, "__pydantic_generic_metadata__"):
                meta = model_cls.__pydantic_generic_metadata__
                origin = meta.get("origin", origin) or origin
                args = tuple(meta.get("args") or [])

            # Construct data by manually running model_dump and encoding BaseModel
            data: dict[str, Any] = {}
            for name in origin.model_fields:
                value = getattr(obj, name, None)
                data[name] = cls.to_primitive(value)
            extras = getattr(obj, "__pydantic_extras__", {})
            for name, value in extras.items():
                data[name] = cls.to_primitive(value)

            encoded = {
                cls.PYDANTIC_TAG: f"{origin.__module__}.{origin.__name__}",
                cls.PYDANTIC_DATA: data,
            }

            if args:
                encoded[cls.PYDANTIC_ARGS] = [
                    f"{arg.__module__}.{arg.__qualname__}"
                    for arg in args
                    if isinstance(arg, type)
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
            payload = {
                key: cls.from_primitive(value)
                for key, value in obj[cls.PYDANTIC_DATA].items()
            }

            return model_cls.model_validate(payload)

        if isinstance(obj, dict):
            return {
                cls.from_primitive(k): cls.from_primitive(v) for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [cls.from_primitive(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(cls.from_primitive(v) for v in obj)

        return obj
