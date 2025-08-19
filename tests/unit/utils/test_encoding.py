from typing import Any, Generic, TypeVar

import pytest
from pydantic import BaseModel, Field

from guidellm.utils.encoding import MsgpackEncoding


class SimpleModel(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    simple: SimpleModel
    items: list[str]
    metadata: dict[str, Any]


T = TypeVar("T")


class GenericModel(BaseModel, Generic[T]):
    data: T
    count: int


class ComplexModel(BaseModel):
    id: str = Field(description="Unique identifier")
    nested: NestedModel
    numbers: list[int]
    mapping: dict[str, SimpleModel]


class TestMsgpackEncoding:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "primitive_data",
        [
            # Basic primitives
            42,
            3.14,
            True,
            False,
            None,
            "hello world",
            "",
            [],
            [1, 2, 3],
            {},
            {"key": "value"},
            # Nested collections
            [1, [2, 3], {"nested": True}],
            {"outer": {"inner": [1, 2, 3]}},
            # Mixed types
            [1, "string", 3.14, True, None],
            {"int": 42, "str": "hello", "float": 3.14, "bool": True, "null": None},
        ],
    )
    def test_encode_decode_primitives(self, primitive_data):
        """Test encoding and decoding of Python primitives and collections."""
        encoded = MsgpackEncoding.encode(primitive_data)
        assert isinstance(encoded, bytes)

        decoded = MsgpackEncoding.decode(encoded)
        assert decoded == primitive_data
        assert isinstance(decoded, type(primitive_data))

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("tuple_data", "expected_list"),
        [
            ((), []),
            ((1, 2, 3), [1, 2, 3]),
            ((1, (2, 3), {"tuple_dict": True}), [1, [2, 3], {"tuple_dict": True}]),
        ],
    )
    def test_encode_decode_tuples(self, tuple_data, expected_list):
        encoded = MsgpackEncoding.encode(tuple_data)
        assert isinstance(encoded, bytes)

        decoded = MsgpackEncoding.decode(encoded)
        assert decoded == expected_list
        assert isinstance(decoded, list)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "model_data",
        [
            SimpleModel(name="test", value=42),
            NestedModel(
                simple=SimpleModel(name="nested", value=100),
                items=["a", "b", "c"],
                metadata={"key": "value", "number": 123},
            ),
            ComplexModel(
                id="test-123",
                nested=NestedModel(
                    simple=SimpleModel(name="complex", value=999),
                    items=["x", "y"],
                    metadata={"complex": True},
                ),
                numbers=[1, 2, 3, 4, 5],
                mapping={
                    "first": SimpleModel(name="first", value=1),
                    "second": SimpleModel(name="second", value=2),
                },
            ),
        ],
    )
    def test_encode_decode_pydantic_models(self, model_data):
        """Test encoding and decoding of Pydantic models."""
        encoded = MsgpackEncoding.encode(model_data)
        assert isinstance(encoded, bytes)

        decoded = MsgpackEncoding.decode(encoded)
        assert decoded == model_data
        assert isinstance(decoded, type(model_data))
        assert decoded.model_dump() == model_data.model_dump()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("generic_model", "expected_type"),
        [
            (GenericModel[str](data="hello", count=1), str),
            (GenericModel[int](data=42, count=2), int),
            (GenericModel[list[str]](data=["a", "b"], count=3), list),
        ],
    )
    def test_encode_decode_generic_models(self, generic_model, expected_type):
        """Test encoding and decoding of generic Pydantic models."""
        encoded = MsgpackEncoding.encode(generic_model)
        assert isinstance(encoded, bytes)

        decoded = MsgpackEncoding.decode(encoded)
        assert decoded == generic_model
        assert decoded.data == generic_model.data
        assert decoded.count == generic_model.count
        assert isinstance(decoded.data, expected_type)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "mixed_data",
        [
            [SimpleModel(name="item1", value=1), SimpleModel(name="item2", value=2)],
            {"model": SimpleModel(name="dict_value", value=42), "primitive": "string"},
            {
                "models": [
                    SimpleModel(name="item1", value=1),
                    SimpleModel(name="item2", value=2),
                ],
                "data": {"nested": {"deep": SimpleModel(name="deep", value=999)}},
            },
            [
                {
                    "id": "test",
                    "model": NestedModel(
                        simple=SimpleModel(name="nested_in_list", value=456),
                        items=["nested", "list"],
                        metadata={"in_list": True},
                    ),
                    "primitives": [1, 2, 3],
                }
            ],
        ],
    )
    def test_encode_decode_mixed_collections(self, mixed_data):
        encoded = MsgpackEncoding.encode(mixed_data)
        assert isinstance(encoded, bytes)

        decoded = MsgpackEncoding.decode(encoded)
        assert decoded == mixed_data
        assert isinstance(decoded, type(mixed_data))

    @pytest.mark.smoke
    def test_round_trip_consistency(self):
        original_data = {
            "simple": SimpleModel(name="test", value=42),
            "nested": NestedModel(
                simple=SimpleModel(name="nested", value=100),
                items=["a", "b", "c"],
                metadata={"key": "value"},
            ),
            "primitives": [1, 2, 3, "string", True, None],
            "list_data": [1, 2, SimpleModel(name="list", value=999)],
        }

        current_data = original_data
        for _ in range(3):
            encoded = MsgpackEncoding.encode(current_data)
            current_data = MsgpackEncoding.decode(encoded)

        assert current_data == original_data

    @pytest.mark.smoke
    def test_empty_collections(self):
        test_cases = [[], {}]

        for empty_collection in test_cases:
            encoded = MsgpackEncoding.encode(empty_collection)
            decoded = MsgpackEncoding.decode(encoded)
            assert decoded == empty_collection
            assert isinstance(decoded, type(empty_collection))

    @pytest.mark.smoke
    def test_pydantic_constants(self):
        """Test that the Pydantic-related constants are properly defined."""
        assert MsgpackEncoding.PYDANTIC_TAG == "__pydantic__"
        assert MsgpackEncoding.PYDANTIC_DATA == "data"
        assert MsgpackEncoding.PYDANTIC_ARGS == "args"

    @pytest.mark.sanity
    def test_encode_invalid_data(self):
        """Test encoding behavior with edge cases."""

        class CustomClass:
            def __init__(self, value):
                self.value = value

        custom_obj = CustomClass(42)
        primitive = MsgpackEncoding.to_primitive(custom_obj)
        assert primitive is custom_obj
