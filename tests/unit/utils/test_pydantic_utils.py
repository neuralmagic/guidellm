"""
Unit tests for the pydantic_utils module in the Speculators library.
"""

from typing import ClassVar
from unittest import mock

import pytest
from pydantic import BaseModel

from guidellm.utils import PydanticClassRegistryMixin, ReloadableBaseModel

# ===== ReloadableBaseModel Tests =====


@pytest.mark.smoke
def test_reloadable_base_model_initialization():
    class TestModel(ReloadableBaseModel):
        name: str

    model = TestModel(name="test")
    assert model.name == "test"


@pytest.mark.smoke
def test_reloadable_base_model_reload_schema():
    class TestModel(ReloadableBaseModel):
        name: str

    model = TestModel(name="test")
    assert model.name == "test"

    # Mock the model_rebuild method to simulate schema reload
    with mock.patch.object(TestModel, "model_rebuild") as mock_rebuild:
        TestModel.reload_schema()
        mock_rebuild.assert_called_once()


# ===== PydanticClassRegistryMixin Tests =====


@pytest.mark.smoke
def test_pydantic_class_registry_subclass_init():
    class TestBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["TestBaseModel"]:
            return cls

    assert TestBaseModel.registry is None
    assert TestBaseModel.schema_discriminator == "test_type"


@pytest.mark.smoke
def test_pydantic_class_registry_subclass_missing_base_type():
    class InvalidBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

    with pytest.raises(TypeError):
        InvalidBaseModel(test_type="test")  # type: ignore[abstract]


@pytest.mark.sanity
def test_pydantic_class_registry_decorator():
    class TestBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["TestBaseModel"]:
            if cls.__name__ == "TestBaseModel":
                return cls
            return TestBaseModel

    @TestBaseModel.register()
    class TestSubModel(TestBaseModel):
        test_type: str = "TestSubModel"
        value: str

    assert TestBaseModel.registry is not None
    assert "TestSubModel" in TestBaseModel.registry
    assert TestBaseModel.registry["TestSubModel"] is TestSubModel


@pytest.mark.sanity
def test_pydantic_class_registry_decorator_with_name():
    class TestBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["TestBaseModel"]:
            if cls.__name__ == "TestBaseModel":
                return cls
            return TestBaseModel

    @TestBaseModel.register("custom_name")
    class TestSubModel(TestBaseModel):
        test_type: str = "custom_name"
        value: str

    assert TestBaseModel.registry is not None
    assert "custom_name" in TestBaseModel.registry
    assert TestBaseModel.registry["custom_name"] is TestSubModel


@pytest.mark.smoke
def test_pydantic_class_registry_decorator_invalid_type():
    class TestBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["TestBaseModel"]:
            if cls.__name__ == "TestBaseModel":
                return cls
            return TestBaseModel

    class RegularClass:
        pass

    with pytest.raises(TypeError) as exc_info:
        TestBaseModel.register_decorator(RegularClass)  # type: ignore[arg-type]

    assert "not a subclass of Pydantic BaseModel" in str(exc_info.value)


@pytest.mark.smoke
def test_pydantic_class_registry_subclass_marshalling():
    class TestBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["TestBaseModel"]:
            if cls.__name__ == "TestBaseModel":
                return cls
            return TestBaseModel

    @TestBaseModel.register("test_sub")
    class TestSubModel(TestBaseModel):
        test_type: str = "test_sub"
        value: str

    TestBaseModel.reload_schema()

    # Test direct construction of subclass
    sub_instance = TestSubModel(value="test_value")
    assert isinstance(sub_instance, TestSubModel)
    assert sub_instance.test_type == "test_sub"
    assert sub_instance.value == "test_value"

    # Test serialization with model_dump
    dump_data = sub_instance.model_dump()
    assert isinstance(dump_data, dict)
    assert dump_data["test_type"] == "test_sub"
    assert dump_data["value"] == "test_value"

    # Test deserialization via model_validate
    recreated = TestSubModel.model_validate(dump_data)
    assert isinstance(recreated, TestSubModel)
    assert recreated.test_type == "test_sub"
    assert recreated.value == "test_value"

    # Test polymorphic deserialization via base class
    recreated = TestBaseModel.model_validate(dump_data)  # type: ignore[assignment]
    assert isinstance(recreated, TestSubModel)
    assert recreated.test_type == "test_sub"
    assert recreated.value == "test_value"


@pytest.mark.smoke
def test_pydantic_class_registry_parent_class_marshalling():
    class TestBaseModel(PydanticClassRegistryMixin):
        schema_discriminator: ClassVar[str] = "test_type"
        test_type: str

        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["TestBaseModel"]:
            if cls.__name__ == "TestBaseModel":
                return cls
            return TestBaseModel

        @classmethod
        def __pydantic_generate_base_schema__(cls, handler):
            return handler(cls)

    @TestBaseModel.register("sub_a")
    class TestSubModelA(TestBaseModel):
        test_type: str = "sub_a"
        value_a: str

    @TestBaseModel.register("sub_b")
    class TestSubModelB(TestBaseModel):
        test_type: str = "sub_b"
        value_b: int

    class ContainerModel(BaseModel):
        name: str
        model: TestBaseModel
        models: list[TestBaseModel]

    sub_a = TestSubModelA(value_a="test")
    sub_b = TestSubModelB(value_b=123)

    container = ContainerModel(name="container", model=sub_a, models=[sub_a, sub_b])
    assert isinstance(container.model, TestSubModelA)
    assert container.model.test_type == "sub_a"
    assert container.model.value_a == "test"
    assert isinstance(container.models[0], TestSubModelA)
    assert isinstance(container.models[1], TestSubModelB)
    assert container.models[0].test_type == "sub_a"
    assert container.models[1].test_type == "sub_b"
    assert container.models[0].value_a == "test"
    assert container.models[1].value_b == 123

    # Test serialization with model_dump
    dump_data = container.model_dump()
    assert isinstance(dump_data, dict)
    assert dump_data["name"] == "container"
    assert dump_data["model"]["test_type"] == "sub_a"
    assert dump_data["model"]["value_a"] == "test"
    assert len(dump_data["models"]) == 2
    assert dump_data["models"][0]["test_type"] == "sub_a"
    assert dump_data["models"][0]["value_a"] == "test"
    assert dump_data["models"][1]["test_type"] == "sub_b"
    assert dump_data["models"][1]["value_b"] == 123

    # Test deserialization via model_validate
    recreated = ContainerModel.model_validate(dump_data)
    assert isinstance(recreated, ContainerModel)
    assert recreated.name == "container"
    assert isinstance(recreated.model, TestSubModelA)
    assert recreated.model.test_type == "sub_a"
    assert recreated.model.value_a == "test"
    assert len(recreated.models) == 2
    assert isinstance(recreated.models[0], TestSubModelA)
    assert isinstance(recreated.models[1], TestSubModelB)
    assert recreated.models[0].test_type == "sub_a"
    assert recreated.models[1].test_type == "sub_b"
    assert recreated.models[0].value_a == "test"
    assert recreated.models[1].value_b == 123
