import pytest
from pydantic import computed_field

from guidellm.objects.pydantic import StandardBaseModel


class ExampleModel(StandardBaseModel):
    name: str
    age: int

    @computed_field  # type: ignore[misc]
    @property
    def computed(self) -> str:
        return self.name + " " + str(self.age)


@pytest.mark.smoke()
def test_standard_base_model_initialization():
    example = ExampleModel(name="John Doe", age=30)
    assert example.name == "John Doe"
    assert example.age == 30
    assert example.computed == "John Doe 30"


@pytest.mark.smoke()
def test_standard_base_model_invalid_initialization():
    with pytest.raises(ValueError):
        ExampleModel(name="John Doe", age="thirty")  # type: ignore[arg-type]


@pytest.mark.smoke()
def test_standard_base_model_marshalling():
    example = ExampleModel(name="John Doe", age=30)
    serialized = example.model_dump()
    assert serialized["name"] == "John Doe"
    assert serialized["age"] == 30
    assert serialized["computed"] == "John Doe 30"

    serialized["computed"] = "Jane Doe 40"
    deserialized = ExampleModel.model_validate(serialized)
    assert deserialized.name == "John Doe"
    assert deserialized.age == 30
    assert deserialized.computed == "John Doe 30"
