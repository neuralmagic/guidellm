import pytest

from guidellm.core.serializable import Serializable


class ExampleModel(Serializable):
    name: str
    age: int


@pytest.mark.smoke
def test_serializable_to_json():
    example = ExampleModel(name="John Doe", age=30)
    json_str = example.to_json()
    assert '"name":"John Doe"' in json_str
    assert '"age":30' in json_str


@pytest.mark.smoke
def test_serializable_from_json():
    json_str = '{"name": "John Doe", "age": 30}'
    example = ExampleModel.from_json(json_str)
    assert example.name == "John Doe"
    assert example.age == 30


@pytest.mark.smoke
def test_serializable_to_yaml():
    example = ExampleModel(name="John Doe", age=30)
    yaml_str = example.to_yaml()
    assert "name: John Doe" in yaml_str
    assert "age: 30" in yaml_str


@pytest.mark.smoke
def test_serializable_from_yaml():
    yaml_str = "name: John Doe\nage: 30\n"
    example = ExampleModel.from_yaml(yaml_str)
    assert example.name == "John Doe"
    assert example.age == 30
