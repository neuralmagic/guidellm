import os
import tempfile

import pytest

from guidellm.core.serializable import Serializable, SerializableFileExtensions


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


@pytest.mark.smoke
def test_serializable_file_json():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "example.json")
        saved_path = example.save_file(file_path, SerializableFileExtensions.JSON)
        assert os.path.exists(saved_path)
        loaded_example = ExampleModel.load_file(saved_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30


@pytest.mark.smoke
def test_serializable_file_yaml():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "example.yaml")
        saved_path = example.save_file(file_path, SerializableFileExtensions.YAML)
        assert os.path.exists(saved_path)
        loaded_example = ExampleModel.load_file(saved_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30


@pytest.mark.smoke
def test_serializable_file_without_extension():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_path = example.save_file(temp_dir)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".yaml")
        loaded_example = ExampleModel.load_file(saved_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30


@pytest.mark.smoke
def test_serializable_file_with_directory_json():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_path = example.save_file(temp_dir, SerializableFileExtensions.JSON)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".json")
        loaded_example = ExampleModel.load_file(saved_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30


@pytest.mark.smoke
def test_serializable_file_with_directory_yaml():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_path = example.save_file(temp_dir, SerializableFileExtensions.YAML)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".yaml")
        loaded_example = ExampleModel.load_file(saved_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30


@pytest.mark.smoke
def test_serializable_save_file_invalid_extension():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_file_path = os.path.join(temp_dir, "example.txt")
        with pytest.raises(ValueError, match="Unsupported file extension.*"):
            example.save_file(invalid_file_path)


@pytest.mark.smoke
def test_serializable_load_file_invalid_extension():
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_file_path = os.path.join(temp_dir, "example.txt")
        with open(invalid_file_path, "w") as file:
            file.write("invalid content")
        with pytest.raises(ValueError, match="Unsupported file extension: TXT"):
            ExampleModel.load_file(invalid_file_path)


@pytest.mark.smoke
def test_serializable_file_no_type_provided():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "example")
        saved_path = example.save_file(file_path)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".yaml")
        loaded_example = ExampleModel.load_file(saved_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30


@pytest.mark.smoke
def test_serializable_file_infer_extension():
    example = ExampleModel(name="John Doe", age=30)
    with tempfile.TemporaryDirectory() as temp_dir:
        inferred_path = example.save_file(temp_dir, SerializableFileExtensions.JSON)
        assert os.path.exists(inferred_path)
        assert inferred_path.endswith(".json")
        loaded_example = ExampleModel.load_file(inferred_path)
        assert loaded_example.name == "John Doe"
        assert loaded_example.age == 30
