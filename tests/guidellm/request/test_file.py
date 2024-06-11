import pytest
from unittest.mock import patch, mock_open
from guidellm.request import FileRequestGenerator


@pytest.fixture
def mock_tokenizer():
    with patch(
        "guidellm.request.file_request_generator.AutoTokenizer.from_pretrained"
    ) as mock:
        mock_instance = mock.return_value
        mock_instance.return_value = {"input_ids": [0, 1, 2, 3, 4]}
        yield mock_instance


def test_file_request_generator_txt():
    file_content = "Sample text 1\nSample text 2\n"
    with patch("builtins.open", mock_open(read_data=file_content)):
        generator = FileRequestGenerator(file_path="dummy.txt", mode="sync")

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"


def test_file_request_generator_csv():
    file_content = "text\nSample text 1\nSample text 2\n"
    with patch("builtins.open", mock_open(read_data=file_content)):
        generator = FileRequestGenerator(file_path="dummy.csv", mode="sync")

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"


def test_file_request_generator_jsonl():
    file_content = '{"text": "Sample text 1"}\n{"text": "Sample text 2"}\n'
    with patch("builtins.open", mock_open(read_data=file_content)):
        generator = FileRequestGenerator(file_path="dummy.jsonl", mode="sync")

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"


def test_file_request_generator_json():
    file_content = '[{"text": "Sample text 1"}, {"text": "Sample text 2"}]'
    with patch("builtins.open", mock_open(read_data=file_content)):
        generator = FileRequestGenerator(file_path="dummy.json", mode="sync")

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"


def test_file_request_generator_with_tokenizer(mock_tokenizer):
    file_content = "Sample text 1\nSample text 2\n"
    with patch("builtins.open", mock_open(read_data=file_content)):
        generator = FileRequestGenerator(
            file_path="dummy.txt", tokenizer="mock_tokenizer", mode="sync"
        )

    items = []
    for _ in range(2):
        items.append(generator.create_item())

    assert len(items) == 2
    assert items[0].prompt == "Sample text 1"
    assert items[1].prompt == "Sample text 2"
    assert items[0].token_count == 5
    assert items[1].token_count == 5
