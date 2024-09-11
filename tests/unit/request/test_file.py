import tempfile
from pathlib import Path

import pytest

from guidellm.core.request import TextGenerationRequest
from guidellm.request.file import FileRequestGenerator


@pytest.mark.smoke()
def test_file_request_generator_constructor(mock_auto_tokenizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "example.txt"
        file_path.write_text("This is a test.\nThis is another test.")
        generator = FileRequestGenerator(file_path, tokenizer="mock-tokenizer")
        assert generator._path == file_path
        assert generator._data == ["This is a test.", "This is another test."]
        assert generator._iterator is not None


@pytest.mark.smoke()
def test_file_request_generator_create_item(mock_auto_tokenizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "example.txt"
        file_path.write_text("This is a test.\nThis is another test.")
        generator = FileRequestGenerator(
            file_path, tokenizer="mock-tokenizer", mode="sync"
        )
        request = generator.create_item()
        assert isinstance(request, TextGenerationRequest)
        assert request.prompt == "This is a test."


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("file_extension", "file_content"),
    [
        ("txt", "Test content 1.\nTest content 2.\nTest content 3.\n"),
        (
            "csv",
            "text,label,extra\n"
            "Test content 1.,1,extra 1\n"
            "Test content 2.,2,extra 2\n"
            "Test content 3.,3,extra 3\n",
        ),
        (
            "jsonl",
            '{"text": "Test content 1."}\n'
            '{"text": "Test content 2."}\n'
            '{"text": "Test content 3."}\n',
        ),
        (
            "csv",
            "prompt,text,extra\n"
            "Test content 1., text 1, extra 1\n"
            "Test content 2., text 2, extra 2\n"
            "Test content 3., text 3, extra 3\n",
        ),
        (
            "json",
            '[{"text": "Test content 1."}, '
            '{"text": "Test content 2."}, '
            '{"text": "Test content 3."}]\n',
        ),
        (
            "json",
            '{"object_1": {"text": "Test content 1."}, '
            '"object_2": {"text": "Test content 2."}, '
            '"object_3": {"text": "Test content 3."}}\n',
        ),
        (
            "yaml",
            "items:\n"
            "   - text: Test content 1.\n"
            "   - text: Test content 2.\n"
            "   - text: Test content 3.\n",
        ),
        (
            "yaml",
            "object_1:\n  text: Test content 1.\n"
            "object_2:\n  text: Test content 2.\n"
            "object_3:\n  text: Test content 3.\n",
        ),
    ],
)
def test_file_request_generator_file_types_lifecycle(
    mock_auto_tokenizer, file_extension, file_content
):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / f"example.{file_extension}"
        file_path.write_text(file_content)
        generator = FileRequestGenerator(file_path, tokenizer="mock-tokenizer")

        for index, request in enumerate(generator):
            assert isinstance(request, TextGenerationRequest)
            assert request.prompt == f"Test content {index + 1}."
            assert request.prompt_token_count == 3

            if index == 2:
                break


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("file_extension", "file_content"),
    [
        ("txt", "Test content 1.\nTest content 2.\nTest content 3.\n"),
        (
            "csv",
            "text,label,extra\n"
            "Test content 1.,1,extra 1\n"
            "Test content 2.,2,extra 2\n"
            "Test content 3.,3,extra 3\n",
        ),
        (
            "jsonl",
            '{"text": "Test content 1."}\n'
            '{"text": "Test content 2."}\n'
            '{"text": "Test content 3."}\n',
        ),
        (
            "csv",
            "prompt,text,extra\n"
            "Test content 1., text 1, extra 1\n"
            "Test content 2., text 2, extra 2\n"
            "Test content 3., text 3, extra 3\n",
        ),
        (
            "json",
            '[{"text": "Test content 1."}, '
            '{"text": "Test content 2."}, '
            '{"text": "Test content 3."}]\n',
        ),
        (
            "json",
            '{"object_1": {"text": "Test content 1."}, '
            '"object_2": {"text": "Test content 2."}, '
            '"object_3": {"text": "Test content 3."}}\n',
        ),
        (
            "yaml",
            "items:\n"
            "   - text: Test content 1.\n"
            "   - text: Test content 2.\n"
            "   - text: Test content 3.\n",
        ),
        (
            "yaml",
            "object_1:\n  text: Test content 1.\n"
            "object_2:\n  text: Test content 2.\n"
            "object_3:\n  text: Test content 3.\n",
        ),
    ],
)
def test_file_request_generator_len(mock_auto_tokenizer, file_extension, file_content):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / f"example.{file_extension}"
        file_path.write_text(file_content)
        generator = FileRequestGenerator(file_path, tokenizer="mock-tokenizer")

        assert len(generator) == 3
