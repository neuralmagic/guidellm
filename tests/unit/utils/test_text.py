from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from guidellm.utils.text import (
    clean_text,
    filter_text,
    is_path,
    is_path_like,
    is_url,
    load_text,
    load_text_lines,
    parse_text_objects,
    split_lines_by_punctuation,
    split_text,
)


@pytest.fixture()
def sample_text():
    return "This is a sample text.\nThis is another line!"


@pytest.fixture()
def sample_dict_data():
    return [{"text": "line 1"}, {"text": "line 2"}, {"text": "line 3"}]


@pytest.fixture()
def sample_csv_data():
    return "text\nline 1\nline 2\nline 3"


@pytest.fixture()
def sample_jsonl_data():
    return '{"text": "line 1"}\n{"text": "line 2"}\n{"text": "line 3"}'


@pytest.fixture()
def sample_yaml_data():
    return """
    text:
      - line 1
      - line 2
      - line 3
    """


@pytest.fixture()
def mock_response():
    response = requests.Response()
    response.status_code = 200
    response._content = b"Mock content"
    return response


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("text", "start", "end", "expected"),
    [
        ("hello world", "hello", "world", "hello "),
        ("hello world", "world", None, "world"),
        ("hello world", None, "hello", ""),
        ("hello world", None, None, "hello world"),
    ],
)
def test_filter_text(text, start, end, expected):
    assert filter_text(text, start, end) == expected


@pytest.mark.smoke()
@pytest.mark.parametrize(
    (
        "text",
        "fix_encoding",
        "clean_whitespace",
        "remove_empty_lines",
        "force_new_line_punctuation",
        "expected",
    ),
    [
        (
            "This is\ta test.\n   New line.",
            True,
            True,
            False,
            False,
            "This is a test.\nNew line.",
        ),
        (
            "This is\ta test.\n   New line.",
            True,
            True,
            True,
            False,
            "This is a test.\nNew line.",
        ),
        (
            "This is a test. New line.",
            True,
            False,
            False,
            True,
            "This is a test.\nNew line.",
        ),
    ],
)
def test_clean_text(
    text,
    fix_encoding,
    clean_whitespace,
    remove_empty_lines,
    force_new_line_punctuation,
    expected,
):
    assert (
        clean_text(
            text,
            fix_encoding,
            clean_whitespace,
            remove_empty_lines,
            force_new_line_punctuation,
        )
        == expected
    )


@pytest.mark.smoke()
def test_split_lines_by_punctuation(sample_text):
    expected = ["This is a sample text.", "This is another line!"]
    assert split_lines_by_punctuation(sample_text) == expected


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com", True),
        ("ftp://example.com", True),
        ("not a url", False),
    ],
)
def test_is_url(url, expected):
    assert is_url(url) == expected


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (str(Path(__file__)), True),
        ("/non/existent/path", False),
    ],
)
def test_is_path(path, expected):
    assert is_path(path) == expected


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("path", "enforce_file", "expected"),
    [
        (str(Path(__file__)), True, True),
        ("/non/existent/path", False, True),
        ("https://example.com", False, False),
    ],
)
def test_is_path_like(path, enforce_file, expected):
    assert is_path_like(path, enforce_file) == expected


@pytest.mark.smoke()
def test_split_text(sample_text):
    words, separators, new_lines = split_text(sample_text)
    assert words == [
        "This",
        "is",
        "a",
        "sample",
        "text.",
        "This",
        "is",
        "another",
        "line!",
    ]
    assert separators == [" ", " ", " ", " ", "\n", " ", " ", " ", " "]
    assert new_lines == [0, 5]


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("data", "format_", "expected"),
    [
        ("text\nline 1\nline 2", "csv", [{"text": "line 1"}, {"text": "line 2"}]),
        (
            '{"text": "line 1"}\n{"text": "line 2"}',
            "jsonl",
            [{"text": "line 1"}, {"text": "line 2"}],
        ),
    ],
)
def test_parse_text_objects(data, format_, expected):
    assert parse_text_objects(data, format_) == expected


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("https://example.com", "Mock content"),
        (str(Path(__file__)), Path(__file__).read_text()),
    ],
)
def test_load_text(data, expected, mock_response):
    with patch("requests.get", return_value=mock_response):
        assert load_text(data) == expected


@pytest.mark.regression()
def test_load_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_text("/non/existent/file.txt")


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("data", "format_", "filters", "expected"),
    [
        ("text\nline 1\nline 2", "csv", None, ["line 1", "line 2"]),
        ('{"text": "line 1"}\n{"text": "line 2"}', "jsonl", None, ["line 1", "line 2"]),
        ("text\nline 1\nline 2", "txt", None, ["text", "line 1", "line 2"]),
    ],
)
def test_load_text_lines(data, format_, filters, expected):
    assert load_text_lines(data, format_=format_, filters=filters) == expected


@pytest.mark.regression()
def test_load_text_lines_invalid_data():
    with pytest.raises(ValueError):
        load_text_lines(123)  # type: ignore


@pytest.mark.regression()
def test_parse_text_objects_invalid_format():
    with pytest.raises(ValueError):
        parse_text_objects("text", format_="unsupported")


@pytest.mark.regression()
def test_parse_text_objects_invalid_data():
    with pytest.raises(ValueError):
        parse_text_objects(123)  # type: ignore


@pytest.mark.regression()
@pytest.mark.parametrize(
    ("data", "format_", "filters", "expected"),
    [
        (
            "text\nline 1\nline 2\n",
            "csv",
            ["text"],
            ["line 1", "line 2"],
        ),
    ],
)
def test_load_text_lines_with_filters(data, format_, filters, expected):
    assert load_text_lines(data, format_=format_, filters=filters) == expected


@pytest.mark.regression()
def test_is_path_with_symlink(tmp_path):
    # Create a symlink to a temporary file
    target_file = tmp_path / "target_file.txt"
    target_file.write_text("Sample content")
    symlink_path = tmp_path / "symlink"
    symlink_path.symlink_to(target_file)

    assert is_path(str(symlink_path)) is True


@pytest.mark.regression()
def test_is_path_like_with_symlink(tmp_path):
    # Create a symlink to a temporary file
    target_file = tmp_path / "target_file.txt"
    target_file.write_text("Sample content")
    symlink_path = tmp_path / "symlink.file"
    symlink_path.symlink_to(target_file)

    assert is_path_like(str(symlink_path), enforce_file=True) is True


@pytest.mark.regression()
def test_load_text_lines_empty():
    # Test loading text lines from an empty string
    assert load_text_lines("") == []


@pytest.mark.regression()
def test_split_text_with_empty_string():
    words, separators, new_lines = split_text("")
    assert words == []
    assert separators == []
    assert new_lines == []


@pytest.mark.regression()
def test_split_lines_by_punctuation_with_no_punctuation():
    text = "This is a test without punctuation"
    assert split_lines_by_punctuation(text) == [text]


@pytest.mark.regression()
def test_is_path_invalid_type():
    assert not is_path(None)
    assert not is_path(123)
    assert not is_path(["not", "a", "path"])


@pytest.mark.regression()
def test_is_path_like_invalid_type():
    assert not is_path_like(None, enforce_file=False)
    assert not is_path_like(123, enforce_file=True)
    assert not is_path_like(["not", "a", "path"], enforce_file=False)


@pytest.mark.regression()
def test_load_text_invalid_url():
    with pytest.raises(requests.ConnectionError):
        load_text("http://invalid.url")


@pytest.mark.regression()
def test_parse_text_objects_empty_csv():
    assert parse_text_objects("text\n", "csv") == []


@pytest.mark.regression()
def test_parse_text_objects_empty_jsonl():
    assert parse_text_objects("", "jsonl") == []


@pytest.mark.regression()
def test_parse_text_objects_invalid_jsonl():
    with pytest.raises(ValueError):
        parse_text_objects("{invalid_json}", "jsonl")


@pytest.mark.regression()
def test_parse_text_objects_empty_yaml():
    assert parse_text_objects("", "yaml") == []


@pytest.mark.regression()
def test_clean_text_with_unicode():
    text = "This is a test with unicode: \u2013 \u2014"
    cleaned_text = clean_text(text, fix_encoding=True, clean_whitespace=True)
    assert cleaned_text == "This is a test with unicode: – —"


@pytest.mark.regression()
def test_split_lines_by_punctuation_with_multiple_punctuations():
    text = "First sentence. Second sentence? Third sentence!"
    expected = ["First sentence.", "Second sentence?", "Third sentence!"]
    assert split_lines_by_punctuation(text) == expected


@pytest.mark.regression()
def test_is_url_empty_string():
    assert not is_url("")


@pytest.mark.regression()
def test_load_text_invalid_data():
    with pytest.raises(TypeError):
        load_text(123)  # type: ignore


@pytest.mark.regression()
def test_load_text_lines_empty_format():
    data = "text\nline 1\nline 2"
    assert load_text_lines(data, format_="") == ["text", "line 1", "line 2"]


@pytest.mark.regression()
def test_split_text_with_mixed_separators():
    text = "This\tis a test\nwith mixed separators."
    words, separators, new_lines = split_text(text)
    assert words == ["This", "is", "a", "test", "with", "mixed", "separators."]
    assert separators == ["\t", " ", " ", "\n", " ", " ", " "]
    assert new_lines == [0, 4]
