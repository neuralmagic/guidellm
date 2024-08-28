import csv
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import ftfy
import requests
import yaml
from loguru import logger

from guidellm.config import settings

__all__ = [
    "clean_text",
    "filter_text",
    "is_path",
    "is_path_like",
    "is_url",
    "load_text",
    "load_text_lines",
    "parse_text_objects",
    "split_lines_by_punctuation",
    "split_text",
    "random_strings",
]


NAME_TITLES = [
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Jr.",
    "Sr.",
    "St.",
    "Lt.",
    "Col.",
    "Gen.",
    "Rep.",
    "Sen.",
    "Gov.",
    "Pres.",
]
SENTENCE_REGEX = r'[^.!?]*[.!?]["\']?\s*(?=[A-Z])'
MAX_EXTENSION_LENGTH = 8
MAX_PATH_LENGTH = 4096
EXTENSION_TYPES = {
    "csv": "csv",
    "jsonl": "jsonl",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "txt": "txt",
    "text": "txt",
}


def filter_text(
    text: str,
    filter_start: Optional[Union[str, int]] = None,
    filter_end: Optional[Union[str, int]] = None,
) -> str:
    """
    Filter text by start and end strings or indices

    :param text: the text to filter
    :param filter_start: the start string or index to filter from
    :param filter_end: the end string or index to filter to
    :return: the filtered text
    """
    filter_start_index = -1
    filter_end_index = -1

    if filter_start and isinstance(filter_start, str):
        filter_start_index = text.index(filter_start)
    elif filter_start:
        if not isinstance(filter_start, int):
            raise ValueError(f"Invalid filter start index: {filter_start}")
        filter_start_index = filter_start

    if filter_end and isinstance(filter_end, str):
        filter_end_index = text.index(filter_end)
    elif filter_end:
        if not isinstance(filter_end, int):
            raise ValueError(f"Invalid filter end index: {filter_end}")
        filter_end_index = filter_end

    if filter_start_index > -1:
        text = text[filter_start_index:]
    if filter_end_index > -1:
        text = text[:filter_end_index]

    return text


def clean_text(
    text: str,
    fix_encoding: bool = True,
    clean_whitespace: bool = False,
    remove_empty_lines: bool = False,
    force_new_line_punctuation: bool = False,
) -> str:
    """
    Clean text by fixing encoding, cleaning whitespace, removing empty lines,
    and forcing new line punctuation

    :param text: the text to clean
    :param fix_encoding: True to fix the encoding of the text, False to leave as is
    :param clean_whitespace: True to clean the whitespace in the text
        (remove extra spaces, tabs, etc), False to leave as is
    :param remove_empty_lines: True to remove empty lines from the text
        (lines with only whitespace), False to leave as is
    :param force_new_line_punctuation: True to force new lines at punctuation
        (line ends in a period, exclamation point, or question mark),
        False to leave as is
    :return: The cleaned text
    """

    if fix_encoding:
        text = ftfy.fix_text(text)

    if clean_whitespace:
        text = "\n".join(
            [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        )

    if remove_empty_lines:
        text = "\n".join([line for line in text.splitlines() if line.strip()])

    if force_new_line_punctuation:
        # first remove any existing new lines
        text = " ".join(line for line in text.splitlines() if line.strip())
        lines = split_lines_by_punctuation(text)
        text = "\n".join(lines)

    return text


def split_lines_by_punctuation(text: str) -> List[str]:
    """
    Split text into lines based on punctuation

    :param text: the text to split
    :return: the list of lines
    """

    lines = []
    current_line = ""
    skip_next = False

    for index, char in enumerate(text):
        if skip_next:
            skip_next = False
            continue

        current_line += char

        if char not in [".", "!", "?"]:
            # must match end of sentence punctuation
            continue

        # if this is the character for a title, don't split
        if any(current_line.endswith(title) for title in NAME_TITLES):
            continue

        char_next_1 = text[index + 1] if index + 1 < len(text) else None
        char_next_2 = text[index + 2] if index + 2 < len(text) else None
        char_next_3 = text[index + 3] if index + 3 < len(text) else None

        next_is_space = char_next_1 and char_next_1.isspace()
        next_is_quote_and_space = char_next_1 in ["'", '"'] and char_next_2 == " "

        # next character must be a space or a quote, otherwise skip
        if not next_is_space and not next_is_quote_and_space:
            continue

        # after this, next character must be an upper case letter
        upper_char = char_next_3 if next_is_quote_and_space else char_next_2
        next_is_upper = upper_char and (
            upper_char.isupper() or upper_char in ["'", '"']
        )

        if not next_is_upper:
            continue

        # if next char is a quote, add it and skip next
        if next_is_quote_and_space:
            current_line += text[index + 1]
            skip_next = True

        lines.append(current_line.strip())
        current_line = ""

    if current_line:
        lines.append(current_line.strip())

    return lines


def is_url(url: str) -> bool:
    """
    Check if a string is a URL

    :param url: the string to check
    :return: True if the string is a URL, False if not
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:  # noqa: BLE001
        return False


def is_path(path: Any) -> bool:
    """
    Check if a string is a path

    :param path: the string to check
    :return: True if the string is a path, False if not
    """
    if not isinstance(path, (str, Path)):
        return False

    if isinstance(path, str):
        path = Path(path)

    return path.exists()


def is_path_like(path: Any, enforce_file: bool = False) -> bool:
    """
    Check if a string has a path like structure where it doesn't need to exist

    :param path: the string to check
    :param enforce_file: True if the path should be a file, False if not
    :return: True if the string is path like, False if not
    """
    # if path isn't a str or Path, it's not a path
    if not isinstance(path, (str, Path)):
        return False

    if isinstance(path, Path):
        path = str(path)

    # if text is too long, it's not a path (4096 for most linux setups)
    if len(path) > MAX_PATH_LENGTH:
        return False

    # if it starts with a URL scheme, it's not a path
    if path.startswith(("http", "ftp")):
        return False

    test_path = Path(path)

    # if it's supposed to be a file and there's no extension or
    # the extension is too long, it's not a path
    return not enforce_file or (
        bool(test_path.suffix) and len(test_path.suffix) <= MAX_EXTENSION_LENGTH
    )


def split_text(text: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Split text into words / tokens, the white space separators between words,
    and the indices for each new line

    :param text: the text to split
    :return: the words, the white space separators, and the new line indices
    """
    if not text or not text.strip():
        return [], [], []

    text = text.strip()
    tokens = []  # type: List[str]
    separators = []  # type: List[str]
    new_lines = [0]
    buffer = text[0]
    is_token = not text[0].isspace()

    for char in text[1:]:
        char_whitespace = char.isspace()

        if char == "\n":
            new_lines.append(len(tokens) + 1)

        if char_whitespace and is_token:
            tokens.append(buffer)
            buffer = char
            is_token = False
        elif char_whitespace:
            buffer += char
        elif not char_whitespace and not is_token:
            separators.append(buffer)
            buffer = char
            is_token = True
        else:
            buffer += char

    if buffer and is_token:
        tokens.append(buffer)
        separators.append(" ")
    elif buffer:
        separators.append(buffer)

    return tokens, separators, new_lines


def load_text(data: Union[str, Path], encoding: Optional[str] = None) -> str:
    """
    Load an HTML file from a path or URL

    :param data: the path or URL to load the HTML file from
    :type data: Union[str, Path]
    :param encoding: the encoding to use when reading the file
    :type encoding: str
    :return: the HTML content
    :rtype: str
    """
    logger.debug("Loading text: {}", data)

    if not data:
        return ""

    # check URLs
    if isinstance(data, str) and data.startswith("http"):
        response = requests.get(data, timeout=settings.request_timeout)
        response.raise_for_status()
        return response.text

    # check raw text
    if isinstance(data, str) and not is_path_like(data, enforce_file=True):
        return data

    # assume local file
    if not isinstance(data, Path):
        data = Path(data)

    if not data.exists():
        raise FileNotFoundError(f"File not found: {data}")

    if not data.is_file():
        raise IsADirectoryError(f"Path is a directory: {data}")

    return data.read_text(encoding=encoding)


def parse_text_objects(data: str, format_: str = "txt") -> List[Dict]:
    """
    Parse text data into a list of dictionaries based on the format given
    (csv, jsonl, json, yaml, txt).

    :param data: the text data to parse
    :param format_: the format of the data to parse:
        'csv', 'jsonl', 'json', 'yaml', 'txt'
    :return: the list of dictionaries parsed from the data, if text
        then each line is a dictionary with a single key 'text'
    """
    if not isinstance(data, str):
        raise ValueError(f"Unsupported data given of type: {type(data)}")

    if format_ == "csv":
        reader = csv.DictReader(data.splitlines())
        columns = reader.fieldnames
        return [{col: row[col] for col in columns} for row in reader]  # type: ignore # noqa: PGH003

    if format_ == "jsonl":
        return [json.loads(line) for line in data.splitlines() if line]

    if format_ in ("json", "yaml"):
        data = json.loads(data) if format_ == "json" else yaml.safe_load(data)

        if not data:
            return []

        if isinstance(data, dict) and len(data) == 1:
            logger.debug("Getting first value from JSON/YAML object: {}", data)
            data = list(data.values())[0]
        elif isinstance(data, dict):
            logger.debug("Converting JSON/YAML object to list: {}", data)
            data = list(data.values())

        if not isinstance(data, list) or not isinstance(data[0], dict):
            raise ValueError(f"Unsupported data structure given: {data}")

        return data

    if format_ == "txt":
        return [{"text": line} for line in data.splitlines() if line]

    raise ValueError(f"Unsupported format given: {format_}")


def load_text_lines(
    data: Union[str, Path, List[Dict]],
    format_: Optional[str] = None,
    filters: Optional[List[str]] = None,
    encoding: Optional[str] = None,
) -> List[str]:
    """
    Load text lines from a file or data object with optional filtering and formatting.


    :param data: the data to load the text lines from
    :param format_: the format of the data to load, if not provided will be inferred.
        Supported formats: 'csv', 'jsonl', 'json', 'yaml', 'txt'
    :param filters: the keys to filter the data by when loading in order of preference.
        If not provided, will use the first key in the data object.
    :param encoding: the encoding to use when reading the file
    :return: the list of text lines
    """
    logger.debug(
        "Loading text lines with format {}, filters {}, encoding {} for data: {}",
        format_,
        filters,
        encoding,
        data,
    )

    if not data:
        return []

    if not format_ and isinstance(data, (str, Path)) and "." in str(data):
        extension = str(data).split(".")[-1]
        format_ = EXTENSION_TYPES.get(extension, "txt")
    elif not format_:
        format_ = "txt"

    # load the data if it's a path or URL
    if isinstance(data, Path) or (isinstance(data, str) and data.startswith("http")):
        data = load_text(data, encoding=encoding)
        data = clean_text(data)

    # parse the data into a list of dictionaries based on the format
    if isinstance(data, str):
        data = parse_text_objects(data, format_)

    if not isinstance(data, list):
        raise ValueError(f"Unsupported data given of type: {type(data)}")

    if not isinstance(data[0], dict):
        raise ValueError(f"Unsupported data item type given: {type(data[0])}")

    # grab the first available filter key to use if preference order as provided
    filter_ = list(data[0].keys())[0]
    for filt in filters or []:
        if filt not in data[0]:
            continue

        filter_ = filt
        break

    # extract the lines from the data
    return [row[filter_] for row in data] if filter_ else [str(row) for row in data]


def random_strings(
    min_chars: int, max_chars: int, n: int = 0, dataset: Optional[str] = None
) -> Generator[str, None, None]:
    """Yield random strings.

    :param min: the min number of output characters
    :param max: the max number of output characters
    :param n: the number of outputs. If `0` -> works for infinite
    :param dataset: represents allowed characters for the operation
    """

    characters: str = dataset or string.printable

    if n < 0:
        raise ValueError("'n' must be >= '0'")
    elif n == 0:
        while True:
            yield "".join(
                random.choice(characters)
                for _ in range(random.randint(min_chars, max_chars))
            )
    else:
        for _ in range(n):
            yield "".join(
                random.choice(characters)
                for _ in range(random.randint(min_chars, max_chars))
            )
