import os


__all__ = [
    "is_file_name",
    "is_directory_name",
]


def is_file_name(path: str) -> bool:
    """
    Check if the path has an extension and is not a directory.

    :param path: The path to check.
    :type path: str
    :return: True if the path is a file naming convention.
    """

    _, ext = os.path.splitext(path)

    return bool(ext) and not path.endswith(os.path.sep)


def is_directory_name(path: str) -> bool:
    """
    Check if the path does not have an extension and is a directory.

    :param path: The path to check.
    :type path: str
    :return: True if the path is a directory naming convention.
    """
    _, ext = os.path.splitext(path)
    return not ext or path.endswith(os.path.sep)
