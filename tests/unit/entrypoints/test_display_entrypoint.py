import os
import unittest
from pathlib import Path

import pytest

from guidellm.benchmark import display_benchmarks_report


@pytest.fixture
def get_test_asset_dir():
    def _() -> Path:
        return Path(__file__).parent / "assets"

    return _


def test_display_entrypoint_json(capfd, get_test_asset_dir):
    generic_test_display_entrypoint(
        "benchmarks_stripped.json",
        capfd,
        get_test_asset_dir,
    )


def test_display_entrypoint_yaml(capfd, get_test_asset_dir):
    generic_test_display_entrypoint(
        "benchmarks_stripped.yaml",
        capfd,
        get_test_asset_dir,
    )


def generic_test_display_entrypoint(filename, capfd, get_test_asset_dir):
    os.environ["COLUMNS"] = "120"  # CLI output depends on terminal width.
    asset_dir = get_test_asset_dir()
    display_benchmarks_report(asset_dir / filename)
    out, err = capfd.readouterr()
    expected_output_path = asset_dir / "benchmarks_stripped_output.txt"
    with expected_output_path.open(encoding="utf_8") as file:
        expected_output = file.read()
    assert out == expected_output


if __name__ == "__main__":
    unittest.main()
