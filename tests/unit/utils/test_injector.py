from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import requests
from guidellm.config import settings
from guidellm.utils.constants import (
    REPORT_HTML_MATCH,
    REPORT_HTML_PLACEHOLDER,
)
from guidellm.utils.injector import create_report, inject_data, load_html_file
from pydantic import BaseModel


class ExampleModel(BaseModel):
    name: str
    version: str


@pytest.mark.smoke()
def test_inject_data():
    model = ExampleModel(name="Example App", version="1.0.0")
    html = "window.report_data = {};"
    expected_html = 'window.report_data = {"name":"Example App","version":"1.0.0"};'

    result = inject_data(model, html, REPORT_HTML_MATCH, REPORT_HTML_PLACEHOLDER)
    assert result == expected_html


@pytest.mark.smoke()
def test_load_html_file_from_url(requests_mock):
    url = "http://example.com/report.html"
    mock_content = "<html>Sample Report</html>"
    requests_mock.get(url, text=mock_content)

    result = load_html_file(url)
    assert result == mock_content


@pytest.mark.sanity()
def test_load_html_file_from_invalid_url(requests_mock):
    url = "http://example.com/404.html"
    requests_mock.get(url, status_code=404)

    with pytest.raises(requests.exceptions.HTTPError):
        load_html_file(url)


@pytest.mark.smoke()
def test_load_html_file_from_path():
    path = "sample_report.html"
    mock_content = "<html>Sample Report</html>"

    with patch("pathlib.Path.open", mock_open(read_data=mock_content)), patch(
        "pathlib.Path.exists", return_value=True
    ):
        result = load_html_file(path)

    assert result == mock_content


@pytest.mark.sanity()
def test_load_html_file_from_invalid_path():
    path = "invalid_report.html"

    with pytest.raises(FileNotFoundError):
        load_html_file(path)


@pytest.mark.smoke()
def test_create_report_to_file(tmpdir):
    model = ExampleModel(name="Example App", version="1.0.0")
    html_content = "window.report_data = {};"
    expected_html_content = (
        'window.report_data = {"name":"Example App","version":"1.0.0"};'
    )

    mock_html_path = tmpdir.join("template.html")
    mock_html_path.write(html_content)
    settings.report_generation.source = str(mock_html_path)

    output_path = tmpdir.join("output.html")
    result_path = create_report(model, str(output_path))
    result_content = result_path.read_text()

    assert result_path == output_path
    assert result_content == expected_html_content


@pytest.mark.smoke()
def test_create_report_to_directory(tmpdir):
    model = ExampleModel(name="Example App", version="1.0.0")
    html_content = "window.report_data = {};"
    expected_html_content = (
        'window.report_data = {"name":"Example App","version":"1.0.0"};'
    )

    mock_html_path = tmpdir.join("template.html")
    mock_html_path.write(html_content)
    settings.report_generation.source = str(mock_html_path)

    output_dir = tmpdir.mkdir("output_dir")
    output_path = Path(output_dir) / "report.html"
    result_path = create_report(model, str(output_dir))

    with Path(result_path).open("r") as file:
        result_content = file.read()

    assert result_path == output_path
    assert result_content == expected_html_content
