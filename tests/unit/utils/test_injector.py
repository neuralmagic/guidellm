from pathlib import Path

import pytest
from pydantic import BaseModel

from guidellm.config import settings
from guidellm.utils.injector import create_report, inject_data


class ExampleModel(BaseModel):
    name: str
    version: str


@pytest.mark.smoke()
def test_inject_data():
    model = ExampleModel(name="Example App", version="1.0.0")
    html = "window.report_data = {};"
    expected_html = 'window.report_data = {"name":"Example App","version":"1.0.0"};'

    result = inject_data(
        model,
        html,
        settings.report_generation.report_html_match,
        settings.report_generation.report_html_placeholder,
    )
    assert result == expected_html


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
