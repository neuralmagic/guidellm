from pathlib import Path
from typing import Union

import requests
from pydantic import BaseModel

from guidellm.config import settings
from guidellm.utils.constants import (
    REPORT_HTML_MATCH,
    REPORT_HTML_PLACEHOLDER,
)

__all__ = ["create_report", "inject_data", "load_html_file"]


def create_report(model: BaseModel, output_path: Union[str, Path]) -> Path:
    """
    Creates a report from the model and saves it to the output path.

    :param model: the model to serialize and inject
    :type model: BaseModel
    :param output_path: the path, either a file or a directory,
        to save the report to. If a directory, the report will be saved
        as "report.html" inside of the directory.
    :type output_path: str
    :return: the path to the saved report
    :rtype: str
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    html_content = load_html_file(settings.report_generation.source)
    report_content = inject_data(
        model, html_content, REPORT_HTML_MATCH, REPORT_HTML_PLACEHOLDER
    )

    if not output_path.suffix:
        # assume directory, save as report.html
        output_path = output_path / "report.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content)

    return output_path


def inject_data(
    model: BaseModel,
    html: str,
    match: str,
    placeholder: str,
) -> str:
    """
    Injects the data from the model into the HTML while replacing the placeholder.

    :param model: the model to serialize and inject
    :type model: BaseModel
    :param html: the html to inject the data into
    :type html: str
    :param match: the string to match in the html to find the placeholder
    :type match: str
    :param placeholder: the placeholder to replace with the model data
        inside of the placeholder
    :type placeholder: str
    :return: the html with the model data injected
    :rtype: str
    """
    model_str = model.json()
    inject_str = match.replace(placeholder, model_str)

    return html.replace(match, inject_str)


def load_html_file(path_or_url: str) -> str:
    """
    Load an HTML file from a path or URL

    :param path_or_url: the path or URL to load the HTML file from
    :type path_or_url: str
    :return: the HTML content
    :rtype: str
    """
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url, timeout=settings.request_timeout)
        response.raise_for_status()

        return response.text

    path = Path(path_or_url)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_or_url}")

    return path.read_text()
