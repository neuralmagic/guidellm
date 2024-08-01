from pydantic import BaseModel
import requests
import os
from pathlib import Path

from guidellm.utils.constants import (
    REPORT_HTML_PLACEHOLDER,
    REPORT_HTML_MATCH,
    STANDARD_REQUEST_TIMEOUT,
)
from guidellm.config import settings


__all__ = ["create_report", "inject_data", "load_html_file"]


def create_report(model: BaseModel, output_path: str) -> str:
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
    html_content = load_html_file(settings.report_generation.source)
    report_content = inject_data(
        model, html_content, REPORT_HTML_MATCH, REPORT_HTML_PLACEHOLDER
    )

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "report.html")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with Path(output_path).open("w") as file:
        file.write(report_content)

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
        response = requests.get(path_or_url, timeout=STANDARD_REQUEST_TIMEOUT)
        response.raise_for_status()

        return response.text

    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"File not found: {path_or_url}")

    with Path(path_or_url).open("r") as file:
        return file.read()
