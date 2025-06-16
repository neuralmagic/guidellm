from pathlib import Path
from typing import Union

from guidellm.config import settings
from guidellm.utils.text import load_text

__all__ = ["create_report", "inject_data"]


def create_report(js_data: dict, output_path: Union[str, Path]) -> Path:
    """
    Creates a report from the dictionary and saves it to the output path.

    :param js_data: dict with match str and json data to inject
    :type js_data: dict
    :param output_path: the path, either a file or a directory,
        to save the report to. If a directory, the report will be saved
        as "report.html" inside of the directory.
    :type output_path: str
    :return: the path to the saved report
    :rtype: str
    """
    
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if output_path.is_dir():
        output_path = output_path / "report.html"

    html_content = load_text(settings.report_generation.source)
    report_content = inject_data(
        js_data,
        html_content,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content)
    print(f'Report saved to {output_path}')
    return output_path

def inject_data(
    js_data: dict,
    html: str,
) -> str:
    """
    Injects the json data into the HTML while replacing the placeholder.

    :param js_data: the json data to inject
    :type js_data: dict
    :param html: the html to inject the data into
    :type html: str
    :return: the html with the json data injected
    :rtype: str
    """
    for placeholder, script in js_data.items():
        html = html.replace(placeholder, script)
    return html