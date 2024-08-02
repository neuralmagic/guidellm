__all__ = [
    "PREFERRED_DATA_COLUMNS",
    "PREFERRED_DATA_SPLITS",
    "REPORT_HTML_MATCH",
    "REPORT_HTML_PLACEHOLDER",
]


PREFERRED_DATA_COLUMNS = [
    "prompt",
    "instruction",
    "input",
    "inputs",
    "question",
    "context",
    "text",
    "content",
    "body",
    "data",
]

PREFERRED_DATA_SPLITS = ["test", "validation", "train"]

REPORT_HTML_MATCH = "window.report_data = {};"

REPORT_HTML_PLACEHOLDER = "{}"
