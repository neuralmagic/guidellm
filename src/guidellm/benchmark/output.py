import csv
import json
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import humps  # type: ignore[import-not-found]
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from guidellm.benchmark.benchmark import (
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
    GenerativeMetrics,
)
from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    SweepProfile,
    ThroughputProfile,
)
from guidellm.config import settings
from guidellm.objects import (
    DistributionSummary,
    StatusDistributionSummary,
)
from guidellm.presentation import UIDataBuilder
from guidellm.presentation.injector import create_report
from guidellm.scheduler import strategy_display_str
from guidellm.utils import Colors, split_text_list_by_length

__all__ = [
    "GenerativeBenchmarkerCSV",
    "GenerativeBenchmarkerConsole",
    "GenerativeBenchmarkerHTML",
    "GenerativeBenchmarkerOutput",
]


class GenerativeBenchmarkerOutput(ABC):
    @abstractmethod
    async def finalize(self, report: GenerativeBenchmarksReport) -> Any: ...


class GenerativeBenchmarkerConsole(GenerativeBenchmarkerOutput):
    """Console output formatter for benchmark results with rich formatting."""

    def __init__(self):
        """
        Initialize the console output formatter.
        """
        self.console = Console()

    def print_line(self, text: str):
        """
        Print a line of text to the console.

        :param text: The text to print.
        """
        self.console.print(text)

    def print_full_report(self):
        """
        Print a placeholder for the full report.
        This method is called but appears to be intended for a different use case.
        """

    async def finalize(self, report: GenerativeBenchmarksReport):
        """
        Print the complete benchmark report to the console.

        :param report: The completed benchmark report.
        :return: None (console output doesn't save to a file).
        """
        self._print_benchmarks_metadata(report.benchmarks)
        self._print_benchmarks_info(report.benchmarks)
        self._print_benchmarks_stats(report.benchmarks)

    def _print_benchmarks_metadata(self, benchmarks: list[GenerativeBenchmark]):
        start_time = benchmarks[0].run_stats.start_time
        end_time = benchmarks[-1].run_stats.end_time
        duration = end_time - start_time

        self._print_section_header("Benchmarks Metadata")
        self._print_labeled_line("Run id", str(benchmarks[0].run_id))
        self._print_labeled_line("Duration", f"{duration:.1f} seconds")
        self._print_labeled_line("Profile", self._get_profile_str(benchmarks[0]))
        self._print_labeled_line("Scheduler", self._get_scheduler_str(benchmarks[0]))
        self._print_labeled_line("Environment", self._get_env_args_str(benchmarks[0]))
        self._print_labeled_line("Extras", self._get_extras_str(benchmarks[0]))

    def _print_benchmarks_info(self, benchmarks: list[GenerativeBenchmark]):
        sections = {
            "Metadata": (0, 3),
            "Requests Made": (4, 6),
            "Prompt Tok/Req": (7, 9),
            "Output Tok/Req": (10, 12),
            "Prompt Tok Total": (13, 15),
            "Output Tok Total": (16, 18),
        }
        headers = [
            "Benchmark",
            "Start Time",
            "End Time",
            "Duration (s)",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
            "Comp",
            "Inc",
            "Err",
        ]

        rows = []
        for benchmark in benchmarks:
            rows.append(
                [
                    strategy_display_str(benchmark.scheduler["strategy"]),
                    datetime.fromtimestamp(benchmark.start_time).strftime("%H:%M:%S"),
                    datetime.fromtimestamp(benchmark.end_time).strftime("%H:%M:%S"),
                    f"{(benchmark.end_time - benchmark.start_time):.1f}",
                    f"{benchmark.request_totals.successful:.0f}",
                    f"{benchmark.request_totals.incomplete:.0f}",
                    f"{benchmark.request_totals.errored:.0f}",
                    f"{benchmark.metrics.prompt_token_count.successful.mean:.1f}",
                    f"{benchmark.metrics.prompt_token_count.incomplete.mean:.1f}",
                    f"{benchmark.metrics.prompt_token_count.errored.mean:.1f}",
                    f"{benchmark.metrics.output_token_count.successful.mean:.1f}",
                    f"{benchmark.metrics.output_token_count.incomplete.mean:.1f}",
                    f"{benchmark.metrics.output_token_count.errored.mean:.1f}",
                    f"{benchmark.metrics.prompt_token_count.successful.total_sum:.0f}",
                    f"{benchmark.metrics.prompt_token_count.incomplete.total_sum:.0f}",
                    f"{benchmark.metrics.prompt_token_count.errored.total_sum:.0f}",
                    f"{benchmark.metrics.output_token_count.successful.total_sum:.0f}",
                    f"{benchmark.metrics.output_token_count.incomplete.total_sum:.0f}",
                    f"{benchmark.metrics.output_token_count.errored.total_sum:.0f}",
                ]
            )

        self._print_table(headers, rows, "Benchmarks Info", sections)

    def _print_benchmarks_stats(self, benchmarks: list[GenerativeBenchmark]):
        sections = {
            "Metadata": (0, 0),
            "Request Stats": (1, 2),
            "Out Tok/sec": (3, 3),
            "Tot Tok/sec": (4, 4),
            "Req Latency (sec)": (5, 7),
            "TTFT (ms)": (8, 10),
            "ITL (ms)": (11, 13),
            "TPOT (ms)": (14, 16),
        }
        headers = [
            "Benchmark",
            "Per Second",
            "Concurrency",
            "mean",
            "mean",
            "mean",
            "median",
            "p99",
            "mean",
            "median",
            "p99",
            "mean",
            "median",
            "p99",
            "mean",
            "median",
            "p99",
        ]

        rows = []
        for benchmark in benchmarks:
            rows.append(
                [
                    strategy_display_str(benchmark.scheduler["strategy"]),
                    f"{benchmark.metrics.requests_per_second.successful.mean:.2f}",
                    f"{benchmark.metrics.request_concurrency.successful.mean:.2f}",
                    f"{benchmark.metrics.output_tokens_per_second.successful.mean:.1f}",
                    f"{benchmark.metrics.tokens_per_second.successful.mean:.1f}",
                    f"{benchmark.metrics.request_latency.successful.mean:.2f}",
                    f"{benchmark.metrics.request_latency.successful.median:.2f}",
                    f"{benchmark.metrics.request_latency.successful.percentiles.p99:.2f}",
                    f"{benchmark.metrics.time_to_first_token_ms.successful.mean:.1f}",
                    f"{benchmark.metrics.time_to_first_token_ms.successful.median:.1f}",
                    f"{benchmark.metrics.time_to_first_token_ms.successful.percentiles.p99:.1f}",
                    f"{benchmark.metrics.inter_token_latency_ms.successful.mean:.1f}",
                    f"{benchmark.metrics.inter_token_latency_ms.successful.median:.1f}",
                    f"{benchmark.metrics.inter_token_latency_ms.successful.percentiles.p99:.1f}",
                    f"{benchmark.metrics.time_per_output_token_ms.successful.mean:.1f}",
                    f"{benchmark.metrics.time_per_output_token_ms.successful.median:.1f}",
                    f"{benchmark.metrics.time_per_output_token_ms.successful.percentiles.p99:.1f}",
                ]
            )

        self._print_table(headers, rows, "Benchmarks Stats", sections)

    def _get_profile_str(self, benchmark: GenerativeBenchmark) -> str:
        profile = benchmark.benchmarker.get("profile")
        if profile is None:
            return "None"

        profile_args = OrderedDict(
            {
                "type": profile.type_,
                "strategies": getattr(profile, "strategy_types", []),
            }
        )

        if isinstance(profile, ConcurrentProfile):
            profile_args["streams"] = str(profile.streams)
        elif isinstance(profile, ThroughputProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
        elif isinstance(profile, AsyncProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
            profile_args["rate"] = str(profile.rate)
            profile_args["initial_burst"] = str(profile.initial_burst)
        elif isinstance(profile, SweepProfile):
            profile_args["sweep_size"] = str(profile.sweep_size)

        return ", ".join(f"{key}={value}" for key, value in profile_args.items())

    def _get_args_str(self, benchmark: GenerativeBenchmark) -> str:
        args = benchmark.args
        args_dict = OrderedDict(
            {
                "max_number": args.max_number,
                "max_duration": args.max_duration,
                "warmup_number": args.warmup_number,
                "warmup_duration": args.warmup_duration,
                "cooldown_number": args.cooldown_number,
                "cooldown_duration": args.cooldown_duration,
            }
        )
        return ", ".join(f"{key}={value}" for key, value in args_dict.items())

    def _get_extras_str(self, benchmark: GenerativeBenchmark) -> str:
        extras = benchmark.extras
        if not extras:
            return "None"
        return ", ".join(f"{key}={value}" for key, value in extras.items())

    def _print_section_header(self, title: str, indent: int = 0, new_lines: int = 2):
        self._print_line(
            f"{title}:",
            f"bold underline {Colors.INFO}",
            indent=indent,
            new_lines=new_lines,
        )

    def _print_labeled_line(
        self, label: str, value: str, indent: int = 4, new_lines: int = 0
    ):
        self._print_line(
            [label + ":", value],
            ["bold " + Colors.INFO, "italic"],
            new_lines=new_lines,
            indent=indent,
        )

    def _print_line(
        self,
        value: Union[str, list[str]],
        style: Union[str, list[str]] = "",
        indent: int = 0,
        new_lines: int = 0,
    ):
        text = Text()
        for _ in range(new_lines):
            text.append("\n")

        if not isinstance(value, list):
            value = [value]
        if not isinstance(style, list):
            style = [style for _ in range(len(value))]

        if len(value) != len(style):
            raise ValueError(
                f"Value and style length mismatch: {len(value)} vs {len(style)}"
            )

        for val, sty in zip(value, style):
            text.append(val, style=sty)

        self.console.print(Padding.indent(text, indent))

    def _print_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        title: str,
        sections: Optional[dict[str, tuple[int, int]]] = None,
        max_char_per_col: int = 1024,
        indent: int = 0,
        new_lines: int = 2,
    ):
        if rows and any(len(row) != len(headers) for row in rows):
            raise ValueError(
                f"Headers and rows length mismatch: {len(headers)} vs {len(rows[0]) if rows else 'N/A'}"
            )

        max_chars_per_column = self._calculate_max_chars_per_column(
            headers, rows, sections, max_char_per_col
        )

        self._print_section_header(title, indent=indent, new_lines=new_lines)
        self._print_table_divider(max_chars_per_column, False, indent)
        if sections:
            self._print_table_sections(sections, max_chars_per_column, indent)
        self._print_table_row(
            split_text_list_by_length(headers, max_chars_per_column),
            f"bold {Colors.INFO}",
            indent,
        )
        self._print_table_divider(max_chars_per_column, True, indent)
        for row in rows:
            self._print_table_row(
                split_text_list_by_length(row, max_chars_per_column),
                "italic",
                indent,
            )
        self._print_table_divider(max_chars_per_column, False, indent)

    def _calculate_max_chars_per_column(
        self,
        headers: list[str],
        rows: list[list[Any]],
        sections: Optional[dict[str, tuple[int, int]]],
        max_char_per_col: int,
    ) -> list[int]:
        """Calculate maximum characters per column for table formatting."""
        max_chars_per_column = []
        for ind in range(len(headers)):
            max_chars_per_column.append(min(len(headers[ind]), max_char_per_col))
            for row in rows:
                max_chars_per_column[ind] = max(
                    max_chars_per_column[ind], len(str(row[ind]))
                )

        if not sections:
            return max_chars_per_column

        for section, (start_col, end_col) in sections.items():
            min_section_len = len(section) + (end_col - start_col)
            chars_in_columns = sum(
                max_chars_per_column[start_col : end_col + 1]
            ) + 2 * (end_col - start_col)
            if min_section_len > chars_in_columns:
                add_chars_per_col = math.ceil(
                    (min_section_len - chars_in_columns) / (end_col - start_col + 1)
                )
                for col in range(start_col, end_col + 1):
                    max_chars_per_column[col] += add_chars_per_col

        return max_chars_per_column

    def _print_table_divider(
        self, max_chars_per_column: list[int], include_separators: bool, indent: int = 0
    ):
        """Print table divider line."""
        if include_separators:
            columns = [
                settings.table_headers_border_char * max_chars
                + settings.table_column_separator_char
                + settings.table_headers_border_char
                for max_chars in max_chars_per_column
            ]
        else:
            columns = [
                settings.table_border_char * (max_chars + 2)
                for max_chars in max_chars_per_column
            ]
        columns[-1] = columns[-1][:-2]
        self._print_line(columns, Colors.INFO, indent)

    def _print_table_sections(
        self,
        sections: dict[str, tuple[int, int]],
        max_chars_per_column: list[int],
        indent: int = 0,
    ):
        section_tuples = [(start, end, name) for name, (start, end) in sections.items()]
        section_tuples.sort(key=lambda x: x[0])

        if any(start > end for start, end, _ in section_tuples):
            raise ValueError(f"Invalid section ranges: {section_tuples}")

        if (
            any(
                section_tuples[ind][1] + 1 != section_tuples[ind + 1][0]
                for ind in range(len(section_tuples) - 1)
            )
            or section_tuples[0][0] != 0
            or section_tuples[-1][1] != len(max_chars_per_column) - 1
        ):
            raise ValueError(f"Invalid section ranges: {section_tuples}")

        line_values = []
        line_styles = []
        for section, (start_col, end_col) in sections.items():
            section_length = sum(max_chars_per_column[start_col : end_col + 1]) + 2 * (
                end_col - start_col + 1
            )
            num_separators = end_col - start_col
            line_values.extend(
                [
                    section,
                    " " * (section_length - len(section) - num_separators - 2),
                    settings.table_column_separator_char * num_separators,
                    settings.table_column_separator_char + " ",
                ]
            )
            line_styles.extend(["bold " + Colors.INFO, "", "", Colors.INFO])

        line_values = line_values[:-1]
        line_styles = line_styles[:-1]
        self._print_line(line_values, line_styles, indent)

    def _print_table_row(
        self, column_lines: list[list[str]], style: str, indent: int = 0
    ):
        for row in range(len(column_lines[0])):
            print_line = []
            print_styles = []
            for column in range(len(column_lines)):
                print_line.extend(
                    [
                        column_lines[column][row],
                        settings.table_column_separator_char,
                        " ",
                    ]
                )
                print_styles.extend([style, Colors.INFO, ""])
            print_line = print_line[:-2]
            print_styles = print_styles[:-2]
            self._print_line(print_line, print_styles, indent)


class GenerativeBenchmarkerCSV(GenerativeBenchmarkerOutput):
    """CSV output formatter for benchmark results."""

    DEFAULT_FILE: ClassVar[str] = "benchmarks.json"

    def __init__(self, output_path: Optional[Union[str, Path]] = None):
        """
        Initialize the CSV output formatter.

        :param output_path: Optional path where CSV file should be saved.
            If not provided, will be saved to the default location.
        """
        output_path = output_path or GenerativeBenchmarkerCSV.DEFAULT_FILE
        output_path = (
            Path(output_path) if not isinstance(output_path, Path) else output_path
        )

        if output_path.is_dir():
            output_path = output_path / GenerativeBenchmarkerCSV.DEFAULT_FILE

        self.output_path = output_path

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Save the benchmark report as a CSV file.

        :param report: The completed benchmark report.
        :return: Path to the saved CSV file.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            headers: list[str] = []
            rows: list[list[Union[str, float, list[float]]]] = []

            for benchmark in report.benchmarks:
                benchmark_headers: list[str] = []
                benchmark_values: list[Union[str, float, list[float]]] = []

                # Add description headers and values
                desc_headers, desc_values = self._get_benchmark_desc_headers_and_values(
                    benchmark
                )
                benchmark_headers.extend(desc_headers)
                benchmark_values.extend(desc_values)

                # Add status-based metrics
                for status in StatusDistributionSummary.model_fields:
                    status_headers, status_values = (
                        self._get_benchmark_status_headers_and_values(benchmark, status)
                    )
                    benchmark_headers.extend(status_headers)
                    benchmark_values.extend(status_values)

                # Add extra fields
                extras_headers, extras_values = (
                    self._get_benchmark_extras_headers_and_values(benchmark)
                )
                benchmark_headers.extend(extras_headers)
                benchmark_values.extend(extras_values)

                if not headers:
                    headers = benchmark_headers
                rows.append(benchmark_values)

            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)

        return self.output_path

    def _get_benchmark_desc_headers_and_values(
        self, benchmark: GenerativeBenchmark
    ) -> tuple[list[str], list[Union[str, float]]]:
        """Get description headers and values for a benchmark."""
        headers = [
            "Type",
            "Run Id",
            "Id",
            "Name",
            "Start Time",
            "End Time",
            "Duration",
        ]
        values: list[Union[str, float]] = [
            benchmark.type_,
            benchmark.run_id,
            benchmark.id_,
            strategy_display_str(benchmark.args.strategy),
            datetime.fromtimestamp(benchmark.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(benchmark.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            benchmark.duration,
        ]
        return headers, values

    def _get_benchmark_extras_headers_and_values(
        self, benchmark: GenerativeBenchmark
    ) -> tuple[list[str], list[str]]:
        """Get extra fields headers and values for a benchmark."""
        headers = ["Args", "Worker", "Request Loader", "Extras"]
        values: list[str] = [
            json.dumps(benchmark.args.model_dump()),
            json.dumps(benchmark.worker.model_dump()),
            json.dumps(benchmark.request_loader.model_dump()),
            json.dumps(benchmark.extras),
        ]
        return headers, values

    def _get_benchmark_status_headers_and_values(
        self, benchmark: GenerativeBenchmark, status: str
    ) -> tuple[list[str], list[Union[float, list[float]]]]:
        """Get status-based metrics headers and values for a benchmark."""
        headers = [f"{status.capitalize()} Requests"]
        values = [getattr(benchmark.request_totals, status)]

        for metric in GenerativeMetrics.model_fields:
            metric_headers, metric_values = self._get_benchmark_status_metrics_stats(
                benchmark, status, metric
            )
            headers.extend(metric_headers)
            values.extend(metric_values)

        return headers, values

    def _get_benchmark_status_metrics_stats(
        self, benchmark: GenerativeBenchmark, status: str, metric: str
    ) -> tuple[list[str], list[Union[float, list[float]]]]:
        """Get statistical metrics for a specific status and metric."""
        status_display = status.capitalize()
        metric_display = metric.replace("_", " ").capitalize()
        status_dist_summary: StatusDistributionSummary = getattr(
            benchmark.metrics, metric
        )
        dist_summary: DistributionSummary = getattr(status_dist_summary, status)

        headers = [
            f"{status_display} {metric_display} mean",
            f"{status_display} {metric_display} median",
            f"{status_display} {metric_display} std dev",
            f"{status_display} {metric_display} [min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]",
        ]
        values: list[Union[float, list[float]]] = [
            dist_summary.mean,
            dist_summary.median,
            dist_summary.std_dev,
            [
                dist_summary.min,
                dist_summary.percentiles.p001,
                dist_summary.percentiles.p01,
                dist_summary.percentiles.p05,
                dist_summary.percentiles.p10,
                dist_summary.percentiles.p25,
                dist_summary.percentiles.p75,
                dist_summary.percentiles.p90,
                dist_summary.percentiles.p95,
                dist_summary.percentiles.p99,
                dist_summary.max,
            ],
        ]
        return headers, values


class GenerativeBenchmarkerHTML(GenerativeBenchmarkerOutput):
    """HTML output formatter for benchmark results."""

    DEFAULT_FILE: ClassVar[str] = "benchmarks.html"

    def __init__(self, output_path: Optional[Union[str, Path]] = None):
        """
        Initialize the HTML output formatter.

        :param output_path: Optional path where HTML file should be saved.
            If not provided, will be saved to the default location.
        """
        output_path = output_path or GenerativeBenchmarkerCSV.DEFAULT_FILE
        output_path = (
            Path(output_path) if not isinstance(output_path, Path) else output_path
        )

        if output_path.is_dir():
            output_path = output_path / GenerativeBenchmarkerCSV.DEFAULT_FILE

        self.output_path = output_path

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Save the benchmark report as an HTML file.

        :param report: The completed benchmark report.
        :return: Path to the saved HTML file.
        """
        data_builder = UIDataBuilder(report.benchmarks)
        data = data_builder.to_dict()
        camel_data = humps.camelize(data)

        ui_api_data = {}
        for key, value in camel_data.items():
            placeholder_key = f"window.{humps.decamelize(key)} = {{}};"
            replacement_value = (
                f"window.{humps.decamelize(key)} = {json.dumps(value, indent=2)};\n"
            )
            ui_api_data[placeholder_key] = replacement_value

        create_report(ui_api_data, self.output_path)

        return str(self.output_path)
