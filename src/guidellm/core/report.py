import time
from datetime import datetime
from typing import List, Optional

from loguru import logger
from pydantic import Field
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from guidellm.core.result import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.core.serializable import Serializable

__all__ = ["GuidanceReport"]


def _create_benchmark_report_details(report: TextGenerationBenchmarkReport) -> str:
    """
    Create a detailed string representation of a benchmark report.

    :param report: The benchmark report to generate details for.
    :type report: TextGenerationBenchmarkReport
    :return: A string containing the backend, data, rate, and limits of
        the benchmark report.
    :rtype: str
    """
    backend = (
        f"Backend(type={report.args.get('backend_type', 'N/A')}, "
        f"target={report.args.get('target', 'N/A')}, "
        f"model={report.args.get('model', 'N/A')})"
    )
    data = (
        f"Data(type={report.args.get('data_type', 'N/A')}, "
        f"source={report.args.get('data', 'N/A')}, "
        f"tokenizer={report.args.get('tokenizer', 'N/A')})"
    )
    rate = (
        f"Rate(type={report.args.get('mode', 'N/A')}, "
        f"rate={report.args.get('rate', 'N/A')})"
    )
    limits = (
        f"Limits(max_number={report.args.get('max_number', 'N/A')} requests, "
        f"max_duration={report.args.get('max_duration', 'N/A')} sec)"
    )

    logger.debug(
        "Created benchmark report details for backend={}, data={}, rate={}, limits={}",
        backend,
        data,
        rate,
        limits,
    )

    return backend + "\n" + data + "\n" + rate + "\n" + limits + "\n"


def _benchmark_rate_id(benchmark: TextGenerationBenchmark) -> str:
    """
    Generate a string identifier for a benchmark rate.

    :param benchmark: The benchmark for which to generate the rate ID.
    :type benchmark: TextGenerationBenchmark
    :return: A string representing the benchmark rate ID.
    :rtype: str
    """
    rate_id = (
        f"{benchmark.mode}@{benchmark.rate:.2f} req/sec"
        if benchmark.rate
        else f"{benchmark.mode}"
    )
    logger.debug("Generated benchmark rate ID: {}", rate_id)
    return rate_id


def _create_benchmark_report_requests_summary(
    report: TextGenerationBenchmarkReport,
) -> Table:
    """
    Create a table summarizing the requests of a benchmark report.

    :param report: The benchmark report to summarize.
    :type report: TextGenerationBenchmarkReport
    :return: A rich Table object summarizing the requests.
    :rtype: Table
    """
    table = Table(
        "Benchmark",
        "Requests Completed",
        "Request Failed",
        "Duration",
        "Start Time",
        "End Time",
        title="[magenta]Requests Data by Benchmark[/magenta]",
        title_style="bold",
        title_justify="left",
        show_header=True,
    )

    for benchmark in report.benchmarks_sorted:
        start_time_str = (
            datetime.fromtimestamp(benchmark.start_time).strftime("%H:%M:%S")
            if benchmark.start_time
            else "N/A"
        )
        end_time_str = (
            datetime.fromtimestamp(benchmark.end_time).strftime("%H:%M:%S")
            if benchmark.end_time
            else "N/A"
        )

        table.add_row(
            _benchmark_rate_id(benchmark),
            f"{benchmark.request_count}/{benchmark.total_count}",
            f"{benchmark.error_count}/{benchmark.total_count}",
            f"{benchmark.duration:.2f} sec",
            f"{start_time_str}",
            f"{end_time_str}",
        )
    logger.debug("Created requests summary table for the report.")
    return table


def _create_benchmark_report_data_tokens_summary(
    report: TextGenerationBenchmarkReport,
) -> Table:
    """
    Create a table summarizing data tokens of a benchmark report.

    :param report: The benchmark report to summarize.
    :type report: TextGenerationBenchmarkReport
    :return: A rich Table object summarizing the data tokens.
    :rtype: Table
    """
    table = Table(
        "Benchmark",
        "Prompt",
        "Prompt (1%, 5%, 50%, 95%, 99%)",
        "Output",
        "Output (1%, 5%, 50%, 95%, 99%)",
        title="[magenta]Tokens Data by Benchmark[/magenta]",
        title_style="bold",
        title_justify="left",
        show_header=True,
    )

    for benchmark in report.benchmarks_sorted:
        table.add_row(
            _benchmark_rate_id(benchmark),
            f"{benchmark.prompt_token:.2f}",
            ", ".join(
                f"{percentile:.1f}"
                for percentile in benchmark.prompt_token_percentiles
            ),
            f"{benchmark.output_token:.2f}",
            ", ".join(
                f"{percentile:.1f}"
                for percentile in benchmark.output_token_percentiles
            ),
        )
    logger.debug("Created data tokens summary table for the report.")
    return table


def _create_benchmark_report_dist_perf_summary(
    report: TextGenerationBenchmarkReport,
) -> Table:
    """
    Create a table summarizing distribution performance of a benchmark report.

    :param report: The benchmark report to summarize.
    :type report: TextGenerationBenchmarkReport
    :return: A rich Table object summarizing the performance statistics.
    :rtype: Table
    """
    table = Table(
        "Benchmark",
        "Request Latency [1%, 5%, 10%, 50%, 90%, 95%, 99%] (sec)",
        "Time to First Token [1%, 5%, 10%, 50%, 90%, 95%, 99%] (ms)",
        "Inter Token Latency [1%, 5%, 10%, 50%, 90%, 95%, 99%] (ms)",
        title="[magenta]Performance Stats by Benchmark[/magenta]",
        title_style="bold",
        title_justify="left",
        show_header=True,
    )

    for benchmark in report.benchmarks_sorted:
        table.add_row(
            _benchmark_rate_id(benchmark),
            ", ".join(
                f"{percentile:.2f}"
                for percentile in benchmark.request_latency_percentiles
            ),
            ", ".join(
                f"{percentile * 1000:.1f}"
                for percentile in benchmark.time_to_first_token_percentiles
            ),
            ", ".join(
                f"{percentile * 1000:.1f}"
                for percentile in benchmark.inter_token_latency_percentiles
            ),
        )
    logger.debug("Created distribution performance summary table for the report.")
    return table


def _create_benchmark_report_summary(report: TextGenerationBenchmarkReport) -> Table:
    """
    Create a summary table for a benchmark report.

    :param report: The benchmark report to summarize.
    :type report: TextGenerationBenchmarkReport
    :return: A rich Table object summarizing overall performance.
    :rtype: Table
    """
    table = Table(
        "Benchmark",
        "Requests per Second",
        "Request Latency",
        "Time to First Token",
        "Inter Token Latency",
        "Output Token Throughput",
        title="[magenta]Performance Summary by Benchmark[/magenta]",
        title_style="bold",
        title_justify="left",
        show_header=True,
    )

    for benchmark in report.benchmarks_sorted:
        table.add_row(
            _benchmark_rate_id(benchmark),
            f"{benchmark.completed_request_rate:.2f} req/sec",
            f"{benchmark.request_latency:.2f} sec",
            f"{benchmark.time_to_first_token:.2f} ms",
            f"{benchmark.inter_token_latency:.2f} ms",
            f"{benchmark.output_token_throughput:.2f} tokens/sec",
        )
    logger.debug("Created overall performance summary table for the report.")
    return table


class GuidanceReport(Serializable):
    """
    A class to manage the guidance reports that include the benchmarking details,
    potentially across multiple runs, for saving and loading from disk.

    :param benchmarks: The list of benchmarking reports.
    :type benchmarks: List[TextGenerationBenchmarkReport]
    """

    benchmarks: List[TextGenerationBenchmarkReport] = Field(
        default_factory=list, description="The list of benchmark reports."
    )

    def print(
        self, save_path: Optional[str] = None, continual_refresh: bool = False
    ) -> None:
        """
        Print the guidance report to the console.

        :param save_path: Optional path to save the report to disk.
        :type save_path: Optional[str]
        :param continual_refresh: Whether to continually refresh the report.
        :type continual_refresh: bool
        :return: None
        """
        logger.info("Printing guidance report to console with save_path={}", save_path)
        report_viz = Panel(
            Group(
                *[
                    Panel(
                        Group(
                            _create_benchmark_report_details(benchmark),
                            "",
                            _create_benchmark_report_requests_summary(benchmark),
                            "",
                            _create_benchmark_report_data_tokens_summary(benchmark),
                            "",
                            _create_benchmark_report_dist_perf_summary(benchmark),
                            "",
                            _create_benchmark_report_summary(benchmark),
                        ),
                        title=(
                            f"[bold magenta]Benchmark Report "
                            f"{index + 1}[/bold magenta]"
                        ),
                        expand=True,
                        title_align="left",
                    )
                    for index, benchmark in enumerate(self.benchmarks)
                ],
            ),
            title=(
                "[bold cyan]GuideLLM Benchmarks Report[/bold cyan] [italic]"
                f"({save_path})[/italic]"
            ),
            expand=True,
            title_align="left",
        )
        console = Console()

        if continual_refresh:
            logger.info("Starting live report with continual refresh.")
            with Live(report_viz, refresh_per_second=1, console=console) as live:
                while True:
                    live.update(report_viz)
                    time.sleep(1)
        else:
            console.print(report_viz)

        logger.info("Guidance report printing completed.")
