from datetime import datetime
from typing import List, Optional

from pydantic import Field
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from guidellm.core.result import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.core.serializable import Serializable

__all__ = [
    "GuidanceReport",
]


def _create_benchmark_report_details(report: TextGenerationBenchmarkReport) -> str:
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
        f"Limits(max_number={report.args.get('max_number', 'N/A')}, "
        f"max_duration={report.args.get('max_duration', 'N/A')})"
    )

    return backend + "\n" + data + "\n" + rate + "\n" + limits + "\n"


def _benchmark_rate_id(benchmark: TextGenerationBenchmark) -> str:
    return (
        f"{benchmark.mode}@{benchmark.rate:.2f} req/sec"
        if benchmark.rate
        else f"{benchmark.mode}"
    )


def _create_benchmark_report_requests_summary(
    report: TextGenerationBenchmarkReport,
) -> Table:
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
        start_time_str = datetime.fromtimestamp(benchmark.start_time).strftime(  # type: ignore  # noqa: PGH003
            "%H:%M:%S"
        )
        end_time_str = datetime.fromtimestamp(benchmark.end_time).strftime(  # type: ignore  # noqa: PGH003
            "%H:%M:%S"
        )

        table.add_row(
            _benchmark_rate_id(benchmark),
            f"{benchmark.request_count}/{benchmark.total_count}",
            f"{benchmark.error_count}/{benchmark.total_count}",
            f"{benchmark.duration:.2f} sec",
            f"{start_time_str}",
            f"{end_time_str}",
        )

    return table


def _create_benchmark_report_data_tokens_summary(
    report: TextGenerationBenchmarkReport,
) -> Table:
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
            f"{benchmark.prompt_token_distribution.mean:.2f}",
            ", ".join(
                f"{percentile:.1f}"
                for percentile in benchmark.prompt_token_distribution.percentiles(
                    [1, 5, 50, 95, 99]
                )
            ),
            f"{benchmark.output_token_distribution.mean:.2f}",
            ", ".join(
                f"{percentile:.1f}"
                for percentile in benchmark.output_token_distribution.percentiles(
                    [1, 5, 50, 95, 99]
                )
            ),
        )

    return table


def _create_benchmark_report_dist_perf_summary(
    report: TextGenerationBenchmarkReport,
) -> Table:
    table = Table(
        "Benchmark",
        "Request Latency [1%, 5%, 10%, 50%, 90%, 95%, 99%] (sec)",
        "Time to First Token [1%, 5%, 10%, 50%, 90%, 95%, 99%] (ms)",
        "Inter Token Latency [1%, 5%, 10%, 50%, 90% 95%, 99%] (ms)",
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
                for percentile in benchmark.request_latency_distribution.percentiles(
                    [1, 5, 10, 50, 90, 95, 99]
                )
            ),
            ", ".join(
                f"{percentile*1000:.1f}"
                for percentile in benchmark.ttft_distribution.percentiles(
                    [1, 5, 10, 50, 90, 95, 99]
                )
            ),
            f"{benchmark.inter_token_latency:.2f} ms",
            ", ".join(
                f"{percentile*1000:.1f}"
                for percentile in benchmark.itl_distribution.percentiles(
                    [1, 5, 10, 50, 90, 95, 99]
                )
            ),
            f"{benchmark.output_token_throughput:.2f} tokens/sec",
        )

    return table


def _create_benchmark_report_summary(report: TextGenerationBenchmarkReport) -> Table:
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

    return table


class GuidanceReport(Serializable):
    """
    A class to manage the guidance reports that include the benchmarking details,
    potentially across multiple runs, for saving and loading from disk.
    """

    benchmarks: List[TextGenerationBenchmarkReport] = Field(
        default_factory=list, description="The list of report reports."
    )

    def print(self, save_path: Optional[str] = None):
        """
        Print the guidance report to the console.
        """
        console = Console()
        console.print(
            f"\n\n📊 [bold cyan]GuideLLM Benchmarks Report[/bold cyan] [italic]"
            f"({save_path})[/italic]\n",
            style="underline",
        )

        for index, benchmark in enumerate(self.benchmarks):
            console.print(
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
                    title=f"[magenta]Benchmark Report {index + 1}[/magenta]",
                    expand=True,
                    title_align="left",
                )
            )
