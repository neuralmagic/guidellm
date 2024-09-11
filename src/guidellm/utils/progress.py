from datetime import datetime
from typing import List

from loguru import logger
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

__all__ = ["BenchmarkReportProgress"]


class BenchmarkReportProgress:
    """
    Manages the progress display for benchmarks and report generation using Rich.

    This class provides a visual representation of the benchmarking process
    and report generation using Rich's progress bars and panels.
    """

    def __init__(self):
        """
        Initialize the BenchmarkReportProgress with default settings.

        This method sets up the progress displays for both individual benchmarks
        and the overall report, as well as initializing internal task management
        structures.
        """
        logger.info("Initializing BenchmarkReportProgress instance")

        self.benchmarks_progress = Progress(
            TextColumn("[{task.fields[start_time_str]}]"),
            SpinnerColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
            TextColumn(" "),
            TextColumn(
                "[bold cyan]({task.fields[req_per_sec]} req/sec avg)[/bold cyan]"
            ),
        )
        self.benchmarks_panel = Panel(
            self.benchmarks_progress,
            title="Benchmarks",
            title_align="left",
            expand=True,
        )
        self.report_progress = Progress(
            SpinnerColumn(),
            TextColumn("Generating report..."),
            BarColumn(bar_width=None),
            TextColumn(
                "({task.fields[completed_benchmarks]}/{task.fields[total_benchmarks]})"
            ),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("]"),
        )
        self.render_group = Group(self.benchmarks_panel, self.report_progress)
        self.live = Live(self.render_group, redirect_stdout=True, redirect_stderr=True)

        self.report_task: TaskID = None  # type: ignore  # noqa: PGH003
        self.benchmark_tasks: List[TaskID] = []
        self.benchmark_tasks_started: List[bool] = []
        self.benchmark_tasks_completed: List[bool] = []
        self.benchmark_tasks_progress: List[float] = []

    def start(self, task_descriptions: List[str]) -> None:
        """
        Starts the live progress display and initializes benchmark tasks.

        :param task_descriptions: List of descriptions for each benchmark task.
        :type task_descriptions: List[str]
        """
        logger.info(
            "Starting BenchmarkReportProgress with task descriptions: {}",
            task_descriptions,
        )
        self.live.start()

        for task_description in task_descriptions:
            logger.debug("Adding task with description: {}", task_description)
            task_id = self.benchmarks_progress.add_task(
                task_description,
                start=False,
                total=None,
                start_time_str="--:--:--",
                req_per_sec="#.##",
            )
            self.benchmark_tasks.append(task_id)
            self.benchmark_tasks_started.append(False)
            self.benchmark_tasks_completed.append(False)
            self.benchmark_tasks_progress.append(0)

        self.report_task = self.report_progress.add_task(
            "",
            total=len(self.benchmark_tasks) * 100,  # 100 points per report
            completed_benchmarks=0,
            total_benchmarks=len(task_descriptions),
        )
        logger.info("Initialized {} benchmark tasks", len(task_descriptions))

    def update_benchmark(
        self,
        index: int,
        description: str,
        completed: bool,
        completed_count: int,
        completed_total: int,
        start_time: float,
        req_per_sec: float,
    ) -> None:
        """
        Updates the progress of a specific benchmark task.

        :param index: Index of the benchmark task to update.
        :type index: int
        :param description: Description of the current benchmark task.
        :type description: str
        :param completed: Flag indicating if the benchmark is completed.
        :type completed: bool
        :param completed_count: Number of completed operations for the task.
        :type completed_count: int
        :param completed_total: Total number of operations for the task.
        :type completed_total: int
        :param start_time: Start time of the benchmark in timestamp format.
        :type start_time: float
        :param req_per_sec: Average requests per second.
        :type req_per_sec: float
        :raises ValueError: If trying to update a completed benchmark.
        """
        if self.benchmark_tasks_completed[index]:
            err = ValueError(f"Benchmark {index} already completed")
            logger.error("Error updating benchmark: {}", err)
            raise err

        if not self.benchmark_tasks_started[index]:
            self.benchmark_tasks_started[index] = True
            self.benchmarks_progress.start_task(self.benchmark_tasks[index])
            logger.info("Starting benchmark task at index {}", index)

        if completed:
            self.benchmark_tasks_completed[index] = True
            self.benchmark_tasks_progress[index] = 100
            self.benchmarks_progress.stop_task(self.benchmark_tasks[index])
            logger.info("Completed benchmark task at index {}", index)

        self.benchmark_tasks_progress[index] = completed_count / completed_total * 100
        self.benchmarks_progress.update(
            self.benchmark_tasks[index],
            description=description,
            total=completed_total,
            completed=completed_count if not completed else completed_total,
            req_per_sec=(f"{req_per_sec:.2f}" if req_per_sec else "#.##"),
            start_time_str=(
                datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
                if start_time
                else "--:--:--"
            ),
        )
        logger.debug(
            "Updated benchmark task at index {}: {}% complete",
            index,
            self.benchmark_tasks_progress[index],
        )
        self.report_progress.update(
            self.report_task,
            total=len(self.benchmark_tasks) * 100,
            completed=sum(self.benchmark_tasks_progress),
            completed_benchmarks=sum(self.benchmark_tasks_completed),
            total_benchmarks=len(self.benchmark_tasks),
        )

    def finish(self) -> None:
        """
        Marks the overall report task as finished and stops the live display.
        """
        logger.info("Finishing BenchmarkReportProgress")
        self.report_progress.update(
            self.report_task,
            total=len(self.benchmark_tasks) * 100,
            completed=len(self.benchmark_tasks) * 100,
            completed_benchmarks=len(self.benchmark_tasks),
            total_benchmarks=len(self.benchmark_tasks),
        )
        self.report_progress.stop_task(self.report_task)
        self.live.stop()
        logger.info("BenchmarkReportProgress finished and live display stopped")
