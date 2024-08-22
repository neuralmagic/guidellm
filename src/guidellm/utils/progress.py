from datetime import datetime
from typing import List

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
    def __init__(self):
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
        self.render_group = Group(
            self.benchmarks_panel,
            self.report_progress,
        )
        self.live = Live(self.render_group, redirect_stdout=True, redirect_stderr=True)

        self.report_task: TaskID = None  # type: ignore  # noqa: PGH003
        self.benchmark_tasks: List[TaskID] = []
        self.benchmark_tasks_started: List[bool] = []
        self.benchmark_tasks_completed: List[bool] = []
        self.benchmark_tasks_progress: List[float] = []

    def start(self, task_descriptions: List[str]):
        self.live.start()

        for task_description in task_descriptions:
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

    def update_benchmark(
        self,
        index: int,
        description: str,
        completed: bool,
        completed_count: int,
        completed_total: int,
        start_time: float,
        req_per_sec: float,
    ):
        if self.benchmark_tasks_completed[index]:
            raise ValueError(f"Benchmark {index} already completed")

        if not self.benchmark_tasks_started[index]:
            self.benchmark_tasks_started[index] = True
            self.benchmarks_progress.start_task(self.benchmark_tasks[index])

        if completed:
            self.benchmark_tasks_completed[index] = True
            self.benchmark_tasks_progress[index] = 100
            self.benchmarks_progress.stop_task(self.benchmark_tasks[index])

        self.benchmark_tasks_progress[index] = completed_count / completed_total * 100
        self.benchmarks_progress.update(
            self.benchmark_tasks[index],
            description=description,
            total=completed_total,
            completed=completed_count if not completed else completed_total,
            req_per_sec=(f"{req_per_sec:.2f}" if req_per_sec else "#.##"),
            start_time_str=datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
            if start_time
            else "--:--:--",
        )
        self.report_progress.update(
            self.report_task,
            total=len(self.benchmark_tasks) * 100,
            completed=sum(self.benchmark_tasks_progress),
            completed_benchmarks=sum(self.benchmark_tasks_completed),
            total_bechnmarks=len(self.benchmark_tasks),
        )

    def finish(self):
        self.report_progress.update(
            self.report_task,
            total=len(self.benchmark_tasks) * 100,
            completed=len(self.benchmark_tasks) * 100,
            completed_benchmarks=len(self.benchmark_tasks),
            total_bechnmarks=len(self.benchmark_tasks),
        )
        self.report_progress.stop_task(self.report_task)
        self.live.stop()
