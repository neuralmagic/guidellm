import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Generic, List, Optional, TypeVar, Union

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from guidellm.benchmark.benchmarker import BenchmarkerResult
from guidellm.scheduler import (
    SchedulingStrategy,
    StrategyType,
    strategy_display_str,
)
from guidellm.utils import Colors


@dataclass
class BenchmarkerTaskProgressState:
    display_scheduler_stats: bool

    task_id: TaskID
    strategy: Union[StrategyType, SchedulingStrategy]
    started: bool = False
    compiling: bool = False
    ended: bool = False

    start_time: Optional[float] = None
    max_number: Optional[int] = None
    max_duration: Optional[float] = None
    in_warmup: bool = False
    in_cooldown: bool = False

    requests_rate: float = 0
    request_latency: float = 0
    requests_processing: int = 0
    requests_successful: int = 0
    requests_incomplete: int = 0
    requests_errored: int = 0

    worker_overheads_time_ms: float = 0.0
    backend_overheads_time_ms: float = 0.0
    requests_sleep_time_ms: float = 0.0
    requests_targeted_start_time_delay_ms: float = 0.0

    @property
    def description(self) -> str:
        return strategy_display_str(self.strategy)

    @property
    def total(self) -> Optional[float]:
        if self.max_number is None and self.max_duration is None:
            return None

        return 1000

    @property
    def completed(self) -> int:
        if self.ended:
            return 1000

        if self.max_number is None and self.max_duration is None:
            return 0

        number = self.requests_successful + self.requests_errored
        number_percent = (
            number / float(self.max_number) * 1000 if self.max_number else -math.inf
        )
        duration_percent = (
            (time.time() - self.start_time) / self.max_duration * 1000
            if self.max_duration
            else -math.inf
        )

        return min(int(max(number_percent, duration_percent)), 1000)

    @property
    def fields(self) -> Dict[str, str]:
        fields = {
            "start_time": self.formatted_start_time,
            "progress_status": self.formatted_progress_status,
            "requests_summary": self.formatted_requests_summary,
        }

        if self.display_scheduler_stats:
            fields["scheduler_stats"] = self.formatted_scheduler_stats

        return fields

    @property
    def formatted_start_time(self) -> str:
        if self.start_time is None:
            return "--:--:--"

        return datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")

    @property
    def formatted_progress_status(self) -> str:
        if self.ended:
            status = "complete"
            color = Colors.SUCCESS
        elif self.compiling:
            status = "compiling"
            color = Colors.PROGRESS
        elif self.started and self.in_warmup:
            status = "warmup"
            color = Colors.PROGRESS
        elif self.started and self.in_cooldown:
            status = "cooldown"
            color = Colors.PROGRESS
        elif self.started:
            status = "running"
            color = Colors.PROGRESS
        else:
            status = "pending"
            color = Colors.INFO

        return f"[{color}]{status.ljust(8)}[/{color}]"

    @property
    def formatted_requests_summary(self) -> str:
        if not self.started:
            return " "

        return (
            f"[{Colors.INFO}]Req:[/{Colors.INFO}] "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_rate,
                label="req/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.request_latency,
                label="Lat",
                units="s",
                total_characters=12,
                digits_places=4,
                decimal_places=2,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_processing,
                label="Conc",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_successful,
                label="Comp",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_incomplete,
                label="Inc",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_errored,
                label="Err",
                total_characters=12,
                digits_places=5,
                decimal_places=0,
            )
        )

    @property
    def formatted_scheduler_stats(self) -> str:
        if not self.started:
            return " "

        return (
            f"[{Colors.INFO}]Sys:[/{Colors.INFO}] "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.worker_overheads_time_ms,
                label="Work OH",
                units="ms",
                total_characters=18,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.backend_overheads_time_ms,
                label="Back OH",
                units="ms",
                total_characters=18,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_sleep_time_ms,
                label="Req Sleep",
                units="ms",
                total_characters=18,
                digits_places=5,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.requests_targeted_start_time_delay_ms,
                label="Start Del",
                units="ms",
                total_characters=18,
                digits_places=5,
                decimal_places=0,
            )
        )

    @staticmethod
    def format_progress_display(
        value: float,
        label: str,
        units: str = "",
        total_characters: Optional[int] = None,
        digits_places: Optional[int] = None,
        decimal_places: Optional[int] = None,
    ) -> str:
        if decimal_places is None and digits_places is None:
            formatted_number = f"{value}:.0f"
        elif digits_places is None:
            formatted_number = f"{value:.{decimal_places}f}"
        elif decimal_places is None:
            formatted_number = f"{value:>{digits_places}f}"
        else:
            formatted_number = f"{value:>{digits_places}.{decimal_places}f}"

        result = f"{formatted_number}{units} [{Colors.INFO}]{label}[/{Colors.INFO}]"
        total_characters += len(Colors.INFO) * 2 + 5

        if total_characters is not None and len(result) < total_characters:
            result = result.rjust(total_characters)

        return result


class GenerativeTextBenchmarkerTaskProgressState(BenchmarkerTaskProgressState):
    output_tokens: float = 0
    prompt_tokens: float = 0
    output_tokens_rate: float = 0
    total_tokens_rate: float = 0
    tokens_ttft: float = 0
    tokens_itl: float = 0

    @property
    def fields(self) -> Dict[str, str]:
        fields = super().fields
        fields["tokens_summary"] = self.formatted_tokens_summary
        return fields

    @property
    def formatted_tokens_summary(self) -> str:
        if not self.started:
            return " "

        return (
            f"[{Colors.INFO}]Tok:[/{Colors.INFO}] "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.output_tokens_rate,
                label="gen/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.total_tokens_rate,
                label="tot/s",
                total_characters=12,
                digits_places=4,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.tokens_ttft,
                label="TTFT",
                units="ms",
                total_characters=12,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.tokens_itl,
                label="ITL",
                units="ms",
                total_characters=12,
                digits_places=3,
                decimal_places=1,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.prompt_tokens,
                label="Prompt",
                total_characters=12,
                digits_places=4,
                decimal_places=0,
            )
            + ", "
            + BenchmarkerTaskProgressState.format_progress_display(
                value=self.output_tokens,
                label="Gen",
                total_characters=12,
                digits_places=4,
                decimal_places=0,
            )
        )


BTPS = TypeVar("BTPS", bound=BenchmarkerTaskProgressState)


class BenchmarkerProgressDisplay(Generic[BTPS]):
    def __init__(self, display_scheduler_stats: bool):
        self.display_scheduler_stats = display_scheduler_stats
        self.started = False
        self.benchmarker_tasks_progress = Progress(*self.create_task_progress_columns())
        self.benchmarker_tasks_panel = Panel(
            self.benchmarker_tasks_progress,
            title="Benchmarks",
            title_align="left",
            expand=True,
        )
        self.benchmarker_progress = Progress(
            TextColumn("Generating...", style=f"italic {Colors.PROGRESS}"),
            BarColumn(
                bar_width=None,
                complete_style=Colors.PROGRESS,
                finished_style=Colors.SUCCESS,
            ),
            TextColumn(
                "({task.fields[completed_benchmarks]}/{task.fields[total_benchmarks]})",
                style=Colors.PROGRESS,
            ),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("]"),
        )
        self.benchmarker_live = Live(
            Group(
                self.benchmarker_tasks_panel,
                self.benchmarker_progress,
            ),
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self.active_task: Optional[TaskID] = None
        self.benchmarker_tasks: List[BenchmarkerTaskProgressState] = []
        self.progress_task: Optional[TaskID] = None

    def update(self, result: BenchmarkerResult):
        if result.type_ == "run_start":
            if self.started:
                raise RuntimeError("Progress display already started.")

            self.handle_start(result)
            self.started = True
        elif result.type_ == "run_complete":
            if not self.started:
                raise RuntimeError("Progress display not started.")

            self.handle_end(result)
            self.started = False
        else:
            if not self.started:
                raise RuntimeError("Progress display not started.")

            self.handle_update(result)

    def handle_start(self, result: BenchmarkerResult):
        self.benchmarker_live.start()

        for index, strategy_type in enumerate(result.profile.strategy_types):
            task_id = self.benchmarker_tasks_progress.add_task(
                description=strategy_type,
                start=False,
                total=None,
                completed=0,
                visible=False,
            )
            task_progress_state = self.create_task_progress_state(
                task_id=task_id,
                index=index,
                strategy_type=strategy_type,
                result=result,
            )
            self.benchmarker_tasks.append(task_progress_state)
            self.benchmarker_tasks_progress.update(
                task_id,
                description=task_progress_state.description,
                visible=True,
                **task_progress_state.fields,
            )

        self.progress_task = self.benchmarker_progress.add_task(
            "",
            total=len(self.benchmarker_tasks) * 1000,
            completed_benchmarks=0,
            total_benchmarks=len(self.benchmarker_tasks),
        )

    def handle_update(self, result: BenchmarkerResult):
        current_state = self.benchmarker_tasks[result.current_index]

        if result.type_ == "scheduler_start":
            self.handle_update_scheduler_start(current_state, result)
            self.active_task = current_state.task_id
        elif result.type_ == "scheduler_update":
            self.handle_update_scheduler_update(current_state, result)
        elif result.type_ == "scheduler_complete":
            self.handle_update_scheduler_complete(current_state, result)
        elif result.type_ == "benchmark_compiled":
            self.handle_update_benchmark_compiled(current_state, result)
        else:
            raise ValueError(f"Unknown result type: {result.type_}")

        self.benchmarker_tasks_progress.update(
            current_state.task_id,
            description=current_state.description,
            completed=current_state.completed,
            total=current_state.total,
            **current_state.fields,
        )
        self.benchmarker_progress.update(
            self.progress_task,
            completed=(result.current_index * 1000) + current_state.completed,
            total=1000 * len(self.benchmarker_tasks),
            completed_benchmarks=(
                result.current_index + (1 if current_state.ended else 0)
            ),
            total_benchmarks=len(self.benchmarker_tasks),
        )

        if current_state.ended:
            self.benchmarker_tasks_progress.stop_task(current_state.task_id)
            self.active_task = None

    def handle_update_scheduler_start(
        self, progress_state: BenchmarkerTaskProgressState, result: BenchmarkerResult
    ):
        if self.active_task is not None:
            raise RuntimeError("Active task already set.")

        progress_state.strategy = result.current_strategy
        progress_state.started = True
        progress_state.start_time = (
            result.current_aggregator.scheduler_created_requests.start_time
        )
        progress_state.max_number = result.current_aggregator.max_number
        progress_state.max_duration = result.current_aggregator.max_duration

    def handle_update_scheduler_update(
        self, progress_state: BenchmarkerTaskProgressState, result: BenchmarkerResult
    ):
        if self.active_task is None:
            raise RuntimeError("Active task not set.")

        if self.active_task != progress_state.task_id:
            raise RuntimeError("Active task does not match current task.")

        progress_state.in_warmup = result.current_aggregator.in_warmup
        progress_state.in_cooldown = result.current_aggregator.in_cooldown
        progress_state.requests_rate = (
            result.current_aggregator.successful_requests.rate
        )
        progress_state.request_latency = result.current_aggregator.request_time.mean
        progress_state.requests_processing = (
            result.current_aggregator.scheduler_processing_requests.last
        )
        progress_state.requests_successful = (
            result.current_aggregator.successful_requests.total
        )
        progress_state.requests_incomplete = (
            result.current_aggregator.incomplete_requests.total
        )
        progress_state.requests_errored = (
            result.current_aggregator.errored_requests.total
        )

        progress_state.worker_overheads_time_ms = (
            result.current_aggregator.scheduled_time_delay.mean_ms
            + result.current_aggregator.worker_start_delay.mean_ms
        )
        progress_state.backend_overheads_time_ms = (
            result.current_aggregator.request_time_delay.mean_ms
        )
        progress_state.requests_sleep_time_ms = (
            result.current_aggregator.scheduled_time_sleep.mean_ms
        )
        progress_state.requests_targeted_start_time_delay_ms = (
            result.current_aggregator.request_start_time_targeted_delay.mean_ms
        )

    def handle_update_scheduler_complete(
        self, progress_state: BenchmarkerTaskProgressState, result: BenchmarkerResult
    ):
        if self.active_task is None:
            raise RuntimeError("Active task not set.")

        if self.active_task != progress_state.task_id:
            raise RuntimeError("Active task does not match current task.")

        progress_state.in_warmup = False
        progress_state.in_cooldown = False
        progress_state.compiling = True

    def handle_update_benchmark_compiled(
        self, progress_state: BenchmarkerTaskProgressState, result: BenchmarkerResult
    ):
        if self.active_task is None:
            raise RuntimeError("Active task not set.")

        if self.active_task != progress_state.task_id:
            raise RuntimeError("Active task does not match current task.")

        progress_state.compiling = False
        progress_state.ended = True
        progress_state.requests_rate = (
            result.current_benchmark.requests_per_second.successful.mean
        )
        progress_state.request_latency = (
            result.current_benchmark.request_latency.successful.mean
        )
        progress_state.requests_processing = (
            result.current_benchmark.requests_concurrency.successful.mean
        )
        progress_state.requests_successful = result.current_benchmark.successful_total
        progress_state.requests_errored = result.current_benchmark.errored_total

    def handle_end(self, result: BenchmarkerResult):
        self.benchmarker_progress.update(
            self.progress_task,
            completed=len(self.benchmarker_tasks) * 1000,
            total=len(self.benchmarker_tasks) * 1000,
            completed_benchmarks=len(self.benchmarker_tasks),
            total_benchmarks=len(self.benchmarker_tasks),
        )
        self.benchmarker_progress.stop_task(self.progress_task)
        self.benchmarker_live.stop()
        self.active_task = None
        self.benchmarker_tasks = []
        self.progress_task = None

    def create_task_progress_columns(self) -> List[ProgressColumn]:
        columns = [
            TextColumn("[{task.fields[start_time]}]"),
            SpinnerColumn(style=Colors.PROGRESS),
            TaskProgressColumn(style=Colors.PROGRESS),
            TextColumn("{task.description}"),
            TextColumn("({task.fields[progress_status]})"),
            TextColumn(" "),
        ]

        if not self.display_scheduler_stats:
            columns += [
                TextColumn("{task.fields[requests_summary]}\n"),
            ]
        else:
            columns += [
                TextColumn(
                    "{task.fields[requests_summary]}\n{task.fields[scheduler_stats]}\n"
                ),
            ]

        return columns

    def create_task_progress_state(
        self,
        task_id: TaskID,
        index: int,
        strategy_type: StrategyType,
        result: BenchmarkerResult,
    ) -> BenchmarkerTaskProgressState:
        return BenchmarkerTaskProgressState(
            display_scheduler_stats=self.display_scheduler_stats,
            task_id=task_id,
            strategy=strategy_type,
        )


class GenerativeTextBenchmarkerProgressDisplay(
    BenchmarkerProgressDisplay[GenerativeTextBenchmarkerTaskProgressState]
):
    def handle_update_scheduler_update(self, progress_state, result):
        super().handle_update_scheduler_update(progress_state, result)
        progress_state.output_tokens = result.current_aggregator.output_tokens.mean
        progress_state.prompt_tokens = result.current_aggregator.prompt_tokens.mean
        progress_state.output_tokens_rate = result.current_aggregator.output_tokens.rate
        progress_state.total_tokens_rate = result.current_aggregator.total_tokens.rate
        progress_state.tokens_ttft = result.current_aggregator.time_to_first_token.mean
        progress_state.tokens_itl = result.current_aggregator.inter_token_latency.mean

    def handle_update_benchmark_compiled(self, progress_state, result):
        super().handle_update_benchmark_compiled(progress_state, result)

        progress_state.output_tokens = (
            result.current_benchmark.output_token_count.successful.mean
        )
        progress_state.prompt_tokens = (
            result.current_benchmark.prompt_token_count.successful.mean
        )
        progress_state.output_tokens_rate = (
            result.current_benchmark.output_tokens_per_second.successful.mean
        )
        progress_state.total_tokens_rate = (
            result.current_benchmark.tokens_per_second.successful.mean
        )
        progress_state.tokens_ttft = (
            result.current_benchmark.time_to_first_token_ms.successful.mean
        )
        progress_state.tokens_itl = (
            result.current_benchmark.inter_token_latency_ms.successful.mean
        )

    def create_task_progress_state(
        self,
        task_id: TaskID,
        index: int,
        strategy_type: StrategyType,
        result: BenchmarkerResult,
    ) -> GenerativeTextBenchmarkerTaskProgressState:
        return GenerativeTextBenchmarkerTaskProgressState(
            display_scheduler_stats=self.display_scheduler_stats,
            task_id=task_id,
            strategy=strategy_type,
        )

    def create_task_progress_columns(self) -> List[ProgressColumn]:
        columns = super().create_task_progress_columns()
        columns = columns[:-1]  # remove the last display info column

        if not self.display_scheduler_stats:
            columns += [
                TextColumn(
                    "{task.fields[requests_summary]}\n{task.fields[tokens_summary]}",
                ),
            ]
        else:
            columns += [
                TextColumn(
                    "{task.fields[requests_summary]}\n{task.fields[tokens_summary]}\n{task.fields[scheduler_stats]}",
                ),
            ]

        return columns
