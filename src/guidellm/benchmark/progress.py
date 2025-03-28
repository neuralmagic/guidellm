import math
import time
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

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
)

SCHEDULING_STRATEGY_DESCRIPTIONS: Dict[StrategyType, str] = {
    "synchronous": "synchronous",
    "concurrent": "concurrent@{RATE}",
    "throughput": "throughput",
    "constant": "constant@{RATE}",
    "poisson": "poisson@{RATE}",
}


@dataclass
class BenchmarkerTaskProgressState:
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
    requests_latency: float = 0
    requests_processing: int = 0
    requests_completed: int = 0
    requests_errored: int = 0

    output_tokens: float = 0
    prompt_tokens: float = 0
    output_tokens_rate: float = 0
    total_tokens_rate: float = 0
    tokens_ttft: float = 0
    tokens_itl: float = 0

    @property
    def description(self) -> str:
        if self.strategy in StrategyType.__args__:
            return SCHEDULING_STRATEGY_DESCRIPTIONS.get(
                self.strategy, self.strategy
            ).format(RATE="##" if self.strategy == "concurrent" else "#.##")

        rate = ""

        if hasattr(self.strategy, "streams"):
            rate = f"{self.strategy.streams:>2}"
        elif hasattr(self.strategy, "rate"):
            rate = f"{self.strategy.rate:.2f}"

        return SCHEDULING_STRATEGY_DESCRIPTIONS.get(
            self.strategy.type_, self.strategy.type_
        ).format(RATE=rate)

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

        number = self.requests_completed + self.requests_errored
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
        return {
            "start_time": self.formatted_start_time,
            "progress_status": self.formatted_progress_status,
            "requests_summary": self.formatted_requests_summary,
            "tokens_summary": self.formatted_tokens_summary,
        }

    @property
    def formatted_start_time(self) -> str:
        if self.start_time is None:
            return "--:--:--"

        return datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")

    @property
    def formatted_progress_status(self) -> str:
        if self.ended:
            status = "complete"
        elif self.compiling:
            status = "compiling"
        elif self.started and self.in_warmup:
            status = "warmup"
        elif self.started and self.in_cooldown:
            status = "cooldown"
        elif self.started:
            status = "running"
        else:
            status = "pending"

        return status.ljust(9)

    @property
    def formatted_requests_summary(self) -> str:
        if not self.started:
            return " "

        return (
            "Req: "
            f"({self.requests_rate:.2f} / sec, "
            f"{self.requests_latency:.2f}s Lat, "
            f"{self.requests_processing:>3.1f} Conc, "
            f"{self.requests_completed:>3} Comp, "
            f"{self.requests_errored:>3} Err)"
        )

    @property
    def formatted_tokens_summary(self) -> str:
        if not self.started:
            return " "

        return (
            "Tok: "
            f"({self.output_tokens_rate:.1f} gen/sec, "
            f"{self.total_tokens_rate:.1f} tot/sec, "
            f"{self.tokens_ttft:.1f}ms TTFT, "
            f"{self.tokens_itl:.1f}ms ITL, "
            f"{self.prompt_tokens:.0f} Prompt, "
            f"{self.output_tokens:.0f} Gen)"
        )


class BenchmarkerProgressDisplay(ABC):
    def __init__(self):
        """
        Progress display view:
        | Benchmarks -----------------------------------------------------------------|
        | [T]   N (S) Req: (#/sec, #sec, #proc, #com, #err) Tok: (#/sec, #TTFT, #ITL) |
        | [T] % N (S) Req: (#/sec, #sec, #proc, #com, #err) Tok: (#/sec, #TTFT, #ITL) |
        | ...                                                                         |
        | ----------------------------------------------------------------------------|
        SP Running... [BAR] (#/#) [ ELAPSED < ETA ]
        """
        self.started = False
        self.benchmarker_tasks_progress = Progress(*self.create_task_progress_columns())
        self.benchmarker_tasks_panel = Panel(
            self.benchmarker_tasks_progress,
            title="Benchmarks",
            title_align="left",
            expand=True,
        )
        self.benchmarker_progress = Progress(
            TextColumn("Generating..."),
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
        progress_state.start_time = result.current_aggregator.start_time
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
        progress_state.requests_rate = result.current_aggregator.successful_requests / (
            time.time() - progress_state.start_time
        )
        progress_state.requests_latency = result.current_aggregator.request_latency.mean
        progress_state.requests_processing = (
            result.current_aggregator.processing_requests
        )
        progress_state.requests_completed = (
            result.current_aggregator.successful_requests
        )
        progress_state.requests_errored = result.current_aggregator.errored_requests
        progress_state.output_tokens = result.current_aggregator.output_tokens.mean
        progress_state.prompt_tokens = result.current_aggregator.prompt_tokens.mean
        progress_state.output_tokens_rate = result.current_aggregator.output_tokens.rate
        progress_state.total_tokens_rate = result.current_aggregator.total_tokens.rate
        progress_state.tokens_ttft = result.current_aggregator.time_to_first_token.mean
        progress_state.tokens_itl = result.current_aggregator.inter_token_latency.mean

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
            result.current_benchmark.requests_per_second.completed.mean
        )
        progress_state.requests_latency = (
            result.current_benchmark.requests_latency.completed.mean
        )
        progress_state.requests_processing = (
            result.current_benchmark.requests_concurrency.completed.mean
        )
        progress_state.requests_completed = result.current_benchmark.completed_total
        progress_state.requests_errored = result.current_benchmark.errored_total
        progress_state.output_tokens = (
            result.current_benchmark.outputs_token_count.completed.mean
        )
        progress_state.prompt_tokens = (
            result.current_benchmark.prompts_token_count.completed.mean
        )
        progress_state.output_tokens_rate = (
            result.current_benchmark.outputs_tokens_per_second.completed.mean
        )
        progress_state.total_tokens_rate = (
            result.current_benchmark.tokens_per_second.completed.mean
        )
        progress_state.tokens_ttft = (
            result.current_benchmark.times_to_first_token_ms.completed.mean
        )
        progress_state.tokens_itl = (
            result.current_benchmark.inter_token_latencies_ms.completed.mean
        )

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
        return [
            TextColumn("[{task.fields[start_time]}]"),
            SpinnerColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
            TextColumn("({task.fields[progress_status]})"),
            TextColumn(" "),
            TextColumn("{task.fields[requests_summary]}"),
            TextColumn(" "),
            TextColumn("{task.fields[tokens_summary]}"),
        ]

    def create_task_progress_state(
        self,
        task_id: TaskID,
        index: int,
        strategy_type: StrategyType,
        result: BenchmarkerResult,
    ) -> BenchmarkerTaskProgressState:
        return BenchmarkerTaskProgressState(task_id=task_id, strategy=strategy_type)
