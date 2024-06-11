from typing import Optional, List, Union, Dict, Any
from time import time, perf_counter
from dataclasses import dataclass
from guidellm.core.distribution import Distribution
from guidellm.core.request import BenchmarkRequest


__all__ = [
    "BenchmarkResult",
    "BenchmarkError",
    "BenchmarkResultSet",
    "BenchmarkReport",
    "QueueMeasurement",
]


class BenchmarkResult:
    """
    A class to represent the result of a benchmark request for generative AI workloads.

    :param request: The benchmark request that generated this result.
    :type request: BenchmarkRequest
    """

    def __init__(self, request: BenchmarkRequest):
        """
        Initialize the BenchmarkResult with a unique identifier.

        :param request: The benchmark request that generated this result.
        :type request: BenchmarkRequest
        """
        self.request = request

        self.prompt = ""
        self.prompt_word_count = 0
        self.prompt_token_count = 0
        self.output = ""
        self.output_word_count = 0
        self.output_token_count = 0

        self._last_time: Optional[float] = None
        self._first_token_set: bool = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.decode_times = Distribution()

    def __str__(self) -> str:
        """
        Return a string representation of the BenchmarkResult.

        :return: String representation of the BenchmarkResult.
        :rtype: str
        """
        return (
            f"BenchmarkResult(request={self.request}, prompt='{self.prompt}', "
            f"output='{self.output}', start_time={self.start_time}, "
            f"end_time={self.end_time}, first_token_time={self.first_token_time})"
        )

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the BenchmarkResult for debugging.

        :return: Unambiguous string representation of the BenchmarkResult.
        :rtype: str
        """
        return (
            f"BenchmarkResult(request={self.request}, prompt='{self.prompt}', "
            f"prompt_word_count={self.prompt_word_count}, prompt_token_count={self.prompt_token_count}, "
            f"output='{self.output}', output_word_count={self.output_word_count}, "
            f"output_token_count={self.output_token_count}, start_time={self.start_time}, "
            f"end_time={self.end_time}, first_token_time={self.first_token_time}, "
            f"decode_times={self.decode_times})"
        )

    def __eq__(self, other: "BenchmarkResult") -> bool:
        """
        Check equality between two BenchmarkResult instances.

        :param other: Another instance of BenchmarkResult.
        :type other: BenchmarkResult
        :return: True if the instances are equal, False otherwise.
        :rtype: bool
        """
        return (
            self.request == other.request
            and self.prompt == other.prompt
            and self.output == other.output
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.first_token_time == other.first_token_time
            and self.decode_times == other.decode_times
        )

    def start(self, prompt: str):
        """
        Start the benchmark by recording the prompt and start time.

        :param prompt: The input prompt for the benchmark.
        :type prompt: str
        """
        self.prompt = prompt
        self.prompt_word_count = len(prompt.split())
        self.prompt_token_count = len(prompt)  # Token count placeholder
        self.start_time = time()
        self._last_time = perf_counter()
        self._first_token_set = False

    def output_token(self, token: str):
        """
        Add a token to the output and record the decode time.

        :param token: The decoded token.
        :type token: str
        """
        current_counter = perf_counter()

        if not self._first_token_set:
            self.first_token_time = current_counter - self._last_time
            self._first_token_set = True
        else:
            decode_time = current_counter - self._last_time
            self.decode_times.add_data([decode_time])

        self._last_time = current_counter
        self.output += f"{token} "

    def end(
        self,
        output: str,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
    ):
        """
        End the benchmark by recording the output and end time.

        :param output: The generated output for the benchmark.
        :type output: str
        :param prompt_token_count: Optional token count for the prompt, defaults to word count.
        :type prompt_token_count: Optional[int]
        :param output_token_count: Optional token count for the output, defaults to word count.
        :type output_token_count: Optional[int]
        """
        self.output = output
        self.end_time = time()
        self.output_word_count = len(output.split())
        self.output_token_count = (
            output_token_count
            if output_token_count is not None
            else self.output_word_count
        )
        self.prompt_token_count = (
            prompt_token_count
            if prompt_token_count is not None
            else self.prompt_word_count
        )


class BenchmarkError:
    """
    A class to represent an error that occurred during a benchmark request for generative AI workloads.

    :param id_: Unique identifier for the benchmark result.
    :type id_: str
    """

    def __init__(self, request: BenchmarkRequest, error: Exception):
        """
        Initialize the BenchmarkError with a unique identifier.

        :param request: The benchmark request that generated this error.
        :type request: BenchmarkRequest
        :param error: The exception that occurred during the benchmark.
        :type error: Exception
        """
        self.request = request
        self.error = error


@dataclass
class QueueMeasurement:
    time: float
    completed: int
    errored: int
    processing: int


class BenchmarkResultSet:
    def __init__(self, mode: str, rate: Optional[float]):
        self._mode = mode
        self._rate = rate
        self.benchmarks: List[BenchmarkResult] = []
        self.errors: List[BenchmarkError] = []
        self.concurrencies: List[QueueMeasurement] = []

    @property
    def args_mode(self) -> str:
        return self._mode

    @property
    def args_rate(self) -> Optional[float]:
        return self._rate

    @property
    def request_count(self) -> int:
        return len(self.benchmarks)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def request_rate(self) -> float:
        if not self.benchmarks:
            return 0.0

        start_time = self.benchmarks[0].start_time
        end_time = self.benchmarks[-1].end_time

        return self.request_count / (end_time - start_time)

    def benchmark_started(self):
        if not self.concurrencies:
            # Add initial measurement
            self.concurrencies.append(
                QueueMeasurement(time=time(), completed=0, errored=0, processing=1)
            )
        else:
            # Increment processing
            last = self.concurrencies[-1]
            self.concurrencies.append(
                QueueMeasurement(
                    time=time(),
                    completed=last.completed,
                    errored=last.errored,
                    processing=last.processing + 1,
                )
            )

    def benchmark_completed(self, benchmark: Union[BenchmarkResult, BenchmarkError]):
        if isinstance(benchmark, BenchmarkError):
            self.errors.append(benchmark)
            last = self.concurrencies[-1]
            self.concurrencies.append(
                QueueMeasurement(
                    time=time(),
                    completed=last.completed,
                    errored=last.errored + 1,
                    processing=last.processing - 1,
                )
            )
        else:
            self.benchmarks.append(benchmark)
            last = self.concurrencies[-1]
            self.concurrencies.append(
                QueueMeasurement(
                    time=time(),
                    completed=last.completed + 1,
                    errored=last.errored,
                    processing=last.processing - 1,
                )
            )


class BenchmarkReport:
    def __init__(self):
        self._benchmarks: List[BenchmarkResultSet] = []
        self._args: List[Dict[str, Any]] = []

    @property
    def benchmarks(self) -> List[BenchmarkResultSet]:
        return self._benchmarks

    @property
    def args(self) -> List[Dict[str, Any]]:
        return self._args

    @property
    def benchmarks_sorted(self) -> List[BenchmarkResultSet]:
        benchmarks = sorted(self._benchmarks, key=lambda x: x.request_rate)

        return benchmarks

    def add_benchmark(self, benchmark: BenchmarkResultSet):
        self._benchmarks.append(benchmark)
