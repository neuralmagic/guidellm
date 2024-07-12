from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from guidellm.core.distribution import Distribution
from guidellm.core.request import TextGenerationRequest

__all__ = [
    "TextGenerationResult",
    "TextGenerationError",
    "TextGenerationBenchmark",
    "TextGenerationBenchmarkReport",
    "RequestConcurrencyMeasurement",
]


class TextGenerationResult:
    """
    A class to represent the result of a text generation request
    for generative AI workloads.

    :param request: The text generation request that generated this result.
    :type request: TextGenerationRequest
    """

    def __init__(self, request: TextGenerationRequest):
        """
        Initialize the TextGenerationResult with the given text generation request.

        :param request: The text generation request that generated this result.
        :type request: TextGenerationRequest
        """
        self._request = request
        self._prompt = ""
        self._prompt_word_count = 0
        self._prompt_token_count = 0
        self._output = ""
        self._output_word_count = 0
        self._output_token_count = 0
        self._last_time: Optional[float] = None
        self._first_token_set: bool = False
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._decode_times = Distribution()

        logger.debug(f"Initialized TextGenerationResult for request: {self._request}")

    def __repr__(self) -> str:
        return (
            f"TextGenerationResult("
            f"request_id={self._request.id}, "
            f"prompt='{self._prompt}', "
            f"output='{self._output}', "
            f"start_time={self._start_time}, "
            f"end_time={self._end_time}, "
            f"first_token_time={self._first_token_time}, "
            f"decode_times={self._decode_times})"
        )

    def __str__(self) -> str:
        return (
            f"TextGenerationResult("
            f"request_id={self._request.id}, "
            f"prompt='{self._prompt}', "
            f"output='{self._output}', "
            f"start_time={self._start_time}, "
            f"end_time={self._end_time})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two TextGenerationResult instances.

        :param other: Another instance of TextGenerationResult.
        :type other: TextGenerationResult
        :return: True if the instances are equal, False otherwise.
        :rtype: bool
        """

        if not isinstance(other, TextGenerationResult):
            raise NotImplementedError(
                "Only TextGenerationResult type could be used in that operation"
            )

        return (
            self._request == other._request
            and self._prompt == other._prompt
            and self._output == other._output
            and self._start_time == other._start_time
            and self._end_time == other._end_time
            and self._first_token_time == other._first_token_time
            and self._decode_times == other._decode_times
        )

    @property
    def request(self) -> TextGenerationRequest:
        """
        Get the text generation request associated with this result.

        :return: The text generation request.
        :rtype: TextGenerationRequest
        """
        return self._request

    @property
    def prompt(self) -> str:
        """
        Get the prompt used in the text generation.

        :return: The prompt.
        :rtype: str
        """
        return self._prompt

    @property
    def output(self) -> str:
        """
        Get the generated output from the text generation.

        :return: The generated output.
        :rtype: str
        """
        return self._output

    @property
    def start_time(self) -> float:
        """
        Get the start time of the text generation.

        :return: The start time.
        :rtype: float
        """

        self._recording_started()
        assert self._start_time

        return self._start_time

    @property
    def end_time(self) -> float:
        """
        Get the end time of the text generation.

        :return: The end time.
        :rtype: float
        """

        self._recording_started()
        assert self._end_time

        return self._end_time

    @property
    def first_token_time(self) -> Optional[float]:
        """
        Get the time taken to generate the first token.

        :return: The time taken to generate the first token.
        :rtype: Optional[float]
        """
        return self._first_token_time

    @property
    def decode_times(self) -> Distribution:
        """
        Get the decode times for each token in the text generation.

        :return: The decode times.
        :rtype: Distribution
        """
        return self._decode_times

    def start(self, prompt: str):
        """
        Start the text generation by recording the prompt and start time.

        :param prompt: The input prompt for the text generation.
        :type prompt: str
        """
        self._prompt = prompt
        self._prompt_word_count = len(prompt.split())
        self._prompt_token_count = len(prompt)  # Token count placeholder
        self._start_time = time()
        self._last_time = perf_counter()
        self._first_token_set = False

        logger.info(f"Text generation started with prompt: '{prompt}'")

    def _recording_started(self, raise_exception: bool = True) -> bool:
        """
        Ensure that the benchmark text generation recording is started.

        We can assume that if the `self.start_time` & `self.end_time` exist
        then the `start()` has been called.
        """

        if (self.start_time is not None) and (self.end_time is not None):
            return True
        else:
            if raise_exception is True:
                raise ValueError(
                    "Last time is not specified. "
                    "Did you make the `text_generation_benchmark.start()`?"
                )
            else:
                return False

    def output_token(self, token: str):
        """
        Add a token to the output and record the decode time.

        :param token: The decoded token.
        :type token: str
        """

        current_counter = perf_counter()

        if not self._first_token_set:
            self._first_token_time = current_counter - self.end_time
            self._first_token_set = True
            logger.debug(f"First token decode time: {self._first_token_time}")
        else:
            decode_time = current_counter - self.end_time
            self._decode_times.add_data([decode_time])
            logger.debug(f"Token '{token}' decoded in {decode_time} seconds")

        self._last_time = current_counter
        self._output += f"{token} "

    def end(
        self,
        output: str,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
    ):
        """
        End the text generation by recording the output and end time.

        :param output: The generated output for the text generation.
        :type output: str
        :param prompt_token_count: Optional token count for the prompt,
            defaults to word count.
        :type prompt_token_count: Optional[int]
        :param output_token_count: Optional token count for the output,
            defaults to word count.
        :type output_token_count: Optional[int]
        """

        self._end_time = time()
        self._output_word_count = len(self.output.split())
        self._output_token_count = output_token_count or self._output_word_count
        self._prompt_token_count = prompt_token_count or self._prompt_word_count

        logger.info(f"Text generation ended with output: '{output}'")


class TextGenerationError:
    """
    A class to represent an error that occurred during a text generation request
    for generative AI workloads.

    :param request: The text generation request that generated this error.
    :type request: TextGenerationRequest
    :param error: The exception that occurred during the text generation.
    :type error: Exception
    """

    def __init__(self, request: TextGenerationRequest, error: Exception):
        """
        Initialize the TextGenerationError with a unique identifier.

        :param request: The text generation request that generated this error.
        :type request: TextGenerationRequest
        :param error: The exception that occurred during the text generation.
        :type error: Exception
        """
        self._request = request
        self._error = error

        logger.error(f"Error occurred for request: {self._request}: {error}")

    def __repr__(self) -> str:
        """
        Return a string representation of the TextGenerationError.

        :return: String representation of the TextGenerationError.
        :rtype: str
        """
        return f"TextGenerationError(request={self._request}, error={self._error})"

    @property
    def request(self) -> TextGenerationRequest:
        """
        Get the text generation request associated with this error.

        :return: The text generation request.
        :rtype: TextGenerationRequest
        """
        return self._request

    @property
    def error(self) -> Exception:
        """
        Get the exception that occurred during the text generation.

        :return: The exception.
        :rtype: Exception
        """
        return self._error


@dataclass
class RequestConcurrencyMeasurement:
    """
    A dataclass to represent the concurrency measurement of a request.

    :param time: The time at which the measurement was taken.
    :type time: float
    :param completed: The number of completed requests.
    :type completed: int
    :param errored: The number of errored requests.
    :type errored: int
    :param processing: The number of requests currently being processed.
    :type processing: int
    """

    time: float
    completed: int
    errored: int
    processing: int


class TextGenerationBenchmark:
    def __init__(self, mode: str, rate: Optional[float]):
        """
        Initialize the TextGenerationBenchmark.

        :param mode: The mode of the result.
        :type mode: str
        :param rate: The rate of requests.
        :type rate: Optional[float]
        """
        self._mode = mode
        self._rate = rate
        self._results: List[TextGenerationResult] = []
        self._errors: List[TextGenerationError] = []
        self._concurrencies: List[RequestConcurrencyMeasurement] = []
        self._overloaded = False
        self._args_rate: Optional[float] = None

        logger.debug(
            f"Initialized TextGenerationBenchmark with mode={mode} and rate={rate}"
        )

    def __repr__(self) -> str:
        return (
            f"TextGenerationBenchmark("
            f"mode={self._mode}, "
            f"rate={self._rate}, "
            f"results={self._results}, "
            f"errors={self._errors}, "
            f"concurrencies={self._concurrencies})"
        )

    def __str__(self) -> str:
        return (
            f"TextGenerationBenchmark("
            f"mode={self._mode}, "
            f"rate={self._rate}, "
            f"request_count={self.request_count}, "
            f"error_count={self.error_count}, "
            f"request_rate={self.request_rate})"
        )

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between two TextGenerationBenchmark instances.

        :param other: Another instance of TextGenerationBenchmark.
        :type other: TextGenerationBenchmark
        :return: True if the instances are equal, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, TextGenerationBenchmark):
            raise TypeError(f"Operations only with {type(self)} are allowed.")
        else:
            return (
                self._mode == other._mode
                and self._rate == other._rate
                and self._results == other._results
                and self._errors == other._errors
                and self._concurrencies == other._concurrencies
            )

    def __iter__(self):
        """
        Provide an iterator interface to iterate over the results.

        :return: An iterator over the results.
        """
        return iter(self._results)

    @property
    def overloaded(self) -> bool:
        """
        Get the overloaded state of the result.

        :return: The overloaded state.
        :rtype: bool
        """
        return self._overloaded

    @property
    def mode(self) -> str:
        """
        Get the mode of the result.

        :return: The mode.
        :rtype: str
        """
        return self._mode

    @property
    def args_rate(self) -> Optional[float]:
        """
        Get the args rate of the result.

        :return: The args rate.
        :rtype: Optional[float]
        """
        return self._args_rate

    @property
    def rate(self) -> Optional[float]:
        """
        Get the rate of requests in the result.

        :return: The rate of requests.
        :rtype: Optional[float]
        """
        return self._rate

    @property
    def results(self) -> List[TextGenerationResult]:
        """
        Get the list of results in the result.

        :return: The list of results.
        :rtype: List[TextGenerationResult]
        """
        return self._results

    @property
    def errors(self) -> List[TextGenerationError]:
        """
        Get the list of errors in the result.

        :return: The list of errors.
        :rtype: List[TextGenerationError]
        """
        return self._errors

    @property
    def concurrencies(self) -> List[RequestConcurrencyMeasurement]:
        """
        Get the list of concurrency measurements in the result.

        :return: The list of concurrency measurements.
        :rtype: List[RequestConcurrencyMeasurement]
        """
        return self._concurrencies

    @property
    def request_count(self) -> int:
        """
        Get the number of requests in the result.

        :return: The number of requests.
        :rtype: int
        """
        return len(self._results)

    @property
    def error_count(self) -> int:
        """
        Get the number of errors in the result.

        :return: The number of errors.
        :rtype: int
        """
        return len(self._errors)

    @property
    def request_rate(self) -> float:
        """
        Get the rate of requests per second in the result.

        :return: The rate of requests per second.
        :rtype: float
        """

        if not self._results:
            return 0.0
        else:
            return self.request_count / (
                self._results[-1].end_time - self._results[0].start_time
            )

    def request_started(self):
        """
        Record the start of a generation request.
        """
        if not self._concurrencies:
            self._concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(), completed=0, errored=0, processing=1
                )
            )
        else:
            last = self._concurrencies[-1]
            self._concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=last.completed,
                    errored=last.errored,
                    processing=last.processing + 1,
                )
            )

        logger.info("Text generation request started")

    def request_completed(
        self, result: Union[TextGenerationResult, TextGenerationError]
    ):
        """
        Record the completion of a text generation request.

        :param result: The completed result or error.
        :type result: Union[TextGenerationResult, TextGenerationError]
        """
        if isinstance(result, TextGenerationError):
            self._errors.append(result)
            last = self._concurrencies[-1]
            self._concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=last.completed,
                    errored=last.errored + 1,
                    processing=last.processing - 1,
                )
            )
            logger.info(f"Text generation request resulted in error: {result}")
        else:
            self._results.append(result)
            last = self._concurrencies[-1]
            self._concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=last.completed + 1,
                    errored=last.errored,
                    processing=last.processing - 1,
                )
            )
            logger.info(f"Text generation request completed successfully: {result}")


class TextGenerationBenchmarkReport:
    """
    A class to represent a report of text generation benchmarks
    for generative AI workloads.
    """

    def __init__(self):
        """
        Initialize the TextGenerationBenchmarkReport.
        """
        self._benchmarks: List[TextGenerationBenchmark] = []
        self._args: List[Dict[str, Any]] = []

        logger.debug("Initialized TextGenerationBenchmarkReport")

    def __repr__(self) -> str:
        return (
            f"TextGenerationBenchmarkReport("
            f"benchmarks={self._benchmarks}, "
            f"args={self._args})"
        )

    def __str__(self) -> str:
        return (
            f"TextGenerationBenchmarkReport("
            f"args={self._args}, "
            f"benchmarks_summary=[{', '.join(str(b) for b in self._benchmarks)}])"
        )

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between two TextGenerationBenchmarkReport instances.

        :param other: Another instance of TextGenerationBenchmarkReport.
        :type other: TextGenerationBenchmarkReport
        :return: True if the instances are equal, False otherwise.
        :rtype: bool
        """

        if not isinstance(other, TextGenerationBenchmarkReport):
            raise TypeError(f"Operations only with {type(self)} are allowed.")

        return self._benchmarks == other._benchmarks and self._args == other._args

    def __iter__(self):
        return iter(self._benchmarks)

    @property
    def benchmarks(self) -> List[TextGenerationBenchmark]:
        """
        Get the list of benchmarks.

        :return: The list of benchmarks.
        :rtype: List[TextGenerationBenchmark]
        """
        return self._benchmarks

    @property
    def args(self) -> List[Dict[str, Any]]:
        """
        Get the list of arguments.

        :return: The list of arguments.
        :rtype: List[Dict[str, Any]]
        """
        return self._args

    @property
    def benchmarks_sorted(self) -> List[TextGenerationBenchmark]:
        """
        Get the list of benchmarks sorted by request rate.

        :return: The sorted list of benchmarks.
        :rtype: List[TextGenerationBenchmark]
        """
        benchmarks = sorted(self._benchmarks, key=lambda x: x.request_rate)
        return benchmarks

    def add_benchmark(self, benchmark: TextGenerationBenchmark):
        """
        Add a result to the report.

        :param benchmark: The result to add.
        :type benchmark: TextGenerationBenchmark
        """
        self._benchmarks.append(benchmark)
        logger.debug(f"Added result: {benchmark}")

    def to_dict(self) -> Dict[str, Any]:
        return {}
