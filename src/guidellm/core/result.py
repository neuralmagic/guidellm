from time import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from guidellm.core.distribution import Distribution
from guidellm.core.request import TextGenerationRequest
from guidellm.core.serializable import Serializable

__all__ = [
    "TextGenerationResult",
    "TextGenerationError",
    "TextGenerationBenchmark",
    "TextGenerationBenchmarkReport",
    "RequestConcurrencyMeasurement",
]


class TextGenerationResult(Serializable):
    """
    A class to represent the result of a text generation request
    for generative AI workloads.
    """

    request: TextGenerationRequest
    prompt: str = ""
    prompt_word_count: int = 0
    prompt_token_count: int = 0
    output: str = ""
    output_word_count: int = 0
    output_token_count: int = 0
    last_time: Optional[float] = None
    first_token_set: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    first_token_time: Optional[float] = None
    decode_times: Distribution = Distribution()

    def start(self, prompt: str):
        """
        Start the text generation by recording the prompt and start time.

        :param prompt: The input prompt for the text generation.
        :type prompt: str
        """
        self.prompt = prompt
        self.prompt_word_count = len(prompt.split())
        self.prompt_token_count = len(prompt)  # Token count placeholder
        self.start_time = time()
        self.last_time = time()
        self.first_token_set = False

        logger.info("Text generation started with prompt: '{}'", prompt)

    def _recording_started(self, raise_exception: bool = True) -> bool:
        """
        Ensure that the benchmark text generation recording is started.

        We can assume that if the `self._start_time` exist,
        then the `start()` has been called.
        """

        if self._start_time is not None:
            return True
        else:
            if raise_exception is True:
                raise ValueError(
                    "start time is not specified. "
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
        current_counter = time()

        if not self._first_token_set:
            self.first_token_time = current_counter - self.last_time
            self.first_token_set = True
            logger.debug(f"First token decode time: {self._first_token_time}")
        else:
            decode_time = current_counter - self.last_time
            self._decode_times.add_data([decode_time])
            logger.debug(f"Token '{token}' decoded in {decode_time} seconds")

        self.last_time = current_counter
        self.output += f"{token} "
        logger.debug("Added token {} to output", token)

    def end(
        self,
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
        self.end_time = time()
        self.output_word_count = len(self.output.split())
        self.output_token_count = output_token_count or self._output_word_count
        self.prompt_token_count = prompt_token_count or self._prompt_word_count

        logger.info(f"Text generation ended with output: '{self.output}'")


class TextGenerationError(Serializable):
    """
    A class to represent an error that occurred during a text generation request
    for generative AI workloads.
    """

    request: TextGenerationRequest
    error: str

    def __init__(self, request: TextGenerationRequest, error: Exception):
        super().__init__(request=request, error=str(error))
        logger.error("Text generation error occurred: {}", error)


class RequestConcurrencyMeasurement(Serializable):
    """
    A dataclass to represent the concurrency measurement of a request.
    """

    time: float
    completed: int
    errored: int
    processing: int


class TextGenerationBenchmark(Serializable):
    """
    A class to represent a benchmark of text generation requests
    (results and errors) for generative AI workloads.
    This is a set of results and errors for a specific mode and rate.
    """

    mode: str
    rate: Optional[float]
    results: List[TextGenerationResult] = []
    errors: List[TextGenerationError] = []
    concurrencies: List[RequestConcurrencyMeasurement] = []

    def __iter__(self):
        """
        Provide an iterator interface to iterate over the results.

        :return: An iterator over the results.
        """
        return iter(self._results)

    @property
    def request_count(self) -> int:
        """
        Get the number of requests in the result.

        :return: The number of requests.
        :rtype: int
        """
        return len(self.results)

    @property
    def error_count(self) -> int:
        """
        Get the number of errors in the result.

        :return: The number of errors.
        :rtype: int
        """
        return len(self.errors)

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
        if not self.concurrencies:
            self.concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(), completed=0, errored=0, processing=1
                )
            )
        else:
            last = self.concurrencies[-1]
            self.concurrencies.append(
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
            self.errors.append(result)
            last = self.concurrencies[-1]
            self.concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=last.completed,
                    errored=last.errored + 1,
                    processing=last.processing - 1,
                )
            )
            logger.warning(
                "Text generation request resulted in error: {}", result.error
            )
        else:
            self.results.append(result)
            last = self.concurrencies[-1]
            self.concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=last.completed + 1,
                    errored=last.errored,
                    processing=last.processing - 1,
                )
            )
            logger.info("Text generation request completed successfully: {}", result)


class TextGenerationBenchmarkReport(Serializable):
    """
    A class to represent a report of text generation benchmarks
    for generative AI workloads.
    This is a collection of benchmarks for different modes and rates.
    """

    benchmarks: List[TextGenerationBenchmark] = []
    args: List[Dict[str, Any]] = []

    def __iter__(self):
        return iter(self.benchmarks)

    @property
    def benchmarks_sorted(self) -> List[TextGenerationBenchmark]:
        """
        Get the list of benchmarks sorted by request rate.

        :return: The sorted list of benchmarks.
        :rtype: List[TextGenerationBenchmark]
        """
        benchmarks = sorted(self.benchmarks, key=lambda x: x.request_rate)
        return benchmarks

    def add_benchmark(self, benchmark: TextGenerationBenchmark):
        """
        Add a result to the report.

        :param benchmark: The result to add.
        :type benchmark: TextGenerationBenchmark
        """
        self.benchmarks.append(benchmark)
        logger.debug("Added result: {}", benchmark)
