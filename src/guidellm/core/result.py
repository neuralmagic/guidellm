from time import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

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

    request: TextGenerationRequest = Field(
        description="The text generation request used to generate the result."
    )
    prompt: str = Field(
        default_factory=str, description="The input prompt for the text generation."
    )
    prompt_word_count: int = Field(
        default=0, description="The number of words in the input prompt."
    )
    prompt_token_count: int = Field(
        default=0, description="The number of tokens in the input prompt."
    )
    output: str = Field(
        default_factory=str, description="The generated output for the text generation."
    )
    output_word_count: int = Field(
        default=0, description="The number of words in the output."
    )
    output_token_count: int = Field(
        default=0, description="The number of tokens in the output."
    )
    last_time: Optional[float] = Field(
        default=None, description="The last time recorded."
    )
    first_token_set: bool = Field(
        default=False, description="Whether the first token time is set."
    )
    start_time: Optional[float] = Field(
        default=None, description="The start time of the text generation."
    )
    end_time: Optional[float] = Field(
        default=None, description="The end time of the text generation."
    )
    first_token_time: Optional[float] = Field(
        default=None, description="The time taken to decode the first token."
    )
    decode_times: Distribution = Field(
        default_factory=Distribution, description="The distribution of decode times."
    )

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

    def output_token(self, token: str):
        """
        Add a token to the output and record the decode time.

        :param token: The decoded token.
        :type token: str
        """
        current_counter = time()

        if not self.last_time:
            raise ValueError("Last time is not specified to get the output token.")

        if not self.first_token_set:
            self.first_token_time = current_counter - self.last_time
            self.first_token_set = True
            logger.debug(f"First token decode time: {self.first_token_time}")
        else:
            decode_time = current_counter - self.last_time
            self.decode_times.add_data([decode_time])
            logger.debug(f"Token '{token}' decoded in {decode_time} seconds")

        self.last_time = current_counter
        self.output += f"{token} "
        logger.debug("Added token {} to output", token)

    def end(
        self,
        output: Optional[str] = None,
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

        if output:
            self.output = output

        self.output_word_count = len(self.output.split())
        self.output_token_count = output_token_count or self.output_word_count
        self.prompt_token_count = prompt_token_count or self.prompt_word_count

        logger.info(f"Text generation ended with output: '{self.output}'")

    def _check_recording_started(self, raise_exception: bool = True) -> bool:
        """
        Ensure that the benchmark text generation recording is started.

        We can assume that if the `self._start_time` exist,
        then the `start()` has been called.
        """

        if self.start_time is not None:
            return True
        else:
            if raise_exception is True:
                raise ValueError(
                    "start time is not specified. "
                    "Did you make the `text_generation_benchmark.start()`?"
                )
            else:
                return False


class TextGenerationError(Serializable):
    """
    A class to represent an error that occurred during a text generation request
    for generative AI workloads.
    """

    request: TextGenerationRequest = Field(
        description="The text generation request that resulted in an error."
    )
    error: BaseException = Field(
        description="The error that occurred during text generation."
    )

    def __init__(self, request: TextGenerationRequest, error: BaseException):
        super().__init__(request=request, error=str(error))
        logger.error("Text generation error occurred: {}", error)


class RequestConcurrencyMeasurement(Serializable):
    """
    A dataclass to represent the concurrency measurement of a request.
    """

    time: float = Field(description="The time of the measurement.")
    completed: int = Field(description="The number of completed requests.")
    errored: int = Field(description="The number of errored requests.")
    processing: int = Field(description="The number of processing requests.")


class TextGenerationBenchmark(Serializable):
    """
    A class to represent a benchmark of text generation requests
    (results and errors) for generative AI workloads.
    This is a set of results and errors for a specific mode and rate.
    """

    mode: str = Field(description="The generation mode, either 'async' or 'sync'.")
    rate: Optional[float] = Field(
        default=None, description="The requested rate of requests per second."
    )
    results: List[TextGenerationResult] = Field(
        default_factory=list, description="The results of the text generation requests."
    )
    errors: List[TextGenerationError] = Field(
        default_factory=list, description="The errors of the text generation requests."
    )
    concurrencies: List[RequestConcurrencyMeasurement] = Field(
        default_factory=list,
        description="The concurrency measurements of the requests.",
    )

    def __iter__(self):
        """
        Provide an iterator interface to iterate over the results.

        :return: An iterator over the results.
        """
        return iter(self.results)

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
    def completed_request_rate(self) -> float:
        """
        Get the rate of requests per second in the result.

        :return: The rate of requests per second.
        :rtype: float
        """
        if not self.results:
            return 0.0
        else:
            if not self.results[0].start_time or not self.results[-1].end_time:
                raise ValueError("Start time and End time are not defined")

            return self.request_count / (
                self.results[-1].end_time - self.results[0].start_time
            )

    @property
    def overloaded(self) -> bool:
        if not self.results or not self.concurrencies:
            raise ValueError("No results or concurrencies to check for overload.")

        if self.rate is None or len(self.concurrencies) < 2:
            # if rate was not set, sync mode is assumed,
            # or we have less than 2 data points,
            # then we cannot be overloaded by definition
            return False

        if self.completed_request_rate < 0.60 * self.rate:
            # if the calculated rate is less than 60% of the requested rate,
            # safe to assume the system is overloaded
            return True

        # rate comparisons did not give a clear signal,
        # let's double check that we aren't overloaded by comparing the
        # compute throughput for the benchmark with the latency for the requests.
        # overall this means that a relatively flat or decreasing throughput curve
        # over time in addition to a growing processing queue is a sign of overload

        # TODO
        return False

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

    benchmarks: List[TextGenerationBenchmark] = Field(
        default_factory=list, description="The benchmarks of text generation requests."
    )
    args: List[Dict[str, Any]] = Field(
        default_factory=list, description="The arguments used for the benchmarks."
    )

    def __iter__(self):
        return iter(self.benchmarks)

    @property
    def benchmarks_sorted(self) -> List[TextGenerationBenchmark]:
        """
        Get the list of benchmarks sorted by request rate.

        :return: The sorted list of benchmarks.
        :rtype: List[TextGenerationBenchmark]
        """
        benchmarks = sorted(self.benchmarks, key=lambda x: x.completed_request_rate)
        return benchmarks

    def add_benchmark(self, benchmark: TextGenerationBenchmark):
        """
        Add a result to the report.

        :param benchmark: The result to add.
        :type benchmark: TextGenerationBenchmark
        """
        self.benchmarks.append(benchmark)
        logger.debug("Added result: {}", benchmark)
