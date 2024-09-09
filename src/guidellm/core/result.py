from time import time
from typing import Any, Dict, List, Literal, Optional, Union

from loguru import logger
from pydantic import Field, computed_field

from guidellm.core.distribution import Distribution
from guidellm.core.request import TextGenerationRequest
from guidellm.core.serializable import Serializable

__all__ = [
    "RequestConcurrencyMeasurement",
    "TextGenerationBenchmark",
    "TextGenerationBenchmarkReport",
    "TextGenerationError",
    "TextGenerationResult",
]


class TextGenerationResult(Serializable):
    """
    A class to represent the result of a text generation request
    for generative AI workloads.
    """

    request: TextGenerationRequest = Field(
        description="The text generation request used to generate the result.",
    )
    prompt: str = Field(
        default_factory=str,
        description="The input prompt for the text generation.",
    )
    prompt_word_count: int = Field(
        default=0,
        description="The number of words in the input prompt.",
    )
    prompt_token_count: int = Field(
        default=0,
        description="The number of tokens in the input prompt.",
    )
    output: str = Field(
        default_factory=str,
        description="The generated output for the text generation.",
    )
    output_word_count: int = Field(
        default=0,
        description="The number of words in the output.",
    )
    output_token_count: int = Field(
        default=0,
        description="The number of tokens in the output.",
    )
    last_time: Optional[float] = Field(
        default=None,
        description="The last time recorded.",
    )
    first_token_set: bool = Field(
        default=False,
        description="Whether the first token time is set.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="The start time of the text generation.",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="The end time of the text generation.",
    )
    first_token_time: Optional[float] = Field(
        default=None,
        description="The time taken to decode the first token.",
    )
    decode_times: Distribution = Field(
        default_factory=Distribution,
        description="The distribution of decode times.",
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
        self._check_recording_started()

        if self.last_time is None:
            raise ValueError(
                "last time is not specified. "
                "Did you call `text_generation_benchmark.start()`?"
            )

        current_counter = time()

        if not self.first_token_set:
            self.first_token_time = current_counter - self.last_time
            self.first_token_set = True
            logger.debug(f"First token decode time: {self.first_token_time}")
        else:
            decode_time = current_counter - self.last_time
            self.decode_times.add_data([decode_time])
            logger.debug(f"Token '{token}' decoded in {decode_time} seconds")

        self.last_time = current_counter
        self.output += token
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
        self._check_recording_started()
        self.end_time = time()

        if output:
            self.output = output

        self.output_word_count = len(self.output.split())
        self.output_token_count = output_token_count or self.output_word_count
        self.prompt_token_count = prompt_token_count or self.prompt_word_count

        logger.info(f"Text generation ended with output: '{self.output}'")

    def _check_recording_started(
        self,
    ):
        if self.start_time is None:
            raise ValueError(
                "start time is not specified. "
                "Did you make the `text_generation_benchmark.start()`?",
            )


class TextGenerationError(Serializable):
    """
    A class to represent an error that occurred during a text generation request
    for generative AI workloads.
    """

    request: TextGenerationRequest = Field(
        description="The text generation request that resulted in an error.",
    )
    message: str = Field(
        description="The error message that occurred during text generation.",
    )


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
    A class to represent a report of text generation requests
    (results and errors) for generative AI workloads.
    This is a set of results and errors for a specific mode and rate.
    """

    mode: Literal["asynchronous", "synchronous", "throughput"] = Field(
        description="The generation mode, one of 'async', 'sync', or 'throughput'."
    )
    rate: Optional[float] = Field(
        default=None,
        description="The requested rate of requests per second.",
    )
    results: List[TextGenerationResult] = Field(
        default_factory=list,
        description="The results of the text generation requests.",
    )
    errors: List[TextGenerationError] = Field(
        default_factory=list,
        description="The errors of the text generation requests.",
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

    @computed_field # type: ignore[misc]
    @property
    def request_count(self) -> int:
        """
        Get the number of requests in the result.

        :return: The number of requests.
        :rtype: int
        """
        return len(self.results)

    @computed_field # type: ignore[misc]
    @property
    def error_count(self) -> int:
        """
        Get the number of errors in the result.

        :return: The number of errors.
        :rtype: int
        """
        return len(self.errors)

    @computed_field # type: ignore[misc]
    @property
    def total_count(self) -> int:
        """
        Get the total number of requests in the result.

        :return: The total number of requests.
        :rtype: int
        """
        return self.request_count + self.error_count

    @computed_field # type: ignore[misc]
    @property
    def start_time(self) -> Optional[float]:
        """
        Get the start time of the first request in the result.

        :return: The start time of the first request.
        :rtype: Optional[float]
        """
        if not self.results:
            return None

        return self.results[0].start_time

    @computed_field # type: ignore[misc]
    @property
    def end_time(self) -> Optional[float]:
        """
        Get the end time of the last request in the result.

        :return: The end time of the last request.
        :rtype: Optional[float]
        """
        if not self.results:
            return None

        return self.results[-1].end_time

    @computed_field # type: ignore[misc]
    @property
    def duration(self) -> float:
        """
        Get the duration of the result in seconds.

        :return: The duration of the result.
        :rtype: float
        """
        if not self.results or not self.start_time or not self.end_time:
            return 0.0

        return self.end_time - self.start_time

    @computed_field # type: ignore[misc]
    @property
    def completed_request_rate(self) -> float:
        """
        Get the rate of requests per second in the result.

        :return: The rate of requests per second.
        :rtype: float
        """
        if not self.results or not self.duration:
            return 0.0

        return len(self.results) / self.duration

    @computed_field # type: ignore[misc]
    @property
    def request_latency(self) -> float:
        """
        Get the average request latency in seconds.

        :return: The average request latency in seconds.
        :rtype: float
        """
        if not self.results:
            return 0.0

        return self.request_latency_distribution.mean

    @property
    def request_latency_distribution(self) -> Distribution:
        """
        Get the distribution of request latencies.

        :return: The distribution of request latencies.
        :rtype: Distribution
        """
        return Distribution(
            data=[
                result.end_time - result.start_time
                for result in self.results
                if result.end_time is not None and result.start_time is not None
            ]
        )

    @computed_field # type: ignore[misc]
    @property
    def request_latency_percentiles(self) -> List[float]:
        """
        Get standard percentiles of request latency in seconds.

        :return: List of percentile request latency in seconds
        :rtype: List[float]
        """
        return self.request_latency_distribution.percentiles([1, 5, 10, 50, 90, 95, 99])


    @computed_field # type: ignore[misc]
    @property
    def time_to_first_token(self) -> float:
        """
        Get the time taken to decode the first token in milliseconds.

        :return: The time taken to decode the first token in milliseconds.
        :rtype: float
        """
        if not self.results:
            return 0.0

        return 1000 * self.ttft_distribution.mean

    @property
    def ttft_distribution(self) -> Distribution:
        """
        Get the distribution of time taken to decode the first token.

        :return: The distribution of time taken to decode the first token.
        :rtype: Distribution
        """
        return Distribution(
            data=[
                result.first_token_time
                for result in self.results
                if result.first_token_time is not None
            ]
        )

    @computed_field # type: ignore[misc]
    @property
    def time_to_first_token_percentiles(self) -> List[float]:
        """
        Get standard percentiles for time taken to decode the first token
        in milliseconds.

        :return: List of percentile time taken to decode the first token
        in milliseconds.
        :rtype: List[float]
        """
        return self.ttft_distribution.percentiles([1, 5, 10, 50, 90, 95, 99])

    @computed_field # type: ignore[misc]
    @property
    def inter_token_latency(self) -> float:
        """
        Get the average time between tokens in milliseconds.

        :return: The average time between tokens.
        :rtype: float
        """
        if not self.results:
            return 0.0

        return 1000 * self.itl_distribution.mean

    @property
    def itl_distribution(self) -> Distribution:
        """
        Get the distribution of time between tokens.

        :return: The distribution of time between tokens.
        :rtype: Distribution
        """
        return Distribution(
            data=[
                decode for result in self.results for decode in result.decode_times.data
            ]
        )

    @computed_field # type: ignore[misc]
    @property
    def inter_token_latency_percentiles(self) -> List[float]:
        """
        Get standard percentiles for the time between tokens in milliseconds.

        :return: List of percentiles for the average time between tokens.
        :rtype: List[float]
        """
        return self.itl_distribution.percentiles([1, 5, 10, 50, 90, 95, 99])

    @computed_field # type: ignore[misc]
    @property
    def output_token_throughput(self) -> float:
        """
        Get the average token throughput in tokens per second.

        :return: The average token throughput.
        :rtype: float
        """
        if not self.results or not self.duration:
            return 0.0

        total_tokens = sum(result.output_token_count for result in self.results)

        return total_tokens / self.duration

    @computed_field # type: ignore[misc]
    @property
    def prompt_token(self) -> float:
        """
        Get the average number of prompt tokens.

        :return: The average number of prompt tokens.
        :rtype: float
        """
        return self.prompt_token_distribution.mean

    @property
    def prompt_token_distribution(self) -> Distribution:
        """
        Get the distribution of prompt token counts.

        :return: The distribution of prompt token counts.
        :rtype: Distribution
        """
        return Distribution(data=[result.prompt_token_count for result in self.results])

    @computed_field # type: ignore[misc]
    @property
    def prompt_token_percentiles(self) -> List[float]:
        """
        Get standard percentiles for number of prompt tokens.

        :return: List of percentiles of number of prompt tokens.
        :rtype: List[float]
        """
        return self.prompt_token_distribution.percentiles([1, 5, 50, 95, 99])

    @computed_field # type: ignore[misc]
    @property
    def output_token(self) -> float:
        """
        Get the average number of output tokens.

        :return: The average number of output tokens.
        :rtype: float
        """
        return self.output_token_distribution.mean

    @property
    def output_token_distribution(self) -> Distribution:
        """
        Get the distribution of output token counts.

        :return: The distribution of output token counts.
        :rtype: Distribution
        """
        return Distribution(data=[result.output_token_count for result in self.results])

    @computed_field # type: ignore[misc]
    @property
    def output_token_percentiles(self) -> List[float]:
        """
        Get standard percentiles for number of output tokens.

        :return: List of percentiles of number of output tokens.
        :rtype: List[float]
        """
        return self.output_token_distribution.percentiles([1, 5, 50, 95, 99])

    @computed_field # type: ignore[misc]
    @property
    def overloaded(self) -> bool:
        if (
            self.rate is None
            or not self.results
            or not self.concurrencies
            or len(self.concurrencies) < 2  # noqa: PLR2004
        ):
            # if rate was not set, sync mode is assumed,
            # or we have less than 2 data points,
            # then we cannot be overloaded by definition
            return False

        # if the calculated rate is less than 75% of the requested rate,
        # safe to assume the system is overloaded
        return self.completed_request_rate < 0.75 * self.rate

    def request_started(self):
        """
        Record the start of a generation request.
        """
        if not self.concurrencies:
            self.concurrencies = [
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=0,
                    errored=0,
                    processing=1,
                ),
            ]
        else:
            last = self.concurrencies[-1]
            self.concurrencies.append(
                RequestConcurrencyMeasurement(
                    time=time(),
                    completed=last.completed,
                    errored=last.errored,
                    processing=last.processing + 1,
                ),
            )

        logger.info("Text generation request started")

    def request_completed(
        self,
        result: Union[TextGenerationResult, TextGenerationError],
    ):
        """
        Record the completion of a text generation request.

        :param result: The completed result or error.
        :type result: Union[TextGenerationResult, TextGenerationError]
        """
        if not self.concurrencies:
            raise ValueError("Request completed without starting")

        if isinstance(result, TextGenerationError):
            is_error = True
            self.errors.append(result)
            logger.info(
                "Text generation request resulted in error: {}",
                result.message,
            )
        else:
            if not result.start_time or not result.end_time:
                raise ValueError("Start time and End time are not defined")

            is_error = False
            self.results.append(result)
            logger.info("Text generation request completed successfully: {}", result)

        last = self.concurrencies[-1]
        self.concurrencies.append(
            RequestConcurrencyMeasurement(
                time=time(),
                completed=last.completed + (not is_error),
                errored=last.errored + is_error,
                processing=last.processing - 1,
            )
        )


class TextGenerationBenchmarkReport(Serializable):
    """
    A class to represent a report of text generation benchmarks
    for generative AI workloads.
    This is a collection of benchmarks for different modes and rates.
    """

    benchmarks: List[TextGenerationBenchmark] = Field(
        default_factory=list,
        description="The benchmarks of text generation requests.",
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="The arguments used for the benchmarks.",
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
        return sorted(self.benchmarks, key=lambda x: x.completed_request_rate)

    def add_benchmark(self, benchmark: TextGenerationBenchmark):
        """
        Add a result to the report.

        :param benchmark: The result to add.
        :type benchmark: TextGenerationBenchmark
        """
        self.benchmarks.append(benchmark)
        logger.debug("Added result: {}", benchmark)
