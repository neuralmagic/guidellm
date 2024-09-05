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


DEFAULT_PERCENTILES = [1, 5, 10, 50, 90, 95, 99]


class TextGenerationResult(Serializable):
    """
    A class to represent the result of a text generation request
    for generative AI workloads.
    """

    request: TextGenerationRequest = Field(
        description="The text generation request used to generate the result.",
    )
    prompt_token_count: Optional[int] = Field(
        default=None,
        description="The number of tokens in the input prompt.",
    )
    output: str = Field(
        default_factory=str,
        description="The generated output for the text generation.",
    )
    output_token_count: Optional[int] = Field(
        default=None,
        description="The number of tokens in the output.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="The absolute start time, in seconds, of the text generation.",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="The absolute end time, in seconds, of the text generation.",
    )
    first_token_time: Optional[float] = Field(
        default=None,
        description="The absolute time, in seconds, the first token was received.",
    )
    last_token_time: Optional[float] = Field(
        default=None,
        description="The absolute time, in seconds, the last token was received.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> Optional[float]:
        """
        Get the request latency in seconds.

        :return: The request latency in seconds.
        """
        if not self.end_time or not self.start_time:
            return None

        return self.end_time - self.start_time

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token(self) -> Optional[float]:
        """
        Get the time taken to decode the first token in milliseconds.

        :return: The time taken to decode the first token in milliseconds.
        """
        if not self.first_token_time or not self.start_time:
            return None

        return 1000 * (self.first_token_time - self.start_time)

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency(self) -> Optional[float]:
        """
        Get the average time between tokens in milliseconds.

        :return: The average time between tokens.
        """
        if (
            not self.last_token_time
            or not self.first_token_time
            or not self.output_token_count
            or self.output_token_count < 2  # noqa: PLR2004
        ):
            return None

        return (
            1000
            * (self.last_token_time - self.first_token_time)
            / (self.output_token_count - 1)  # ignore first token
        )

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> Optional[float]:
        """
        Get the average token throughput in tokens per second for the entire request.
        Note, does not account for the time taken to decode the first token.

        :return: The average token throughput.
        """
        itl = self.inter_token_latency

        if itl is None:
            return None

        return 1000.0 / itl


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

    @computed_field  # type: ignore[misc]
    @property
    def request_count(self) -> int:
        """
        Get the number of requests in the result.

        :return: The number of requests.
        """
        return len(self.results)

    @computed_field  # type: ignore[misc]
    @property
    def error_count(self) -> int:
        """
        Get the number of errors in the result.

        :return: The number of errors.
        """
        return len(self.errors)

    @computed_field  # type: ignore[misc]
    @property
    def total_count(self) -> int:
        """
        Get the total number of requests in the result.

        :return: The total number of requests.
        """
        return self.request_count + self.error_count

    @computed_field  # type: ignore[misc]
    @property
    def start_time(self) -> Optional[float]:
        """
        Get the start time of the first request in the result.

        :return: The start time of the first request.
        """
        return self.results[0].start_time if self.results else None

    @computed_field  # type: ignore[misc]
    @property
    def end_time(self) -> Optional[float]:
        """
        Get the end time of the last request in the result.

        :return: The end time of the last request.
        """
        return self.results[-1].end_time if self.results else None

    @computed_field  # type: ignore[misc]
    @property
    def duration(self) -> float:
        """
        Get the duration of the result in seconds.

        :return: The duration of the result.
        """
        return (
            self.end_time - self.start_time
            if self.end_time and self.start_time
            else 0.0
        )

    @computed_field  # type: ignore[misc]
    @property
    def completed_request_rate(self) -> float:
        """
        Get the rate of requests per second in the result.

        :return: The rate of requests per second.
        """
        return self.request_count / self.duration if self.duration else 0.0

    @property
    def request_latency_distribution(self) -> Distribution:
        """
        Get the distribution of request latencies in seconds.

        :return: The distribution of request latencies.
        """
        return Distribution(
            data=[
                result.request_latency
                for result in self.results
                if result.request_latency
            ]
        )

    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> float:
        """
        Get the average request latency in seconds.

        :return: The average request latency in seconds.
        :rtype: float
        """
        return self.request_latency_distribution.mean

    @computed_field  # type: ignore[misc]
    @property
    def request_latency_percentiles(self) -> Dict[str, float]:
        """
        Get standard percentiles of request latency in seconds.

        :return: A dictionary mapping percentile to request latency in seconds.
        """
        if not self.results:
            return {}

        values = self.request_latency_distribution.percentiles(DEFAULT_PERCENTILES)

        return dict(zip(map(str, DEFAULT_PERCENTILES), values))

    @property
    def ttft_distribution(self) -> Distribution:
        """
        Get the distribution of time taken to decode the first token.

        :return: The distribution of time taken to decode the first token.
        """
        return Distribution(
            data=[
                result.time_to_first_token
                for result in self.results
                if result.time_to_first_token
            ]
        )

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token(self) -> float:
        """
        Get the time taken to decode the first token in milliseconds.

        :return: The time taken to decode the first token in milliseconds.
        """
        return self.ttft_distribution.mean

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token_percentiles(self) -> Dict[str, float]:
        """
        Get standard percentiles for time taken to decode the first token
        in milliseconds.

        :return: A dictionary mapping percentile to time taken for the first token.
        """
        if not self.results:
            return {}

        values = self.ttft_distribution.percentiles(DEFAULT_PERCENTILES)

        return dict(zip(map(str, DEFAULT_PERCENTILES), values))

    @property
    def itl_distribution(self) -> Distribution:
        """
        Get the distribution of time between tokens in milliseconds.

        :return: The distribution of time between tokens.
        """
        return Distribution(
            data=[
                result.inter_token_latency
                for result in self.results
                for _ in range(
                    result.output_token_count - 1
                    if result.output_token_count and result.output_token_count > 1
                    else 0
                )
                if (result.inter_token_latency)
            ]
        )

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency(self) -> float:
        """
        Get the average time between tokens in milliseconds.

        :return: The average time between tokens.
        """
        return self.itl_distribution.mean


    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency_percentiles(self) -> Dict[str, float]:
        """
        Get standard percentiles for the time between tokens in milliseconds.

        :return: A dictionary mapping percentile to time between tokens.
        """
        if not self.results:
            return {}

        values = self.itl_distribution.percentiles(DEFAULT_PERCENTILES)

        return dict(zip(map(str, DEFAULT_PERCENTILES), values))

    @computed_field  # type: ignore[misc]
    @property
    def output_token_throughput(self) -> float:
        """
        Get the average token throughput in tokens per second.

        :return: The average token throughput.
        """
        output_tokens = sum(
            result.output_token_count
            for result in self.results
            if result.output_token_count and result.output_token_count > 0
        )

        return output_tokens / self.duration if self.duration else 0.0

    @property
    def prompt_token_distribution(self) -> Distribution:
        """
        Get the distribution of prompt token counts.

        :return: The distribution of prompt token counts.
        """
        return Distribution(
            data=[
                result.prompt_token_count
                for result in self.results
                if result.prompt_token_count
            ]
        )

    @computed_field  # type: ignore[misc]
    @property
    def prompt_token(self) -> float:
        """
        Get the average number of prompt tokens.

        :return: The average number of prompt tokens.
        """
        return self.prompt_token_distribution.mean

    @computed_field  # type: ignore[misc]
    @property
    def prompt_token_percentiles(self) -> Dict[str, float]:
        """
        Get standard percentiles for number of prompt tokens.

        :return: A dictionary mapping percentile to number of prompt tokens.
        """
        if not self.results:
            return {}

        values = self.prompt_token_distribution.percentiles(DEFAULT_PERCENTILES)

        return dict(zip(map(str, DEFAULT_PERCENTILES), values))

    @property
    def output_token_distribution(self) -> Distribution:
        """
        Get the distribution of output token counts.

        :return: The distribution of output token counts.
        """
        return Distribution(
            data=[
                result.output_token_count
                for result in self.results
                if result.output_token_count
            ]
        )

    @computed_field  # type: ignore[misc]
    @property
    def output_token(self) -> float:
        """
        Get the average number of output tokens.

        :return: The average number of output tokens.
        """
        return self.output_token_distribution.mean

    @computed_field  # type: ignore[misc]
    @property
    def output_token_percentiles(self) -> Dict[str, float]:
        """
        Get standard percentiles for number of output tokens.

        :return: List of percentiles of number of output tokens.
        """
        if not self.results:
            return {}

        values = self.output_token_distribution.percentiles(DEFAULT_PERCENTILES)

        return dict(zip(map(str, DEFAULT_PERCENTILES), values))

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
