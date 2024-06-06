from typing import Optional
from time import time
from guidellm.core.distribution import Distribution


__all__ = ["BenchmarkResult"]


class BenchmarkResult:
    """
    A class to represent the result of a benchmark request for generative AI workloads.

    :param id_: Unique identifier for the benchmark result.
    :type id_: str
    """

    def __init__(self, id_: str):
        """
        Initialize the BenchmarkResult with a unique identifier.

        :param id_: Unique identifier for the benchmark result.
        :type id_: str
        """
        self.id = id_
        self.prompt = ""
        self.prompt_word_count = 0
        self.prompt_token_count = 0
        self.output = ""
        self.output_word_count = 0
        self.output_token_count = 0

        self._last_time: Optional[float] = None
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
            f"BenchmarkResult(id={self.id}, prompt='{self.prompt}', "
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
            f"BenchmarkResult(id={self.id}, prompt='{self.prompt}', "
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
            self.id == other.id
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

    def output_token(self, token: str):
        """
        Add a token to the output and record the decode time.

        :param token: The decoded token.
        :type token: str
        """
        if self._last_time is None:
            # first token
            self._last_time = time()
            self.first_token_time = self._last_time - self.start_time
        else:
            self._last_time = time()
            decode_time = self._last_time - self._last_time
            self.decode_times.add_data([decode_time])

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
