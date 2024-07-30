from typing import List

from pydantic import Field

from guidellm.core.result import TextGenerationBenchmarkReport
from guidellm.core.serializable import Serializable

__all__ = [
    "GuidanceReport",
]


class GuidanceReport(Serializable):
    """
    A class to manage the guidance reports that include the benchmarking details,
    potentially across multiple runs, for saving and loading from disk.
    """

    benchmarks: List[TextGenerationBenchmarkReport] = Field(
        default_factory=list, description="The list of benchmark reports."
    )
