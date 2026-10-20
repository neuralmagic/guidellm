from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from guidellm.schemas.benchmark.outputs.output import BenchmarkOutputArgs
from guidellm.settings import settings

__all__ = ["PlotBenchmarkOutputArgs"]

_ALLOWED_PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


@BenchmarkOutputArgs.register("plot")
class PlotBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for Plot benchmark output arguments.

    Defines parameters for generating static image visualizations, enforcing
    image output suffix.
    """

    kind: Literal["plot"] = Field(
        default="plot",
        description="Type identifier for the plot configuration.",
    )
    path: Path = Field(
        default_factory=lambda: settings.default_results_dir / "benchmarks.png",
        description="The file to save the output plot to.",
    )
    dpi: int = Field(
        default=100,
        description="Resolution of the output image in Dots Per Inch.",
    )

    @field_validator("path", mode="after")
    @classmethod
    def validate_plot_suffix(cls, v: Path) -> Path:
        """Preserve directories and validate the plot filename extension.

        :param v: Output file or existing directory path
        :return: Directory path, or a filename with a supported extension
        :raises ValueError: If a filename has an unsupported extension
        """
        if v.is_dir():
            return v
        if not v.suffix:
            return v.with_suffix(".png")
        suffix = v.suffix.lower()
        if suffix in _ALLOWED_PLOT_SUFFIXES:
            return v
        raise ValueError(
            f"Plot output type {suffix} is not supported: valid types are "
            f"{', '.join(sorted(_ALLOWED_PLOT_SUFFIXES))}"
        )
