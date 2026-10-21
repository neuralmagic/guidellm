from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs

__all__ = ["SweepNextProfileArgs"]


@ProfileArgs.register("sweep_next")
class SweepNextProfileArgs(ProfileArgs):
    """Pydantic model for sweep_next profile creation arguments."""

    kind: Literal["sweep_next"] = Field(
        default="sweep_next",
        description="Profile type discriminator for sweep_next scheduling",
    )
    sweep_size: int = Field(
        default=10,
        description="Number of strategies to generate for the sweep",
        ge=2,
    )
    strategy_type: Literal["constant", "poisson"] = Field(
        default="constant",
        description="Type of strategy to use for the asynchronous sweep",
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description=(
            "Maximum concurrent requests to schedule for the ramp and the "
            "asynchronous steps. Unlimited by default."
        ),
    )
