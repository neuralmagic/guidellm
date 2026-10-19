"""Unit tests for the throughput profile arguments."""

from __future__ import annotations

import pytest

from guidellm.schemas.benchmark import BenchmarkArgs
from guidellm.schemas.benchmark.profiles import ProfileArgs
from guidellm.schemas.benchmark.profiles.throughput import ThroughputProfileArgs

VALID_BASE = {
    "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
    "data": [{"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}],
}


class TestThroughputProfileArgs:
    """Tests for ThroughputProfileArgs."""

    @pytest.mark.regression
    def test_max_concurrency_is_optional(self):
        """
        Accept a throughput profile given only its kind, as the docs show.

        ## WRITTEN BY AI ##
        """
        args = ProfileArgs.model_validate({"kind": "throughput"})

        assert isinstance(args, ThroughputProfileArgs)
        assert args.max_concurrency is None

    @pytest.mark.regression
    def test_full_configuration_without_max_concurrency(self):
        """
        Build a whole benchmark configuration from ``kind=throughput`` alone.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkArgs.model_validate(
            {**VALID_BASE, "profile": {"kind": "throughput"}}
        )

        assert args.profile.kind == "throughput"
        assert args.profile.max_concurrency is None

    @pytest.mark.sanity
    def test_explicit_max_concurrency_is_kept(self):
        """
        Keep an explicit max_concurrency value.

        ## WRITTEN BY AI ##
        """
        args = ProfileArgs.model_validate({"kind": "throughput", "max_concurrency": 8})

        assert args.max_concurrency == 8
