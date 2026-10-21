"""
Generate a multi-run HTML fixture for the Node/jsdom smoke test.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import sys
from pathlib import Path

from guidellm.benchmark.outputs.html import build_report_view, render_html_report
from guidellm.scheduler import ConcurrentStrategy
from tests.unit.benchmark.html_report_fixtures import (
    make_benchmark,
    make_request,
    report,
)


def main() -> None:
    """
    Write a multi-run HTML fixture to the path given on the command line.

    ## WRITTEN BY AI ##
    """
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_fixture.py <output.html>")
    out = Path(sys.argv[1])
    built = report(
        make_benchmark(
            strategy=ConcurrentStrategy(streams=2),
            rps=1.0,
            tps=20.0,
            measure_start=1_700_000_000.0,
            requests=[
                make_request(
                    0,
                    latency_s=0.2 + 0.04 * index,
                    ttft_ms=80.0 + 10.0 * index,
                    itl_ms=10.0,
                    prompt_tokens=50,
                    history_len=0,
                    start_time=1_700_000_001.0 + 2.0 * index,
                    output_tokens=4 + index,
                )
                for index in range(6)
            ],
        ),
        make_benchmark(
            strategy=ConcurrentStrategy(streams=4),
            rps=2.0,
            tps=50.0,
            measure_start=1_700_000_010.0,
            requests=[
                make_request(
                    0,
                    latency_s=0.25 + 0.03 * index,
                    ttft_ms=90.0 + 8.0 * index,
                    itl_ms=11.0,
                    prompt_tokens=60,
                    history_len=0,
                    start_time=1_700_000_011.0 + index,
                    output_tokens=5 + index,
                )
                for index in range(8)
            ],
        ),
    )
    out.write_text(render_html_report(build_report_view(built)), encoding="utf-8")


if __name__ == "__main__":
    main()
