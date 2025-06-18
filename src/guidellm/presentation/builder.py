from typing import Any

from guidellm.benchmark.benchmark import GenerativeBenchmark

from .data_models import BenchmarkDatum, RunInfo, WorkloadDetails

__all__ = ["UIDataBuilder"]


class UIDataBuilder:
    def __init__(self, benchmarks: list[GenerativeBenchmark]):
        self.benchmarks = benchmarks

    def build_run_info(self):
        return RunInfo.from_benchmarks(self.benchmarks)

    def build_workload_details(self):
        return WorkloadDetails.from_benchmarks(self.benchmarks)

    def build_benchmarks(self):
        return [BenchmarkDatum.from_benchmark(b) for b in self.benchmarks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_info": self.build_run_info().dict(),
            "workload_details": self.build_workload_details().dict(),
            "benchmarks": [b.dict() for b in self.build_benchmarks()],
        }
