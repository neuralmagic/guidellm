import random
from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, computed_field

if TYPE_CHECKING:
    from guidellm.benchmark.objects import GenerativeBenchmark

from guidellm.utils.statistics import DistributionSummary


class Bucket(BaseModel):
    value: Union[float, int]
    count: int

    @staticmethod
    def from_data(
        data: Union[list[float], list[int]],
        bucket_width: Optional[float] = None,
        n_buckets: Optional[int] = None,
    ) -> tuple[list["Bucket"], float]:
        if not data:
            return [], 1.0

        min_v = min(data)
        max_v = max(data)
        range_v = (1 + max_v) - min_v

        if bucket_width is None:
            if n_buckets is None:
                n_buckets = 10
            bucket_width = range_v / n_buckets
        else:
            n_buckets = ceil(range_v / bucket_width)

        bucket_counts: defaultdict[Union[float, int], int] = defaultdict(int)
        for val in data:
            idx = int((val - min_v) // bucket_width)
            if idx >= n_buckets:
                idx = n_buckets - 1
            bucket_start = min_v + idx * bucket_width
            bucket_counts[bucket_start] += 1

        buckets = [
            Bucket(value=start, count=count)
            for start, count in sorted(bucket_counts.items())
        ]
        return buckets, bucket_width


class Model(BaseModel):
    name: str
    size: int


class Dataset(BaseModel):
    name: str


class RunInfo(BaseModel):
    model: Model
    task: str
    timestamp: float
    dataset: Dataset

    @classmethod
    def from_benchmarks(cls, benchmarks: list["GenerativeBenchmark"]):
        # TODO: Review Cursor generated code (start)
        # Try to extract model from benchmarker.backend with safe fallbacks
        model = "N/A"
        try:
            backend = benchmarks[0].benchmarker.backend
            if isinstance(backend, dict) and "model" in backend:
                model = backend["model"] or "N/A"
            elif hasattr(backend, "model"):
                model = getattr(backend, "model", "N/A") or "N/A"
            elif isinstance(backend, dict) and "info" in backend:
                # Try to extract model from info string
                info = backend["info"]
                if isinstance(info, str) and "model" in info.lower():
                    model = info
                else:
                    model = "N/A"
        except Exception:
            model = "N/A"
        # TODO: Review Cursor generated code (end)
        timestamp = max(
            bm.run_stats.start_time for bm in benchmarks if bm.start_time is not None
        )
        return cls(
            model=Model(name=model, size=0),
            task="N/A",
            timestamp=timestamp,
            dataset=Dataset(name="N/A"),
        )


class Distribution(BaseModel):
    statistics: Optional[DistributionSummary] = None
    buckets: list[Bucket]
    bucket_width: float


class TokenDetails(BaseModel):
    samples: list[str]
    token_distributions: Distribution


class Server(BaseModel):
    target: str


class RequestOverTime(BaseModel):
    num_benchmarks: int
    requests_over_time: Distribution


class WorkloadDetails(BaseModel):
    prompts: TokenDetails
    generations: TokenDetails
    requests_over_time: RequestOverTime
    rate_type: str
    server: Server

    @classmethod
    def from_benchmarks(cls, benchmarks: list["GenerativeBenchmark"]):
        # TODO: Review Cursor generated code (start)
        # Try to extract target from benchmarker.backend with safe fallbacks
        target = "N/A"
        try:
            backend = benchmarks[0].benchmarker.backend
            if isinstance(backend, dict) and "target" in backend:
                target = backend["target"] or "N/A"
            elif hasattr(backend, "target"):
                target = getattr(backend, "target", "N/A") or "N/A"
        except Exception:
            target = "N/A"
        # TODO: Review Cursor generated code (end)

        # TODO: Review Cursor generated code (start)
        # Try to extract rate_type from benchmarker.profile with safe fallbacks
        rate_type = "N/A"
        try:
            profile = benchmarks[0].benchmarker.profile
            if hasattr(profile, "type_"):
                rate_type = getattr(profile, "type_", "N/A") or "N/A"
            elif isinstance(profile, dict) and "type_" in profile:
                rate_type = profile["type_"] or "N/A"
        except Exception:
            rate_type = "N/A"
        # TODO: Review Cursor generated code (end)
        successful_requests = [
            req for bm in benchmarks for req in bm.requests.successful
        ]
        sample_indices = random.sample(
            range(len(successful_requests)), min(5, len(successful_requests))
        )
        sample_prompts = [
            successful_requests[i].prompt.replace("\n", " ").replace('"', "'")
            for i in sample_indices
        ]
        sample_outputs = [
            successful_requests[i].output.replace("\n", " ").replace('"', "'")
            for i in sample_indices
        ]

        prompt_tokens = [
            float(req.prompt_tokens)
            for bm in benchmarks
            for req in bm.requests.successful
        ]
        output_tokens = [
            float(req.output_tokens)
            for bm in benchmarks
            for req in bm.requests.successful
        ]

        prompt_token_buckets, _prompt_token_bucket_width = Bucket.from_data(
            prompt_tokens, 1
        )
        output_token_buckets, _output_token_bucket_width = Bucket.from_data(
            output_tokens, 1
        )

        prompt_token_stats = DistributionSummary.from_values(prompt_tokens)
        output_token_stats = DistributionSummary.from_values(output_tokens)
        prompt_token_distributions = Distribution(
            statistics=prompt_token_stats, buckets=prompt_token_buckets, bucket_width=1
        )
        output_token_distributions = Distribution(
            statistics=output_token_stats, buckets=output_token_buckets, bucket_width=1
        )

        min_start_time = benchmarks[0].run_stats.start_time

        all_req_times = [
            # TODO: Review Cursor generated code (start)
            req.scheduler_info.request_timings.request_start - min_start_time
            # TODO: Review Cursor generated code (end)
            for bm in benchmarks
            for req in bm.requests.successful
            # TODO: Review Cursor generated code (start)
            if req.scheduler_info.request_timings.request_start is not None
            # TODO: Review Cursor generated code (end)
        ]
        number_of_buckets = len(benchmarks)
        request_over_time_buckets, bucket_width = Bucket.from_data(
            all_req_times, None, number_of_buckets
        )
        request_over_time_distribution = Distribution(
            buckets=request_over_time_buckets, bucket_width=bucket_width
        )
        return cls(
            prompts=TokenDetails(
                samples=sample_prompts, token_distributions=prompt_token_distributions
            ),
            generations=TokenDetails(
                samples=sample_outputs, token_distributions=output_token_distributions
            ),
            requests_over_time=RequestOverTime(
                requests_over_time=request_over_time_distribution,
                num_benchmarks=number_of_buckets,
            ),
            rate_type=rate_type,
            server=Server(target=target),
        )


class TabularDistributionSummary(DistributionSummary):
    """
    Same fields as `DistributionSummary`, but adds a ready-to-serialize/iterate
    `percentile_rows` helper.
    """

    @computed_field
    def percentile_rows(self) -> list[dict[str, float]]:
        rows = [
            {"percentile": name, "value": value}
            for name, value in self.percentiles.model_dump().items()
        ]
        return list(
            filter(lambda row: row["percentile"] in ["p50", "p90", "p95", "p99"], rows)
        )

    @classmethod
    def from_distribution_summary(
        cls, distribution: DistributionSummary
    ) -> "TabularDistributionSummary":
        return cls(**distribution.model_dump())


class BenchmarkDatum(BaseModel):
    requests_per_second: float
    tpot: TabularDistributionSummary
    ttft: TabularDistributionSummary
    throughput: TabularDistributionSummary
    time_per_request: TabularDistributionSummary

    @classmethod
    def from_benchmark(cls, bm: "GenerativeBenchmark"):
        return cls(
            requests_per_second=bm.metrics.requests_per_second.successful.mean,
            tpot=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.inter_token_latency_ms.successful
            ),
            ttft=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.time_to_first_token_ms.successful
            ),
            throughput=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.output_tokens_per_second.successful
            ),
            time_per_request=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.request_latency.successful
            ),
        )
