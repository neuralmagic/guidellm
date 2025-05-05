from collections import defaultdict
from math import ceil
from pydantic import BaseModel
import random
from typing import List, Optional, Tuple

from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.objects.statistics import DistributionSummary

__all__ = ["Bucket", "Model", "Dataset", "RunInfo", "TokenDistribution", "TokenDetails", "Server", "WorkloadDetails", "BenchmarkDatum"]

class Bucket(BaseModel):
  value: float
  count: int

  @staticmethod
  def from_data(
      data: List[float],
      bucket_width: Optional[float] = None,
      n_buckets: Optional[int] = None
  ) -> Tuple[List["Bucket"], float]:
      if not data:
          return [], 1.0

      min_v = min(data)
      max_v = max(data)
      range_v = max_v - min_v

      if bucket_width is None:
          if n_buckets is None:
              n_buckets = 10
          bucket_width = range_v / n_buckets
      else:
          n_buckets = ceil(range_v / bucket_width)

      bucket_counts = defaultdict(int)
      for val in data:
          idx = int((val - min_v) // bucket_width)
          if idx >= n_buckets:
              idx = n_buckets - 1
          bucket_start = min_v + idx * bucket_width
          bucket_counts[bucket_start] += 1

      buckets = [Bucket(value=start, count=count) for start, count in sorted(bucket_counts.items())]
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
  def from_benchmarks(cls, benchmarks: list[GenerativeBenchmark]):
    model = benchmarks[0].worker.backend_model or 'N/A'
    timestamp = max(bm.run_stats.start_time for bm in benchmarks if bm.start_time is not None)
    return cls(
      model=Model(name=model, size=0),
      task='N/A',
      timestamp=timestamp,
      dataset=Dataset(name="N/A")
    )

class TokenDistribution(BaseModel):
  statistics: Optional[DistributionSummary] = None
  buckets: list[Bucket]
  bucket_width: float


class TokenDetails(BaseModel):
  samples: list[str]
  token_distributions: TokenDistribution

class Server(BaseModel):
  target: str

class RequestOverTime(BaseModel):
   num_benchmarks: int
   requests_over_time: TokenDistribution

class WorkloadDetails(BaseModel):
  prompts: TokenDetails
  generations: TokenDetails
  requests_over_time: RequestOverTime
  rate_type: str
  server: Server
  @classmethod
  def from_benchmarks(cls, benchmarks: list[GenerativeBenchmark]):
    target = benchmarks[0].worker.backend_target
    rate_type = benchmarks[0].args.profile.type_
    successful_requests = [req for bm in benchmarks for req in bm.requests.successful]
    sample_indices = random.sample(range(len(successful_requests)), min(5, len(successful_requests)))
    sample_prompts = [successful_requests[i].prompt.replace("\n", " ").replace("\"", "'") for i in sample_indices]
    sample_outputs = [successful_requests[i].output.replace("\n", " ").replace("\"", "'") for i in sample_indices]

    prompt_tokens = [req.prompt_tokens for bm in benchmarks for req in bm.requests.successful]
    output_tokens = [req.output_tokens for bm in benchmarks for req in bm.requests.successful]

    prompt_token_buckets, _prompt_token_bucket_width = Bucket.from_data(prompt_tokens, 1)
    output_token_buckets, _output_token_bucket_width = Bucket.from_data(output_tokens, 1)
    
    prompt_token_stats = DistributionSummary.from_values(prompt_tokens)
    output_token_stats = DistributionSummary.from_values(output_tokens)
    prompt_token_distributions = TokenDistribution(statistics=prompt_token_stats, buckets=prompt_token_buckets, bucket_width=1)
    output_token_distributions = TokenDistribution(statistics=output_token_stats, buckets=output_token_buckets, bucket_width=1)

    min_start_time = benchmarks[0].run_stats.start_time

    all_req_times = [
       req.start_time - min_start_time
       for bm in benchmarks
       for req in bm.requests.successful
       if req.start_time is not None
    ]
    number_of_buckets = len(benchmarks)
    request_over_time_buckets, bucket_width = Bucket.from_data(all_req_times, None, number_of_buckets)
    request_over_time_distribution = TokenDistribution(buckets=request_over_time_buckets, bucket_width=bucket_width)
    return cls(
       prompts=TokenDetails(samples=sample_prompts, token_distributions=prompt_token_distributions),
       generations=TokenDetails(samples=sample_outputs, token_distributions=output_token_distributions),
       requests_over_time=RequestOverTime(requests_over_time=request_over_time_distribution, num_benchmarks=number_of_buckets),
       rate_type=rate_type,
       server=Server(target=target)
    )

class BenchmarkDatum(BaseModel):
  requests_per_second: float
  tpot: DistributionSummary
  ttft: DistributionSummary
  throughput: DistributionSummary
  time_per_request: DistributionSummary

  @classmethod
  def from_benchmark(cls, bm: GenerativeBenchmark):
    return cls(
       requests_per_second=bm.metrics.requests_per_second.successful.mean,
       tpot=bm.metrics.inter_token_latency_ms.successful,
       ttft=bm.metrics.time_to_first_token_ms.successful,
       throughput=bm.metrics.output_tokens_per_second.successful,
       time_per_request=bm.metrics.request_latency.successful,
    )
