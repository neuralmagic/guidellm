export type Name = 'benchmarks';

interface Statistics {
  total: number;
  mean: number;
  std: number;
  median: number;
  min: number;
  max: number;
}

export type PercentileValues = 'p50' | 'p90' | 'p95' | 'p99';

interface Percentile {
  percentile: string;
  value: number;
}

interface Bucket {
  value: number;
  count: number;
}

export interface MetricData {
  statistics: Statistics;
  percentiles: Percentile[];
  buckets: Bucket[];
  bucketWidth: number;
}

export interface BenchmarkMetrics {
  ttft: MetricData;
  tpot: MetricData;
  timePerRequest: MetricData;
  throughput: MetricData;
}

export interface Benchmark extends BenchmarkMetrics {
  requestsPerSecond: number;
}

export type Benchmarks = {
  benchmarks: Benchmark[];
};
