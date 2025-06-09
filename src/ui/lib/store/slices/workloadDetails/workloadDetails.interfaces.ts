export type Name = 'workloadDetails';

interface Statistics {
  total: number;
  mean: number;
  std: number;
  median: number;
  min: number;
  max: number;
}

interface Percentile {
  percentile: string;
  value: number;
}

interface Bucket {
  value: number;
  count: number;
}

interface Distribution {
  statistics: Statistics;
  percentiles: Percentile[];
  buckets: Bucket[];
  bucketWidth: number;
}

interface TokenData {
  samples: string[];
  tokenDistributions: Distribution;
}

interface BenchmarkData {
  numBenchmarks: number;
  requestsOverTime: Distribution;
}

interface Server {
  target: string;
}

export interface WorkloadDetails {
  prompts: TokenData;
  generations: TokenData;
  requestsOverTime: BenchmarkData;
  rateType: string;
  server: Server;
}
