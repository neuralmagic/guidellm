export type Name = 'metrics.state';

export interface MetricsState {
  currentRequestRate: number;
  timePerRequest: SingleMetricsState;
  ttft: SingleMetricsState;
  tpot: SingleMetricsState;
  throughput: SingleMetricsState;
}

export type SingleMetricsState = {
  valuesByRps: Record<number, number>;
};
