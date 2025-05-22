export type Name = 'slo.state';

export interface SloState {
  currentRequestRate: number;
  enforcedPercentile: string;
  current: {
    timePerRequest: number;
    ttft: number;
    tpot: number;
    throughput: number;
  };
  tasksDefaults: {
    timePerRequest: number;
    ttft: number;
    tpot: number;
    throughput: number;
  };
}
