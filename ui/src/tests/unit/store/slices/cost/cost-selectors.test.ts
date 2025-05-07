import {
  selectDeploymentCosts,
  selectRequestAndTokenCosts,
} from '../../../../../lib/store/slices/cost/cost.selectors';
import { CostState } from '../../../../../lib/store/slices/cost/cost.interfaces';
import { SloState } from '../../../../../lib/store/slices/slo/slo.interfaces';
import { RootState } from '../../../../../lib/store';

// Mock state setup
const mockCostState: CostState = {
  server_options: [
    { value: 4, label: 'AWS A100 (p4d.24xlarge): $4.096' },
    { value: 16, label: 'AWS 4xA100 (p4d.24xlarge): $16.386' },
    { value: 32, label: 'AWS 8xA100 (p4d.24xlarge): $32.772' },
  ],
  selected_server_option: 4,
  custom_server_cost: 0,
  is_per_thousand_tokens: false,
  target_request_rate: 20,
  input_token_buckets: { '10.0': 100, '40.0': 100 },
  output_token_buckets: { '10.0': 200, '40.0': 200 },
  start_time: 0,
  end_time: 300,
};

const mockSloState: SloState = {
  enforcedPercentile: 'p90',
  currentRequestRate: 10,
  current: {
    ttft: 0,
    tpot: 0,
    timePerRequest: 0,
    throughput: 0,
  },
  tasksDefaults: {
    ttft: 0,
    tpot: 0,
    timePerRequest: 0,
    throughput: 0,
  },
};

const mockState: Partial<RootState> = {
  cost: mockCostState,
  slo: mockSloState,
};

describe('selectRequestAndTokenCosts', () => {
  it('should calculate cost per requests and tokens correctly', () => {
    const result = selectRequestAndTokenCosts(mockState as RootState);
    expect(result).toEqual({
      costPerXRequests: '$ 1111.11',
      costPerXInputTokens: '$ 666.67',
      costPerXOutputTokens: '$ 333.33',
    });
  });

  it('should calculate the deployment cost data correctly', () => {
    const result = selectDeploymentCosts(mockState as RootState);
    expect(result).toEqual({
      numServers: '2',
      totalCostPerHour: '$ 8.00',
      totalCostPerMonth: '$ 5760.00',
    });
  });
});
