import { CostState } from '../../../../../lib/store/slices/cost/cost.interfaces';
import costReducer, { setCostData } from '../../../../../lib/store/slices/cost/cost.slice';

const initialState = {
  server_options: [
    { value: 4, label: 'AWS A100 (p4d.24xlarge): $4.096' },
    { value: 16, label: 'AWS 4xA100 (p4d.24xlarge): $16.386' },
    { value: 32, label: 'AWS 8xA100 (p4d.24xlarge): $32.772' },
  ],
  selected_server_option: undefined,
  custom_server_cost: 0,
  is_per_thousand_tokens: false,
  target_request_rate: 0,
  input_token_buckets: {},
  output_token_buckets: {},
  start_time: 0,
  end_time: 0,
} as CostState;

test('should handle initial state', () => {
  expect(costReducer(undefined, { type: '' })).toEqual(initialState);
});

test('should set cost data', () => {
  const cost = {
    target_request_rate: 0,
    per_request: 0,
  } as Partial<CostState>;

  const fullCost = {
    ...initialState,
    ...cost,
  };
  expect(costReducer(undefined, setCostData(cost))).toEqual(fullCost);
});
