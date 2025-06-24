import { Benchmarks } from './src/lib/store/slices/benchmarks/benchmarks.interfaces';
import { RunInfo } from './src/lib/store/slices/runInfo/runInfo.interfaces';
import { WorkloadDetails } from './src/lib/store/slices/workloadDetails/workloadDetails.interfaces';

declare global {
  interface Window {
    run_info?: RunInfo;
    workload_details?: WorkloadDetails;
    benchmarks?: Benchmarks;
  }
}
