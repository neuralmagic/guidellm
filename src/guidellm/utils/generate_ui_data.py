import os
import json
import random
import math
from typing import Any, Dict, List
from guidellm.core.distribution import Distribution
from guidellm.core import TextGenerationBenchmarkReport, TextGenerationBenchmark
from guidellm.utils.interpolation import interpolate_data_points

def generate_metric_report(dist: Distribution, metric_label: str, n_buckets: int = 18):
    total = dist.__len__()
    mean = dist.mean
    median = dist.median
    minv = dist.min
    maxv = dist.max
    std_dev = dist.std_deviation

    pvals = dist.percentiles([50, 90, 95, 99])

    percentile_list = [
        {"percentile": "p50", "value": pvals[0]},
        {"percentile": "p90", "value": pvals[1]},
        {"percentile": "p95", "value": pvals[2]},
        {"percentile": "p99", "value": pvals[3]},
    ]

    if dist.range == 0:
        buckets = [{"value": minv, "count": total}]
        bucket_width = 0
    else:
        bucket_width = dist.range / n_buckets
        bucket_counts = [0] * n_buckets

        for val in dist.data:

            idx = int((val - minv) // bucket_width)
            if idx == n_buckets:
                idx = n_buckets - 1
            bucket_counts[idx] += 1

        buckets = []
        for i, count in enumerate(bucket_counts):
            bucket_start = minv + i * bucket_width
            buckets.append({
                "value": bucket_start,
                "count": count
            })

    return {
        metric_label: {
            "statistics": {
                "total": total,
                "mean": mean,
                "median": median,
                "min": minv,
                "max": maxv,
                "std": std_dev,
            },
            "percentiles": percentile_list,
            "buckets": buckets,
            "bucketWidth": bucket_width,
        }
    }

def generate_run_info(report: TextGenerationBenchmarkReport, benchmarks: List[TextGenerationBenchmark]) -> Dict[str, Any]:
    timestamp = max(bm.start_time for bm in benchmarks if bm.start_time is not None)
    return {
        "model": {
            "name": report.args.get('model', 'N/A'),
            "size": 0
        },
        "task": "N/A",
        "dataset": {
            "name": "N/A"
        },
        "timestamp": timestamp
    }

def linearly_interpolate_value(target_input, lower_input, lower_output, upperInput, upper_output):
    fraction = (target_input - lower_input) / (upperInput - lower_input)
    return lower_output + fraction * (upper_output - lower_output)

def generate_request_over_time_data(benchmarks: List[TextGenerationBenchmark]) -> List[Dict[str, Any]]:
    filtered_benchmarks = filter(lambda bm: bm.start_time is not None, benchmarks)
    sorted_benchmarks = list(sorted(filtered_benchmarks, key=lambda bm: bm.start_time))
    min_start_time = sorted_benchmarks[0].start_time

    all_request_times = [
        result.start_time - min_start_time
        for benchmark in sorted_benchmarks
        for result in benchmark.results
        if result.start_time is not None
    ]

    request_distribution = Distribution(data=all_request_times)
    final_result = generate_metric_report(request_distribution, "requestsOverTime")
    return { "numBenchmarks": len(sorted_benchmarks), **final_result }


def generate_workload_details(report: TextGenerationBenchmarkReport, benchmarks: List[TextGenerationBenchmark]) -> Dict[str, Any]:
    all_prompt_token_data = [data for benchmark in benchmarks for data in benchmark.prompt_token_distribution.data]
    all_prompt_token_distribution = Distribution(data=all_prompt_token_data)
    all_output_token_data = [data for benchmark in benchmarks for data in benchmark.output_token_distribution.data]
    all_output_token_distribution = Distribution(data=all_output_token_data)

    prompt_token_data = generate_metric_report(all_prompt_token_distribution, "tokenDistributions")
    output_token_data = generate_metric_report(all_output_token_distribution, "tokenDistributions")

    prompt_token_samples = [result.prompt for benchmark in benchmarks for result in benchmark.results]
    output_token_samples = [result.output for benchmark in benchmarks for result in benchmark.results]

    num_samples = min(5, len(prompt_token_samples), len(output_token_samples))
    sample_indices = random.sample(range(len(prompt_token_samples)), num_samples)

    sample_prompts = [prompt_token_samples[i] for i in sample_indices]
    """
    Need a wholistic approach to parsing out characters in the prompt that don't covert well into the format we need
    """
    sample_prompts = list(map(lambda prompt: prompt.replace("\n", " ").replace("\"", "'"), sample_prompts))

    sample_outputs = [output_token_samples[i] for i in sample_indices]
    sample_outputs = list(map(lambda output: output.replace("\n", " ").replace("\"", "'"), sample_outputs))

    request_over_time_results = generate_request_over_time_data(benchmarks)

    return {
        "prompts": {
            "samples": sample_prompts,
            **prompt_token_data
        },
        "generations": {
            "samples": sample_outputs,
            **output_token_data
        },
        "requestsOverTime": request_over_time_results,
        "rateType": report.args["mode"],
        "server": {
            "target": report.args.get('target', 'N/A')
        }
    }

def generate_benchmark_json(bm: TextGenerationBenchmark) -> Dict[str, Any]:
    ttft_dist_ms = Distribution(data=[val * 1000 for val in bm.ttft_distribution.data])
    ttft_data = generate_metric_report(ttft_dist_ms, 'ttft')
    itl_dist_ms = Distribution(data=[val * 1000 for val in bm.itl_distribution.data])
    itl_data = generate_metric_report(itl_dist_ms, 'tpot')
    throughput_dist_ms = Distribution(data=bm.output_token_throughput_distribution.data)
    throughput_data = generate_metric_report(throughput_dist_ms, 'throughput')
    latency_dist_ms = Distribution(data=[val * 1000 for val in bm.request_latency_distribution.data])
    latency__data = generate_metric_report(latency_dist_ms, 'timePerRequest')
    return {
        "requestsPerSecond": bm.completed_request_rate,
        **itl_data,
        **ttft_data,
        **throughput_data,
        **latency__data,
    }

def generate_interpolated_benchmarks(benchmarks: List[TextGenerationBenchmark]):
    """
    Should we only use constant rate benchmarks here since synchronous and throughput runs might not be appropriate to lump in for interoplation across all rps?

    Other edge-case, what if rps doesn't span more than 1 whole rps even with multiple benchmarks
    ex: 1.1, 1.3, 1.5, 2.1, 2.5, can interpolate at 2rps
    or worse, 1.1, 1.4, 1.6, can't interpolate
    """
    if len(benchmarks) == 1:
        return []
    
    sorted_benchmarks = sorted(benchmarks[:], key=lambda bm: bm.completed_request_rate)
    rps_values = [bm.completed_request_rate for bm in sorted_benchmarks]
    rps_range = list(range(math.ceil(min(rps_values)), math.ceil(max(rps_values))))

    ttft_data_by_rps = list(map(lambda bm: (bm.completed_request_rate, bm.ttft_distribution.data), sorted_benchmarks))
    interpolated_ttft_data_by_rps = interpolate_data_points(ttft_data_by_rps, rps_range)

    itl_data_by_rps = list(map(lambda bm: (bm.completed_request_rate, bm.itl_distribution.data), sorted_benchmarks))
    interpolated_itl_data_by_rps = interpolate_data_points(itl_data_by_rps, rps_range)

    throughput_data_by_rps = list(map(lambda bm: (bm.completed_request_rate, bm.output_token_throughput_distribution.data), sorted_benchmarks))
    interpolated_throughput_data_by_rps = interpolate_data_points(throughput_data_by_rps, rps_range)

    latency_data_by_rps = list(map(lambda bm: (bm.completed_request_rate, bm.request_latency_distribution.data), sorted_benchmarks))
    interpolated_latency_data_by_rps = interpolate_data_points(latency_data_by_rps, rps_range)

    benchmark_json = []
    for i in range(len(interpolated_ttft_data_by_rps)):
        rps, interpolated_ttft_data = interpolated_ttft_data_by_rps[i]
        ttft_dist_ms = Distribution(data=[val * 1000 for val in interpolated_ttft_data])
        final_ttft_data = generate_metric_report(ttft_dist_ms, 'ttft')
        
        _, interpolated_itl_data = interpolated_itl_data_by_rps[i]
        itl_dist_ms = Distribution(data=[val * 1000 for val in interpolated_itl_data])
        final_itl_data = generate_metric_report(itl_dist_ms, 'tpot')

        _, interpolated_throughput_data = interpolated_throughput_data_by_rps[i]
        throughput_dist_ms = Distribution(data=interpolated_throughput_data)
        final_throughput_data = generate_metric_report(throughput_dist_ms, 'throughput')

        _, interpolated_latency_data = interpolated_latency_data_by_rps[i]
        latency_dist_ms = Distribution(data=[val * 1000 for val in interpolated_latency_data])
        final_latency_data = generate_metric_report(latency_dist_ms, 'timePerRequest')

        benchmark_json.append({
            "requestsPerSecond": rps,
            **final_itl_data,
            **final_ttft_data,
            **final_throughput_data,
            **final_latency_data,
        })
    return benchmark_json

def generate_benchmarks_json(benchmarks: List[TextGenerationBenchmark]):
    raw_benchmark_json = []
    for benchmark in benchmarks:
        benchmarks_report = generate_benchmark_json(benchmark)
        raw_benchmark_json.append(benchmarks_report)
    interpolated_benchmark_json = generate_interpolated_benchmarks(benchmarks)

    return { "raw": raw_benchmark_json, "interpolatedByRps": interpolated_benchmark_json }

def generate_js_variable(variable_name: str, data: dict) -> str:
    json_data = json.dumps(data, indent=2)
    return f'`window.{variable_name} = {json_data};`'  # Wrap in quotes

def generate_ui_api_data(report: TextGenerationBenchmarkReport):
    filtered_benchmarks = list(filter(lambda bm: (bm.completed_request_rate > 0) and bm.mode != 'throughput', report.benchmarks))
    run_info_data = generate_run_info(report, filtered_benchmarks)
    workload_details_data = generate_workload_details(report, filtered_benchmarks)
    benchmarks_data = generate_benchmarks_json(filtered_benchmarks)
    run_info_script = generate_js_variable("run_info", run_info_data)
    workload_details_script = generate_js_variable("workload_details", workload_details_data)
    benchmarks_script = generate_js_variable("benchmarks", benchmarks_data)

    os.makedirs("ben_test", exist_ok=True)
    # generate json files based off of api specs, https://codepen.io/dalthecow/pen/bNGVQbq, for consumption by UI
    with open("ben_test/run_info.js", "w") as f:
        f.write(run_info_script)
    with open("ben_test/workload_details.js", "w") as f:
        f.write(workload_details_script)
    with open("ben_test/benchmarks.js", "w") as f:
        f.write(benchmarks_script)

    print("Reports saved to run_info.json, workload_details.json, benchmarks.json")