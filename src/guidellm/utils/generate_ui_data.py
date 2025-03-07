import os
import json
import random
from typing import Any, Dict, List
from guidellm.core.distribution import Distribution
from guidellm.core import TextGenerationBenchmarkReport, TextGenerationBenchmark

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

def generate_run_info(report: TextGenerationBenchmarkReport) -> Dict[str, Any]:
    timestamp = max(map(lambda bm: bm.end_time, report.benchmarks))
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

def generate_request_over_time_data(benchmarks: List[TextGenerationBenchmark]) -> List[Dict[str, Any]]:
    request_over_time_results = []
    for benchmark in benchmarks:
        # compare benchmark start time to text generation result end time
        all_result_end_times = [result.end_time for result in benchmark.results if result.end_time is not None]
        request_over_time_values = list(map(lambda time: time - benchmark.start_time, all_result_end_times))
        request_distribution = Distribution(data=request_over_time_values)
        result = generate_metric_report(request_distribution, "requestsOverTime")
        request_over_time_results.append(result["requestsOverTime"])
    return request_over_time_results


def generate_workload_details(report: TextGenerationBenchmarkReport) -> Dict[str, Any]:
    all_prompt_token_data = [data for benchmark in report.benchmarks for data in benchmark.prompt_token_distribution.data]
    all_prompt_token_distribution = Distribution(data=all_prompt_token_data)
    all_output_token_data = [data for benchmark in report.benchmarks for data in benchmark.output_token_distribution.data]
    all_output_token_distribution = Distribution(data=all_output_token_data)

    prompt_token_data = generate_metric_report(all_prompt_token_distribution, "tokenDistributions")
    prompt_token_samples = [result.prompt for benchmark in report.benchmarks for result in benchmark.results]
    sample_prompts = random.sample(prompt_token_samples, min(5, len(prompt_token_samples)))
    sample_prompts = list(map(lambda prompt: prompt.replace("\n", " ").replace("\"", "'"), sample_prompts))
    output_token_data = generate_metric_report(all_output_token_distribution, "tokenDistributions")
    output_token_samples = [result.output for benchmark in report.benchmarks for result in benchmark.results]
    sample_outputs = random.sample(output_token_samples, min(5, len(output_token_samples)))

    sample_outputs = list(map(lambda output: output.replace("\n", " ").replace("\"", "'"), sample_outputs))

    request_over_time_results = generate_request_over_time_data(report.benchmarks)

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
        "server": {
            "target": report.args.get('target', 'N/A')
        }
    }

def generate_benchmark_json(bm: TextGenerationBenchmark) -> Dict[str, Any]:
    ttft_dist_ms = Distribution(data=[val * 1000 for val in bm.ttft_distribution.data])
    ttft_data = generate_metric_report(ttft_dist_ms, 'ttft')
    tpot_dist_ms = Distribution(data=[val * 1000 for val in bm.itl_distribution.data])
    tpot_data = generate_metric_report(tpot_dist_ms, 'tpot')
    throughput_dist_ms = Distribution(data=[val * 1000 for val in bm.output_token_throughput_distribution.data])
    throughput_data = generate_metric_report(throughput_dist_ms, 'throughput')
    latency_dist_ms = Distribution(data=[val * 1000 for val in bm.request_latency_distribution.data])
    time_per_request_data = generate_metric_report(latency_dist_ms, 'timePerRequest')
    return {
        "requestsPerSecond": bm.completed_request_rate,
        **ttft_data,
        **tpot_data,
        **throughput_data,
        **time_per_request_data,
    }

def generate_benchmarks_json(benchmarks: List[TextGenerationBenchmark]):
    benchmark_report_json = []
    for benchmark in benchmarks:
        benchmarks_report = generate_benchmark_json(benchmark)
        benchmark_report_json.append(benchmarks_report)
    return benchmark_report_json

def generate_js_variable(variable_name: str, data: dict) -> str:
    json_data = json.dumps(data, indent=2)
    return f'`window.{variable_name} = {json_data};`'  # Wrap in quotes

def generate_ui_api_data(report: TextGenerationBenchmarkReport):
    run_info_data = generate_run_info(report)
    workload_details_data = generate_workload_details(report)
    benchmarks_data = generate_benchmarks_json(report.benchmarks)
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