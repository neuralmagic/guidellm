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
        "dataset": "N/A",
        "timestamp": timestamp
    }

def generate_workload_details(report: TextGenerationBenchmarkReport) -> Dict[str, Any]:
    all_prompt_token_data = [data for benchmark in report.benchmarks for data in benchmark.prompt_token_distribution.data]
    all_prompt_token_distribution = Distribution(data=all_prompt_token_data)
    all_output_token_data = [data for benchmark in report.benchmarks for data in benchmark.output_token_distribution.data]
    all_output_token_distribution = Distribution(data=all_output_token_data)

    prompt_token_data = generate_metric_report(all_prompt_token_distribution, "tokenDistributions")
    prompt_token_samples = [result.prompt for benchmark in report.benchmarks for result in benchmark.results]
    sample_prompts = random.sample(prompt_token_samples, min(5, len(prompt_token_samples)))
    output_token_data = generate_metric_report(all_output_token_distribution, "tokenDistributions")
    output_token_samples = [result.output for benchmark in report.benchmarks for result in benchmark.results]
    sample_outputs = random.sample(output_token_samples, min(5, len(output_token_samples)))
    return {
        "prompts": {
            "samples": sample_prompts,
            **prompt_token_data
        },
        "generation": {
            "samples": sample_outputs,
            **output_token_data
        },
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

def generate_ui_api_data(report: TextGenerationBenchmarkReport):
    run_info_json = generate_run_info(report)
    workload_details_json = generate_workload_details(report)
    benchmarks_json = generate_benchmarks_json(report.benchmarks)
    os.makedirs("ben_test", exist_ok=True)
    # generate json files based off of api specs, https://codepen.io/dalthecow/pen/bNGVQbq, for consumption by UI
    with open("ben_test/run_info.json", "w") as f:
        json.dump(run_info_json, f, indent=2)
    with open("ben_test/workload_details.json", "w") as f:
        json.dump(workload_details_json, f, indent=2)
    with open("ben_test/benchmarks.json", "w") as f:
        json.dump(benchmarks_json, f, indent=2)

    print("Reports saved to run_info.json, workload_details.json, benchmarks.json")