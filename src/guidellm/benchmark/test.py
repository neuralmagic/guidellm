import asyncio
import json

from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.benchmark.entrypoints import benchmark_generative_text
from guidellm.benchmark.output import GenerativeBenchmarksConsole


def run_benchmark_synthetic():
    # logging.basicConfig(level=logging.DEBUG)
    results = asyncio.run(
        benchmark_generative_text(
            target="http://192.168.4.13:8000",
            backend_type="openai_http",
            backend_args=None,
            model="neuralmagic/Qwen2.5-7B-quantized.w8a8",
            processor=None,
            processor_args=None,
            data='{"prompt_tokens": 128, "output_tokens": 64}',
            data_args=None,
            data_sampler=None,
            rate_type="sweep",
            rate=10,
            max_seconds=None,
            max_requests=50,
            warmup_percent=None,
            cooldown_percent=None,
            show_progress=True,
            show_progress_scheduler_stats=True,
            output_console=True,
            output_path="benchmarks.json",
            output_extras=None,
            random_seed=42,
        )
    )


def print_benchmark():
    with open("benchmarks.json") as file:
        data = json.load(file)

    benchmarks = [
        GenerativeBenchmark.model_validate_json(json.dumps(bench))
        for bench in data["benchmarks"]
    ]
    console = GenerativeBenchmarksConsole(benchmarks)
    console.print()


if __name__ == "__main__":
    run_benchmark_synthetic()
    # print_benchmark()
