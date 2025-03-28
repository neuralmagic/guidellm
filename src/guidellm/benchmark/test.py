import asyncio
import json

from guidellm.benchmark.entrypoints import benchmark_generative_text


def run_benchmark_synthetic():
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
            rate=5,
            max_seconds=None,
            max_requests=50,
            warmup_percent=None,
            cooldown_percent=None,
            show_progress=True,
            output_path=None,
            output_type=None,
            output_extras=None,
            random_seed=42,
        )
    )

    dict_output = {
        "benchmarks": [res.model_dump() for res in results],
    }
    with open("benchmarks.json", "w") as f:
        json.dump(dict_output, f, indent=4)


if __name__ == "__main__":
    run_benchmark_synthetic()
