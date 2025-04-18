<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
Scale Efficiently: Evaluate and Optimize Your LLM Deployments for Real-World Inference
</h3>

[![GitHub Release](https://img.shields.io/github/release/neuralmagic/guidellm.svg?label=Version)](https://github.com/neuralmagic/guidellm/releases) [![Documentation](https://img.shields.io/badge/Documentation-8A2BE2?logo=read-the-docs&logoColor=%23ffffff&color=%231BC070)](https://github.com/neuralmagic/guidellm/tree/main/docs) [![License](https://img.shields.io/github/license/neuralmagic/guidellm.svg)](https://github.com/neuralmagic/guidellm/blob/main/LICENSE) [![PyPI Release](https://img.shields.io/pypi/v/guidellm.svg?label=PyPI%20Release)](https://pypi.python.org/pypi/guidellm) [![Python Versions](https://img.shields.io/badge/Python-3.8--3.12-orange)](https://pypi.python.org/pypi/guidellm) [![Nightly Build](https://img.shields.io/github/actions/workflow/status/neuralmagic/guidellm/nightly.yml?branch=main&label=Nightly%20Build)](https://github.com/neuralmagic/guidellm/actions/workflows/nightly.yml)

## Overview

<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/guidellm-user-flows-dark.png">
    <img alt="GuideLLM User Flows" src="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/guidellm-user-flows-light.png">
  </picture>
</p>

**GuideLLM** is a platform for evaluating and optimizing the deployment of large language models (LLMs). By simulating real-world inference workloads, GuideLLM enables users to assess the performance, resource requirements, and cost implications of deploying LLMs on various hardware configurations. This approach ensures efficient, scalable, and cost-effective LLM inference serving while maintaining high service quality.

### Key Features

- **Performance Evaluation:** Analyze LLM inference under different load scenarios to ensure your system meets your service level objectives (SLOs).
- **Resource Optimization:** Determine the most suitable hardware configurations for running your models effectively.
- **Cost Estimation:** Understand the financial impact of different deployment strategies and make informed decisions to minimize costs.
- **Scalability Testing:** Simulate scaling to handle large numbers of concurrent users without performance degradation.

## Getting Started

### Installation

Before installing, ensure you have the following prerequisites:

- OS: Linux or MacOS
- Python: 3.9 – 3.13

The latest GuideLLM release can be installed using pip:

```bash
pip install guidellm
```

Or from source code using pip:

```bash
pip install git+https://github.com/neuralmagic/guidellm.git
```

For detailed installation instructions and requirements, see the [Installation Guide](https://github.com/neuralmagic/guidellm/tree/main/docs/install.md).

### Quick Start

#### 1. Start an OpenAI Compatible Server (vLLM)

GuideLLM requires an OpenAI-compatible server to run evaluations. [vLLM](https://github.com/vllm-project/vllm) is recommended for this purpose. After installing vLLM on your desired server (`pip install vllm`), start a vLLM server with a Llama 3.1 8B quantized model by running the following command:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

For more information on starting a vLLM server, see the [vLLM Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

For information on starting other supported inference servers or platforms, see the [Supported Backends documentation](https://github.com/neuralmagic/guidellm/tree/main/docs/backends.md).

#### 2. Run a GuideLLM Benchmark

To run a GuideLLM benchmark, use the `guidellm benchmark` command with the target set to an OpenAI-compatible server. For this example, the target is set to 'http://localhost:8000', assuming that vLLM is active and running on the same server. Otherwise, update it to the appropriate location. By default, GuideLLM automatically determines the model available on the server and uses it. To target a different model, pass the desired name with the `--model` argument. Additionally, the `--rate-type` is set to `sweep`, which automatically runs a range of benchmarks to determine the minimum and maximum rates that the server and model can support. Each benchmark run under the sweep will run for 30 seconds, as set by the `--max-seconds` argument. Finally, `--data` is set to a synthetic dataset with 256 prompt tokens and 128 output tokens per request. For more arguments, supported scenarios, and configurations, jump to the [Configurations Section](#configurations) or run `guidellm benchmark --help`.

Now, to start benchmarking, run the following command:

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --rate-type sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128"
```

The above command will begin the evaluation and provide progress updates similar to the following: <img src= "https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/sample-benchmarks.gif"/>

#### 3. Analyze the Results

After the evaluation is completed, GuideLLM will summarize the results into three sections:

1. Benchmarks Metadata: A summary of the benchmark run and the arguments used to create it, including the server, data, profile, and more.
2. Benchmarks Info: A high-level view of each benchmark and the requests that were run, including the type, duration, request statuses, and number of tokens.
3. Benchmarks Stats: A summary of the statistics for each benchmark run, including the request rate, concurrency, latency, and token-level metrics such as TTFT, ITL, and more.

The sections will look similar to the following: <img alt="Sample GuideLLM benchmark output" src="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/sample-output.png" />

For more details about the metrics and definitions, please refer to the [Metrics documentation](https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/metrics.md).

#### 4. Explore the Results File

By default, the full results, including complete statistics and request data, are saved to a file `benchmarks.json` in the current working directory. This file can be used for further analysis or reporting, and additionally can be reloaded into Python for further analysis using the `guidellm.benchmark.GenerativeBenchmarksReport` class. You can specify a different file name and extension with the `--output` argument.

For more details about the supported output file types, please take a look at the [Outputs documentation](raw.githubusercontent.com/neuralmagic/guidellm/main/docs/outputs.md).

#### 5. Use the Results

The results from GuideLLM are used to optimize your LLM deployment for performance, resource efficiency, and cost. By analyzing the performance metrics, you can identify bottlenecks, determine the optimal request rate, and select the most cost-effective hardware configuration for your deployment.

For example, when deploying a chat application, we likely want to ensure that our time to first token (TTFT) and inter-token latency (ITL) are under certain thresholds to meet our service level objectives (SLOs) or service level agreements (SLAs). For example, setting TTFT to 200ms and ITL 25ms for the sample data provided in the example above, we can see that even though the server is capable of handling up to 13 requests per second, we would only be able to meet our SLOs for 99% of users at a request rate of 3.5 requests per second. If we relax our constraints on ITL to 50 ms, then we can meet the TTFT SLA for 99% of users at a request rate of approximately 10 requests per second.

For further details on determining the optimal request rate and SLOs, refer to the [SLOs documentation](https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/service_level_objectives.md).

### Configurations

GuideLLM offers a range of configurations through both the benchmark CLI command and environment variables, which provide default values and more granular controls. The most common configurations are listed below. A complete list is easily accessible, though, by running `guidellm benchmark --help` or `guidellm config` respectively.

#### Benchmark CLI

The `guidellm benchmark` command is used to run benchmarks against a generative AI backend/server. The command accepts a variety of arguments to customize the benchmark run. The most common arguments include:

- `--target`: Specifies the target path for the backend to run benchmarks against. For example, `http://localhost:8000`. This is required to define the server endpoint.

- `--model`: Allows selecting a specific model from the server. If not provided, it defaults to the first model available on the server. Useful when multiple models are hosted on the same server.

- `--processor`: Used only for synthetic data creation or when the token source configuration is set to local for calculating token metrics locally. It must match the model's processor or tokenizer to ensure compatibility and correctness. This supports either a HuggingFace model ID or a local path to a processor or tokenizer.

- `--data`: Specifies the dataset to use. This can be a HuggingFace dataset ID, a local path to a dataset, or standard text files such as CSV, JSONL, and more. Additionally, synthetic data configurations can be provided using JSON or key-value strings. Synthetic data options include:

  - `prompt_tokens`: Average number of tokens for prompts.
  - `output_tokens`: Average number of tokens for outputs.
  - `TYPE_stdev`, `TYPE_min`, `TYPE_max`: Standard deviation, minimum, and maximum values for the specified type (e.g., `prompt_tokens`, `output_tokens`). If not provided, will use the provided tokens value only.
  - `samples`: Number of samples to generate, defaults to 1000.
  - `source`: Source text data for generation, defaults to a local copy of Pride and Prejudice.

- `--data-args`: A JSON string used to specify the columns to source data from (e.g., `prompt_column`, `output_tokens_count_column`) and additional arguments to pass into the HuggingFace datasets constructor.

- `--data-sampler`: Enables applying `random` shuffling or sampling to the dataset. If not set, no sampling is used.

- `--rate-type`: Defines the type of benchmark to run (default sweep). Supported types include:

  - `synchronous`: Runs a single stream of requests one at a time. `--rate` must not be set for this mode.
  - `throughput`: Runs all requests in parallel to measure the maximum throughput for the server (bounded by GUIDELLM\_\_MAX_CONCURRENCY config argument). `--rate` must not be set for this mode.
  - `concurrent`: Runs a fixed number of streams of requests in parallel. `--rate` must be set to the desired concurrency level/number of streams.
  - `constant`: Sends requests asynchronously at a constant rate set by `--rate`.
  - `poisson`: Sends requests at a rate following a Poisson distribution with the mean set by `--rate`.
  - `sweep`: Automatically determines the minimum and maximum rates the server can support by running synchronous and throughput benchmarks, and then runs a series of benchmarks equally spaced between the two rates. The number of benchmarks is set by `--rate` (default is 10).

- `--max-seconds`: Sets the maximum duration (in seconds) for each benchmark run. If not specified, the benchmark will run until the dataset is exhausted or the `--max-requests` limit is reached.

- `--max-requests`: Sets the maximum number of requests for each benchmark run. If not provided, the benchmark will run until `--max-seconds` is reached or the dataset is exhausted.

- `--warmup-percent`: Specifies the percentage of the benchmark to treat as a warmup phase. Requests during this phase are excluded from the final results.

- `--cooldown-percent`: Specifies the percentage of the benchmark to treat as a cooldown phase. Requests during this phase are excluded from the final results.

- `--output-path`: Defines the path to save the benchmark results. Supports JSON, YAML, or CSV formats. If a directory is provided, the results will be saved as `benchmarks.json` in that directory. If not set, the results will be saved in the current working directory.

## Resources

### Documentation

Our comprehensive documentation offers detailed guides and resources to help you maximize the benefits of GuideLLM. Whether just getting started or looking to dive deeper into advanced topics, you can find what you need in our [documentation](https://github.com/neuralmagic/guidellm/tree/main/docs).

### Core Docs

- [**Installation Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/install.md) - This guide provides step-by-step instructions for installing GuideLLM, including prerequisites and setup tips.
- [**Backends Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/backends.md) - A comprehensive overview of supported backends and how to set them up for use with GuideLLM.
- [**Metrics Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/metrics.md) - Detailed explanations of the metrics used in GuideLLM, including definitions and how to interpret them.
- [**Outputs Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/outputs.md) - Information on the different output formats supported by GuideLLM and how to use them.
- [**Architecture Overview**](https://github.com/neuralmagic/guidellm/tree/main/docs/architecture.md) - A detailed look at GuideLLM's design, components, and how they interact.

### Supporting External Documentation

- [**vLLM Documentation**](https://vllm.readthedocs.io/en/latest/) - Official vLLM documentation provides insights into installation, usage, and supported models.

### Contribution Docs

We appreciate contributions to the code, examples, integrations, documentation, bug reports, and feature requests! Your feedback and involvement are crucial in helping GuideLLM grow and improve. Below are some ways you can get involved:

- [**DEVELOPING.md**](https://github.com/neuralmagic/guidellm/blob/main/DEVELOPING.md) - Development guide for setting up your environment and making contributions.
- [**CONTRIBUTING.md**](https://github.com/neuralmagic/guidellm/blob/main/CONTRIBUTING.md) - Guidelines for contributing to the project, including code standards, pull request processes, and more.
- [**CODE_OF_CONDUCT.md**](https://github.com/neuralmagic/guidellm/blob/main/CODE_OF_CONDUCT.md) - Our expectations for community behavior to ensure a welcoming and inclusive environment.

### Releases

Visit our [GitHub Releases page](https://github.com/neuralmagic/guidellm/releases) and review the release notes to stay updated with the latest releases.

### License

GuideLLM is licensed under the [Apache License 2.0](https://github.com/neuralmagic/guidellm/blob/main/LICENSE).

### Cite

If you find GuideLLM helpful in your research or projects, please consider citing it:

```bibtex
@misc{guidellm2024,
  title={GuideLLM: Scalable Inference and Optimization for Large Language Models},
  author={Neural Magic, Inc.},
  year={2024},
  howpublished={\url{https://github.com/neuralmagic/guidellm}},
}
```
