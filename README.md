<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
Scale Efficiently: Evaluate and Optimize Your LLM Deployments for Real-World Inference
</h3>

[![GitHub Release](https://img.shields.io/github/release/neuralmagic/guidellm.svg?label=Version)](https://github.com/neuralmagic/guidellm/releases) [![Documentation](https://img.shields.io/badge/Documentation-8A2BE2?logo=read-the-docs&logoColor=%23ffffff&color=%231BC070)](https://github.com/neuralmagic/guidellm/tree/main/docs) [![License](https://img.shields.io/github/license/neuralmagic/guidellm.svg)](https://github.com/neuralmagic/guidellm/blob/main/LICENSE) [![PyPI Release](https://img.shields.io/pypi/v/guidellm.svg?label=PyPI%20Release)](https://pypi.python.org/pypi/guidellm) [![Pypi Release](https://img.shields.io/pypi/v/guidellm-nightly.svg?label=PyPI%20Nightly)](https://pypi.python.org/pypi/guidellm-nightly) [![Python Versions](https://img.shields.io/pypi/pyversions/guidellm.svg?label=Python)](https://pypi.python.org/pypi/guidellm) [![Nightly Build](https://img.shields.io/github/actions/workflow/status/neuralmagic/guidellm/nightly.yml?branch=main&label=Nightly%20Build)](https://github.com/neuralmagic/guidellm/actions/workflows/nightly.yml)

## Overview

<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-user-flows-dark.png">
    <img alt="GuideLLM User Flows" src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-user-flows-light.png">
  </picture>
</p>

**GuideLLM** is a powerful tool for evaluating and optimizing the deployment of large language models (LLMs). By simulating real-world inference workloads, GuideLLM helps users gauge the performance, resource needs, and cost implications of deploying LLMs on various hardware configurations. This ensures efficient, scalable, and cost-effective LLM inference serving while maintaining high service quality.

### Key Features

- **Performance Evaluation:** Analyze LLM inference under different load scenarios to ensure your system meets your service level objectives (SLOs).
- **Resource Optimization:** Determine the most suitable hardware configurations for running your models effectively.
- **Cost Estimation:** Understand the financial impact of different deployment strategies and make informed decisions to minimize costs.
- **Scalability Testing:** Simulate scaling to handle large numbers of concurrent users without degradation in performance.

## Getting Started

### Installation

Before installing, ensure you have the following prerequisites:

- OS: Linux or MacOS
- Python: 3.8 – 3.12

GuideLLM is available on PyPI and can be installed using `pip`:

```bash
pip install guidellm
```

For detailed installation instructions and requirements, see the [Installation Guide](https://github.com/neuralmagic/guidellm/tree/main/docs/install.md).

### Quick Start

#### 1. Start an OpenAI Compatible Server (vLLM)

GuideLLM requires an OpenAI-compatible server to run evaluations. It's recommended that [vLLM](https://github.com/vllm-project/vllm) be used for this purpose. To start a vLLM server with a Llama 3.1 8B quantized model, run the following command:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

For more information on starting a vLLM server, see the [vLLM Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

#### 2. Run a GuideLLM Evaluation

To run a GuideLLM evaluation, use the `guidellm` command with the appropriate model name and options on the server hosting the model or one with network access to the deployment server. For example, to evaluate the full performance range of the previously deployed Llama 3.1 8B model, run the following command:

```bash
guidellm \
  --target "http://localhost:8000/v1" \
  --model "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16" \
  --data-type emulated \
  --data "prompt_tokens=512,generated_tokens=128"
```

The above command will begin the evaluation and output progress updates similar to the following (if running on a different server, be sure to update the target!): <img src="https://raw.githubusercontent.com/neuralmagic/guidellm/main/docs/assets/sample-benchmarks.gif" />

Notes:

- The `--target` flag specifies the server hosting the model. In this case, it is a local vLLM server.
- The `--model` flag specifies the model to evaluate. The model name should match the name of the model deployed on the server
- By default, GuideLLM will run a `sweep` of performance evaluations across different request rates, each lasting 120 seconds. The results will be saved to a local directory.

#### 3. Analyze the Results

After the evaluation is completed, GuideLLM will output a summary of the results, including various performance metrics. The results will also be saved to a local directory for further analysis.

The output results will start with a summary of the evaluation, followed by the requests data for each benchmark run. For example, the start of the output will look like the following:

<img alt="Sample GuideLLM benchmark start output" src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/sample-output-start.png" />

The end of the output will include important performance summary metrics such as request latency, time to first token (TTFT), inter-token latency (ITL), and more:

<img alt="Sample GuideLLM benchmark end output" src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/sample-output-end.png" />

### Configurations

GuideLLM provides various CLI and environment options to customize evaluations, including setting the duration of each benchmark run, the number of concurrent requests, and the request rate.

Some common configurations for the CLI include:

- `--rate-type`: The rate to use for benchmarking. Options include `sweep`, `synchronous`, `throughput`, `constant`, and `poisson`.
  - `--rate-type sweep`: (default) Sweep runs through the full range of performance for the server. Starting with a `synchronous` rate first, then `throughput`, and finally 10 `constant` rates between the min and max request rate found.
  - `--rate-type synchronous`: Synchronous runs requests in a synchronous manner, one after the other.
  - `--rate-type throughput`: Throughput runs requests in a throughput manner, sending requests as fast as possible.
  - `--rate-type constant`: Constant runs requests at a constant rate. Specify the rate in requests per second with the `--rate` argument. For example, `--rate 10` or multiple rates with `--rate 10 --rate 20 --rate 30`.
  - `--rate-type poisson`: Poisson draws from a poisson distribution with the mean at the specified rate, adding some real-world variance to the runs. Specify the rate in requests per second with the `--rate` argument. For example, `--rate 10` or multiple rates with `--rate 10 --rate 20 --rate 30`.
- `--data-type`: The data to use for the benchmark. Options include `emulated`, `transformers`, and `file`.
  - `--data-type emulated`: Emulated supports an EmulationConfig in string or file format for the `--data` argument to generate fake data. Specify the number of prompt tokens at a minimum and optionally the number of output tokens and other params for variance in the length. For example, `--data "prompt_tokens=128"`, `--data "prompt_tokens=128,generated_tokens=128"`, or `--data "prompt_tokens=128,prompt_tokens_variance=10"`.
  - `--data-type file`: File supports a file path or URL to a file for the `--data` argument. The file should contain data encoded as a CSV, JSONL, TXT, or JSON/YAML file with a single prompt per line for CSV, JSONL, and TXT or a list of prompts for JSON/YAML. For example, `--data "data.txt"` where data.txt contents are `"prompt1\nprompt2\nprompt3"`.
  - `--data-type transformers`: Transformers supports a dataset name or dataset file path for the `--data` argument. For example, `--data "neuralmagic/LLM_compression_calibration"`.
- `--max-seconds`: The maximum number of seconds to run each benchmark. The default is 120 seconds.
- `--max-requests`: The maximum number of requests to run in each benchmark.

For a full list of supported CLI arguments, run the following command:

```bash
guidellm --help
```

For a full list of configuration options, run the following command:

```bash
guidellm-config
```

For further information, see the [GuideLLM Documentation](#Documentation).

## Resources

### Documentation

Our comprehensive documentation provides detailed guides and resources to help you get the most out of GuideLLM. Whether just getting started or looking to dive deeper into advanced topics, you can find what you need in our [full documentation](https://github.com/neuralmagic/guidellm/tree/main/docs).

### Core Docs

- [**Installation Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/install.md) - Step-by-step instructions to install GuideLLM, including prerequisites and setup tips.
- [**Architecture Overview**](https://github.com/neuralmagic/guidellm/tree/main/docs/architecture.md) - A detailed look at GuideLLM's design, components, and how they interact.
- [**CLI Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/guides/cli.md) - Comprehensive usage information for running GuideLLM via the command line, including available commands and options.
- [**Configuration Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/guides/configuration.md) - Instructions on configuring GuideLLM to suit various deployment needs and performance goals.

### Supporting External Documentation

- [**vLLM Documentation**](https://vllm.readthedocs.io/en/latest/) - Official vLLM documentation provides insights into installation, usage, and supported models.

### Releases

Stay updated with the latest releases by visiting our [GitHub Releases page](https://github.com/neuralmagic/guidellm/releases) and reviewing the release notes.

### License

GuideLLM is licensed under the [Apache License 2.0](https://github.com/neuralmagic/guidellm/blob/main/LICENSE).

## Community

### Contribute

We appreciate contributions to the code, examples, integrations, documentation, bug reports, and feature requests! Your feedback and involvement are crucial in helping GuideLLM grow and improve. Below are some ways you can get involved:

- [**DEVELOPING.md**](https://github.com/neuralmagic/guidellm/blob/main/DEVELOPING.md) - Development guide for setting up your environment and making contributions.
- [**CONTRIBUTING.md**](https://github.com/neuralmagic/guidellm/blob/main/CONTRIBUTING.md) - Guidelines for contributing to the project, including code standards, pull request processes, and more.
- [**CODE_OF_CONDUCT.md**](https://github.com/neuralmagic/guidellm/blob/main/CODE_OF_CONDUCT.md) - Our expectations for community behavior to ensure a welcoming and inclusive environment.

### Join

We invite you to join our growing community of developers, researchers, and enthusiasts passionate about LLMs and optimization. Whether you’re looking for help, want to share your own experiences, or stay up to date with the latest developments, there are plenty of ways to get involved:

- [**Neural Magic Community Slack**](https://neuralmagic.com/community/) - Join our Slack channel to connect with other GuideLLM users and developers. Ask questions, share your work, and get real-time support.
- [**GitHub Issues**](https://github.com/neuralmagic/guidellm/issues) - Report bugs, request features, or browse existing issues. Your feedback helps us improve GuideLLM.
- [**Subscribe to Updates**](https://neuralmagic.com/subscribe/) - Sign up to receive the latest news, announcements, and updates about GuideLLM, webinars, events, and more.
- [**Contact Us**](http://neuralmagic.com/contact/) - Use our contact form for more general questions about Neural Magic or GuideLLM.

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
