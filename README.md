<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-logo-light.png">
    <img alt="GuideLLM Logo" src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-logo-dark.png" width=55%>
  </picture>
</p>

<h3 align="center">
Scale Efficiently: Evaluate and Optimize Your LLM Deployments for Real-World Inference Needs
</h3>

[![GitHub Release](https://img.shields.io/github/release/neuralmagic/guidellm.svg?label=Version)](https://github.com/neuralmagic/guidellm/releases) [![Documentation](https://img.shields.io/badge/Documentation-8A2BE2?logo=read-the-docs&logoColor=%23ffffff&color=%231BC070)](https://github.com/neuralmagic/guidellm/tree/main/docs) [![License](https://img.shields.io/github/license/neuralmagic/guidellm.svg)](https://github.com/neuralmagic/guidellm/blob/main/LICENSE) [![PyPi Release](https://img.shields.io/pypi/v/guidellm.svg?label=PyPi%20Release)](https://pypi.python.org/pypi/guidellm) [![Pypi Release](https://img.shields.io/pypi/v/guidellm-nightly.svg?label=PyPi%20Nightly)](https://pypi.python.org/pypi/guidellm-nightly) [![Python Versions](https://img.shields.io/pypi/pyversions/guidellm.svg?label=Python)](https://pypi.python.org/pypi/guidellm) [![Nightly Build](https://img.shields.io/github/actions/workflow/status/neuralmagic/guidellm/nightly.yml?branch=main&label=Nightly%20Build)](https://github.com/neuralmagic/guidellm/actions/workflows/nightly.yml)

## Overview

<p>
  <picture>
    <img alt="GuideLLM User Flows" src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/guidellm-user-flows.png">
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
  --model "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

The above command will begin the evaluation and output progress updates similar to the following: <img src="https://github.com/neuralmagic/guidellm/blob/main/docs/assets/sample-benchmark.gif" />

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

### Advanced Settings

GuideLLM provides various options to customize evaluations, including setting the duration of each benchmark run, the number of concurrent requests, and the request rate. For a complete list of options and advanced settings, see the [GuideLLM CLI Documentation](https://github.com/neuralmagic/guidellm/blob/main/docs/guides/cli.md).

Some common advanced settings include:

- `--rate-type`: The rate to use for benchmarking. Options include `sweep` (shown above), `synchronous` (one request at a time), `throughput` (all requests at once), `constant` (a constant rate defined by `--rate`), and `poisson` (a poisson distribution rate defined by `--rate`).
- `--data-type`: The data to use for the benchmark. Options include `emulated` (default shown above, emulated to match a given prompt and output length), `transformers` (a transformers dataset), and `file` (a {text, json, jsonl, csv} file with a list of prompts).
- `--max-seconds`: The maximum number of seconds to run each benchmark. The default is 120 seconds.
- `--max-requests`: The maximum number of requests to run in each benchmark.

## Resources

### Documentation

Our comprehensive documentation provides detailed guides and resources to help you get the most out of GuideLLM. Whether just getting started or looking to dive deeper into advanced topics, you can find what you need in our [full documentation](https://github.com/neuralmagic/guidellm/tree/main/docs).

### Core Docs

- [**Installation Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/install.md) - Step-by-step instructions to install GuideLLM, including prerequisites and setup tips.
- [**Architecture Overview**](https://github.com/neuralmagic/guidellm/tree/main/docs/architecture.md) - A detailed look at GuideLLM's design, components, and how they interact.
- [**CLI Guide**](https://github.com/neuralmagic/guidellm/tree/main/docs/guides/cli_usage.md) - Comprehensive usage information for running GuideLLM via the command line, including available commands and options.
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
