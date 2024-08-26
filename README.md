
<!--
Copyright (c) 2024 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="GuideLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Evaulate and visualize LLM inference serving workloads
</h3>


<p>
    <a href="https://github.com/neuralmagic/guidellm">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://neuralmagic.com/community/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://github.com/neuralmagic/guidellm/issues">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height=25>
    </a>
    <a href="https://github.com/neuralmagic/guidellm/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparseml.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/guidellm/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparseml.svg?color=lightgray&style=for-the-badge" height=25>
    </a>
    <a href="https://twitter.com/neuralmagic">
        <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25>
    </a>
</p>

## Overview

Diagram HERE


**GuideLLM is a performance benchmarking tool designed to evaluate and visualize your LLM inference serving performance before your deploy to production.** By evaluating model performance and cost under various workload configurations, it helps users determine the optimal hardware resources needed to meet service level objectives (SLOs). This ensures efficient and cost-effective LLM inference serving without compromising the user experience.

GuideLLM can help you:

-  Evaluate model performance to ensure your configuration can serve scaling with confidence. 
-  Guage GPU hardware requirements to properly serve your LLM application at maximum usage. 
-  Understand the cost implications of different deployment configurations and models. 


## Setup 

Verify that you have the correct software and hardware to run GuideLLM.

**Requirements:**  
-   OS: Linux
-   Python: 3.8 – 3.11 
-   GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

If you encounter issues running GuideLLM,  [file a GitHub issue](https://github.com/neuralmagic/guidellm/issues).

### Installation

Install GuideLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install guidellm
```
**Note:** Before you use GuideLLM, you'll need to have an inference server running on the same network or cluster as GuideLLM such as vLLM, TGI, llamma.cpp, etc. The location of the servers HTTP/Python API will need to be available and have permission for to be passed in as an argument in order to run GuideLLM. To see the recommended hardware setup to run GuideLLM + vLLM, see the Quickstart w/ vLLM Section below. 


## Get Started

### Quick Start w/ vLLM
Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

Visit the official [documentation](https://vllm.readthedocs.io/en/latest/) to learn more.
- [Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Supported Models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)

### End-to-End Example

To get started with GuideLLM  + vLLM running [Neural Magic's Meta-Llama-3.1-8B-Instruct-quantized.w4a16](https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16) model, run the following:

1. Deploy your model with vLLM. 
2. Pip install GuideLLM with: `pip install guidellm`. 
3. Once installed, cd into the GuideLLM directory with: `cd guidellm`. 
4. In order to avoid compatibility issues with GuideLLM, it is recommended to run it in a fresh [virtual environment](https://docs.python.org/3/library/venv.html).
5. Create a virtual environment with: `virtualenv -p python3.8 venv`. 
6. Now activate the venv with: source `venv/bin/activate`.
7. To run GuideLLM on Neural Magic's W4A16 quantized Llama 3.1 8B Instruct model, you can run the following command:
```bash
python src/guidellm/main.py --target "http://HOST_URL/v1" --model "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16" --data-type emulated --data "prompt_tokens=512,generated_tokens=128" --max-seconds 60
```
8. Unless otherwise specified, The above command will generate a GuideLLM Benchmarks Report using the default values for arguments. 

### User Guides
Deep dives into advanced usage of `guidellm`:
[GuideLLM CLI User Guide](https://link.com)


## Questions / Contribution

- If you have any questions or requests open an [issue](https://github.com/neuralmagic/guidellm/issues) and we will add an example or documentation.
- We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here](CONTRIBUTING.md).
  
## Resources

- [LICENSE](LICENSE) - License information for the project.
- [DEVELOPING.md](DEVELOPING.md) - Development guide for contributors.
- [CONTRIBUTING.md](CONTRIBUTING.md) - Guidelines for contributing to the project.
- [.MAINTAINERS](.MAINTAINERS) - Active maintainers of the project.
