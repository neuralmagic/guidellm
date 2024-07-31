
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
Benchmark and visualize LLM inference serving workloads
</h3>


<p>
    <a href="https://docs.neuralmagic.com/sparseml/">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://neuralmagic.com/community/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/issues">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/actions/workflows/test-check.yaml">
        <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/sparseml/Test%20Checks/main?label=build&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparseml.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparseml.svg?color=lightgray&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
        <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25>
    </a>
     <a href="https://medium.com/limitlessai">
        <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25>
    </a>
    <a href="https://twitter.com/neuralmagic">
        <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25>
    </a>
</p>

## Overview

Diagram HERE


**GuideLLM is a tool designed to optimize the deployment of large language models (LLMs) through benchmarking simulated workloads on an inference server.** By evaluating model performance and cost under various workload configurations,it helps users determine the optimal hardware resources needed to meet service level objectives (SLOs). This ensures efficient and cost-effective LLM inference serving without compromising user experience.

GuideLLM can help you:

1.  Evaluate model performance to ensure your configuration can serve scaling users with confidence. 
2.  Guage GPU hardware requirements to properly serve your LLM application at maximum usage. 
3.  Understand the cost implications of different deployment configurations and models. 


## Getting Started

### Setup 

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
**Note:** Before you use GuideLLM, you'll need to have a vLLM inference server running on the same network or cluster as GuideLLM. The location of the servers HTTP/Python API will need to be available and have permission for to be passed in as an argument in order to run GuideLLM. To see the recommended hardware setup to run GuideLLM + vLLM, see the Setup Section below. 

Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

Visit the official [documentation](https://vllm.readthedocs.io/en/latest/) to learn more.
- [Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Supported Models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)

### Quick Start
nice visual of the report that gets generated. 

To get started with GuideLLM with Llama-7B-Chat, you can run the following CLI command: 
```bash
guidellm.padsfa asdfasdf 
```

## Demos

## Using GuideLLM

### Input Metrics
GuideLLM runs via the CLI and takes in a wide array of input arguments to enable you to configure your workload to run benchmarks to the specficiations that you desire. The input arguments are split up into 3 sections: 

- **Workload Overview**
- **Workload Data**
- **Workload Type**

Once you fill out these arguments and run the command, GuideLLM will take between 5-20 minutes (depending on the hardware and model) to run the simulated workload. 

### Workload Overview

This section of input parameters covers what to actually benchmark including the target host location, model, and task. The full list of arguments and their defintions are presented below:

-   **--target** (str, default: localhost with chat completions api for VLLM)
    
	-   optional breakdown args if target isn't specified:
    
		-   **--host** (str)
    
		-   **--port** (str)
    
		-   **--path** (str)
    
-   **--backend** (str, default: server_openai [vllm matches this], room for expansion for either python process benchmarking or additional servers for comparisons)
    
-   **--model** (str, default: auto populated from vllm server)
    
-   **--model-*** (any, additional arguments that should be passed to the benchmark request)
    
-   **--task** (ENUM of tasks or task config file/str, sets default data if supplied and data is not and sets the default constraints such as prefill and generation tokens as well as limits around min and max number of tokens for prompts / generations)


### Workload Data

This section of input parameters covers the data arguments that need to be supplied such as a reference to the dataset and tokenizer. The list of arguments and their defintions are presented below:

-   **--data** (str: alias, text file, config, default: auto populate based on task)
    
-   **--tokenizer** (str: HF/NM model alias or tokenizer file, default pull from model/server or None if not loadable -- used to calculate number of tokens, if not then will fall back on words)

### Workload Type

This section of input parameters covers the type of workload that you want to run to represent the type of load you expect on your server in production such as rate-per-second and the frequency of requests. The full list of arguments and their defintions are presented below:

-   **--rate-type** (ENUM [sweep, serial, constant, poisson] where sweep will cover a range of constant request rates and ensure saturation of server, serial will send one request at a time, constant will send a constant request rate, poisson will send a request rate sampled from a poisson distribution at a given mean)
    
-   **--rate** (float, used for constant and poisson rate types)
    
-   **--num-seconds** (number of seconds to benchmark each request rate at)
    
-   --num-requests (alternative to number of seconds to send a number of requests for each rate)
    
-   --num-warmup (the number of warmup seconds or requests to run before benchmarking starts)

### Output Metrics

Once your GuideLLM run is complete, the output metrics are displayed via the CLI in 3 sections:

- **Workload Report**
- **Workload Details**
- **Metrics Details**

tbd

## Using vLLM

TBD 



## Using Workload Report UI

GuideLLM has a GUI that gets populated upon the completion of the CLI - todo next week. 




## Community

### Contribute

We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/guidellm/blob/main/CONTRIBUTING.md)

- [DEVELOPING.md](DEVELOPING.md) - Development guide for contributors.
- [CONTRIBUTING.md](CONTRIBUTING.md) - Guidelines for contributing to the project.
- [.MAINTAINERS](.MAINTAINERS) - Active maintainers of the project.

### Citation

Find this project useful in your research or other communications? Please consider citing - TBD


### License

The project is licensed under the [Apache License Version 2.0.](https://github.com/neuralmagic/sparseml/blob/main/LICENSE)
