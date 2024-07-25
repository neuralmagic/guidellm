
# GuideLLM CLI User Guide

To use the GuideLLM CLI, make sure you have the proper setup and installation of GuideLLM and vLLM. 

For more details on setup and installation, see the Setup and [Installation](https://apps.neuralmagic.com/GuideLLM/README.MD/#Installation) sections. 


## GuideLLM CLI Overview

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

To use vLLM with 

## Demos

To use GuideLLM and vLLM for a chat task with Llama 7B, first we start by grabbing the model stub and dataset stub from HuggingFace... 




