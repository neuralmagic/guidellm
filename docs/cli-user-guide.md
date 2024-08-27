
# GuideLLM CLI User Guide 

For more details on setup and installation, see the Setup and [Installation](https://github.com/neuralmagic/guidellm?tab=readme-ov-file#installation) sections. 

## About GuideLLM

The GuideLLM CLI is a performance benchmarking tool to enable you to evaluate and visualize your LLM inference serving performance before your deploy to production. GuideLLM runs via the CLI and takes in a wide array of input arguments to enable you to configure your workload to run benchmarks to the specifications that you desire. This ultimately provides the ability to understand bottlenecks in your inference serving pipeline and make changes before your users are effected by your LLM application. 

## GuideLLM CLI Quickstart 

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

	

## GuideLLM CLI Details
### Input Metrics
The input arguments are split up into 3 sections: 

- **Workload Overview**
- **Workload Data**
- **Workload Type**

Once you fill out these arguments and run the command, GuideLLM will run the simulated workload. Note the time it takes to run can be set with <em>max_seconds</em>, but may also depend on the hardware and model. 

### Workload Overview

This section of input parameters covers what to actually benchmark including the target host location, model, and task. The full list of arguments and their defintions are presented below:

-   **--target** <em>(str, default: localhost with chat completions api for VLLM)</em>: Target for benchmarking 
    
	-   optional breakdown args if target isn't specified:
    
		-   **--host** <em>(str)</em>: Host URL for benchmarking
    
		-   **--port** <em>(str)</em>: Port available for benchmarking
    
-   **--backend** <em>(str, default: server_openai [vllm, TGI, llama.cpp, DeepSparse, and many popular servers match this format])</em>: Backend type for benchmarking
    
-   **--model** <em>(str, default: auto populated from vllm server)</em>: Model being used for benchmarking, running on the inference server
        
-   **--task** <em>(str), optional)</em>: Task to use for benchmarking

-  **--output-path** <em>(str), optional)</em>: Path to save report report to



### Workload Data

This section of input parameters covers the data arguments that need to be supplied such as a reference to the dataset and tokenizer. The list of arguments and their defintions are presented below:

-   **--data** <em>(str)</em>: Data file or alias for benchmarking

-   **--data-type** <em>(ENUM, default: emulated; [file, transformers])</em>: The type of data given for benchmarking

-   **--tokenizer** <em>(str)</em>: Tokenizer to use for benchmarking

### Workload Type

This section of input parameters covers the type of workload that you want to run to represent the type of load you expect on your server in production such as rate-per-second and the frequency of requests. The full list of arguments and their definitions are presented below:

-   **--rate-type**  <em>(ENUM, default: sweep; [serial, constant, poisson] where sweep will cover a range of constant request rates and ensure saturation of server, serial will send one request at a time, constant will send a constant request rate, poisson will send a request rate sampled from a poisson distribution at a given mean) </em>: Type of rate generation for benchmarking
    
-   **--rate** <em>(float)</em>: Rate to use for constant and poisson rate types
    
-   **--max-seconds** <em>(integer)</em>: Number of seconds to result each request rate at
    
-   **--max-requests** <em>(integer)</em>: Number of requests to send for each rate
    
### Output Metrics via GuideLLM Benchmarks Report

Once your GuideLLM run is complete, the output metrics are displayed as a GuideLLM Benchmarks Report via the Terminal in the following 4 sections: 

- **Requests Data by Benchmark**
- **Tokens Data by Benchmark**
- **Performance Stats by Benchmark**
- **Performance Summary by Benchmark**

The GuideLLM Benchmarks Report surfaces key LLM metrics to help you determine the health and performance of your inference server. You can use the numbers generated by the GuideLLM Benchmarks Report to make decisions around server request processing, Service Level Objective (SLO) success/failure for your task, general model performance, and hardware impact.

### Requests Data by Benchmark 

This section shows the request statistics for the benchmarks that were run. Request Data statistics highlight the details of the requests hitting the inference server. Viewing this information is essential to understand the health of your server processing requests sent by GuideLLM and can surface potential issues in your inference serving pipeline including software and hardware issues. 

This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Requests Completed:** the number of successful requests handled 
- **Requests Failed:** the number of failed requests
- **Duration (sec):** the time taken to run the specific benchmark, determined by <em>max_seconds</em> 
- **Start Time (HH:MI:SS):** local timestamp the GuideLLM benchmark started 
- **End Time (HH:MI:SS):** local timestamp the GuideLLM benchmark ended 


### Tokens Data by Benchmark
This section shows the prompt and output token distribution statistics for the benchmarks that were run. Token Data statistics highlight the details of your dataset in terms of prompts and generated outputs from the model. Viewing this information is integral to understanding model performance on your task and to ensure you are able to hit SLOs required to guarentee a good user experience from your application. 

This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Prompt (token length):** the average length of prompt tokens 
- **Prompt (1%, 5%, 50%, 95%, 99%):** Distribution of prompt token length
- **Output (token length):** the average length of output tokens
- **Output (1%, 5%, 50%, 95%, 99%):** Distribution of output token length

### Performance Stats by Benchmark
This section shows the LLM peformance statistics for the benchmarks that were run. Performance Statistics highlight the performance of the model across the key LLM performance metrics including: Request Latency, Time to First Token (TTFT), Inter Token Latench (ITL or TPOT), and Output Token Throughput. Viewing these key metrics are integral to ensuring the performance of your inference server for your task on your designated hardware where you are running your inference server. 

This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Request Latency [1%, 5%, 10%, 50%, 90%, 95%, 99%] (sec)**: the time it takes from submitting a query to receiving the full response, including the performance of your queueing/batching mechanisms and network latencies
-  **Time to First Token [1%, 5%, 10%, 50%, 90%, 95%, 99%] (ms)**: the time it takes from submitting the query to receiving the first token (if the response is not empty); often abbreviated as TTFT
- **Inter Token Latency [1%, 5%, 10%, 50%, 90% 95%, 99%] (ms)**: the time between consecutive tokens and is also known as time per output token (TPOT)
-  **Output Token Throughput (tokens/sec)**: the total output tokens per second throughput, accounting for all the requests happening simultaneously


### Performance Summary by Benchmark
This section shows the averages for the LLM peformance statistics for the benchmarks that were run. The average Performance Statistics provide an overall summary of the model performance across the key LLM performance metrics. Viewing these summary metrics are integral to ensuring the performance of your inference server for your task on your designated hardware where you are running your inference server. 

This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Request Latency (sec)**: the average time it takes from submitting a query to receiving the full response, including the performance of your queueing/batching mechanisms and network latencies
-  **Time to First Token (ms)**: the average time it takes from submitting the query to receiving the first token (if the response is not empty); often abbreviated as TTFT
- **Inter Token Latency (ms)**: the average time between consecutive tokens and is also known as time per output token (TPOT)
-  **Output Token Throughput (tokens/sec)**: the total average output tokens per second throughput, accounting for all the requests happening simultaneously


## Report a Bug

To report a bug, file an issue on [GitHub Issues](https://github.com/neuralmagic/guidellm/issues). 
