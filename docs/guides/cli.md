
# GuideLLM CLI User Guide 

For more details on setup and installation, see the Setup and [Installation](https://apps.neuralmagic.com/GuideLLM/README.MD/#Installation) sections. 

## GuideLLM Quickstart

To get started with GuideLLM, check out the [GuideLLM README](https://github.com/neuralmagic/guidellm/edit/main/README.md#getting-started). 

## GuideLLM CLI Details

**GuideLLM**  is a powerful tool for evaluating and optimizing the deployment of large language models (LLMs). The CLI has a large set of input arguments that give you advanced controls over all aspects of a workload that you want to run. 

### Input Metrics
The input arguments are split up into 3 sections: 

- **Workload Overview**
- **Workload Data**
- **Workload Type**

Once you fill out these arguments and run the command, GuideLLM will run the simulated workload. Note the time it takes to run can be set with <em>max_seconds</em>, but may also depend on the hardware and model. 

### Workload Overview

This section of input parameters covers what to actually benchmark including the target host location, model, and task. The full list of arguments and their defintions are presented below:

-   **--target** <em>(str)</em>: The target path or url for the backend to
evaluate. Ex: 'http://localhost:8000/v1'. [required]
    
	-   optional breakdown args if target isn't specified:
    
		-   **--host** <em>(str)</em>: The host URL for benchmarking.
    
		-   **--port** <em>(str)</em>: The port available for benchmarking.
    
-   **--backend** <em>[openai_server]</em>: The backend to use for benchmarking. The default is OpenAI Server enabling compatability with any server that follows the OpenAI spec including vLLM.
    
-   **--model** <em>(str)</em>: The model to use for benchmarking. If not provided, it will use the first available model provided the backend supports listing models.
        
-   **--task** <em>(str)</em>: The task to use for benchmarking.

-  **--output-path** <em>(str)</em>: The output path to save the output report to for loading later. Ex: guidance_report.json. The default is None, meaning no output is saved and results are only printed to the console.


### Workload Data

This section of input parameters covers the data arguments that need to be supplied such as a reference to the dataset and tokenizer. The list of arguments and their defintions are presented below:

-   **--data** <em>(str)</em>: The data source to use for benchmarking. Depending on the data-type, it should be a path to a data file containing prompts to run (ex: data.txt), a HuggingFace dataset name (ex: 'neuralmagic/LLM_compression_calibration'), or a configuration for emulated data (ex: 'prompt_tokens=128,generated_tokens=128'). [required]

-   **--data-type** <em>[emulated, file, transformers]</em>: The type of data to use for benchmarking. Use 'emulated' for synthetic data, 'file' for a file, or 'transformers' for a HuggingFace dataset. Specify the data source with the --data flag.  [required]

-   **--tokenizer** <em>(str)</em>: The tokenizer to use for calculating the number of prompt tokens. This should match the tokenizer used by the model.By default, it will use the --model flag to determine the tokenizer. If not provided and the model is not available, will raise an error. Ex: 'neuralmagic/Meta-Llama-3.1-8B-quantized.w8a8'

### Workload Type

This section of input parameters covers the type of workload that you want to run to represent the type of load you expect on your server in production such as rate-per-second and the frequency of requests. The full list of arguments and their definitions are presented below:

-   **--rate-type**  <em>[sweep|synchronous|throughput|constant|poisson] </em>: The type of request rate to use for benchmarking. Use sweep to run a full range from synchronous to throughput (default), synchronous for sending requests one after the other, throughput to send requests as fast as possible, constant for a fixed request rate, or poisson for a real-world variable request rate.
    
-   **--rate** <em>(float)</em>: The request rate to use for constant and poisson rate types. To run with multiple, specific, rates, provide the flag multiple times. Ex. --rate 1 --rate 2 --rate 5
    
-   **--max-seconds** <em>(integer)</em>: The maximum number of seconds for each benchmark run. Either max-seconds, max- requests, or both must be set. The default is 120 seconds. Note, this is the maximum time for each rate supplied, not the total time. This value should be large enough to allow for the server's performance to stabilize.
    
-   **--max-requests** <em>(integer)</em>: The maximum number of requests for each benchmark run. Either max-seconds, max- requests, or both must be set. Note, this is the maximum number of requests for each rate supplied, not the total number of requests. This value should be large enough to allow for the server's performance to stabilize.

### Output Metrics via GuideLLM Benchmarks Report

Once your GuideLLM run is complete, the output metrics are displayed as a GuideLLM Benchmarks Report via the Terminal in the following 4 sections: 

- **Requests Data by Benchmark**
- **Tokens Data by Benchmark**
- **Performance Stats by Benchmark**
- **Performance Summary by Benchmark**

The GuideLLM Benchmarks Report surfaces key LLM metrics to help you determine the health and performance of your inference server. You can use the numbers generated by the GuideLLM Benchmarks Report to make decisions around server request processing, Service Level Objective (SLO) success/failure for your task, general model performance, and hardware impact.

### Requests Data by Benchmark 

This section shows the request statistics for the benchmarks that were run. Request Data statistics highlight the details of the requests hitting the inference server. Viewing this information is essential to understand the health of your server processing requests sent by GuideLLM and can surface potential issues in your inference serving pipeline including software and hardware issues. 

<p>
  <picture>
    <img alt="GuideLLM Requests Data by Benchmark" src="https://github.com/neuralmagic/guidellm/blob/rgreenberg1-patch-1/docs/assets/request_data.png">
  </picture>
</p>

This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Requests Completed:** the number of successful requests handled 
- **Requests Failed:** the number of failed requests
- **Duration (sec):** the time taken to run the specific benchmark, determined by <em>max_seconds</em> 
- **Start Time (HH:MI:SS):** local timestamp the GuideLLM benchmark started 
- **End Time (HH:MI:SS):** local timestamp the GuideLLM benchmark ended 


### Tokens Data by Benchmark
This section shows the prompt and output token distribution statistics for the benchmarks that were run. Token Data statistics highlight the details of your dataset in terms of prompts and generated outputs from the model. Viewing this information is integral to understanding model performance on your task and to ensure you are able to hit SLOs required to guarentee a good user experience from your application. 

<p>
  <picture>
    <img alt="GuideLLM Requests Data by Benchmark" src="https://github.com/neuralmagic/guidellm/blob/rgreenberg1-patch-1/docs/assets/tokens_data.png">
  </picture>
</p>

This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Prompt (token length):** the average length of prompt tokens 
- **Prompt (1%, 5%, 50%, 95%, 99%):** Distribution of prompt token length
- **Output (token length):** the average length of output tokens
- **Output (1%, 5%, 50%, 95%, 99%):** Distribution of output token length

### Performance Stats by Benchmark
This section shows the LLM peformance statistics for the benchmarks that were run. Performance Statistics highlight the performance of the model across the key LLM performance metrics including: Request Latency, Time to First Token (TTFT), Inter Token Latench (ITL or TPOT), and Output Token Throughput. Viewing these key metrics are integral to ensuring the performance of your inference server for your task on your designated hardware where you are running your inference server. 

<p>
  <picture>
    <img alt="GuideLLM Requests Data by Benchmark" src="https://github.com/neuralmagic/guidellm/blob/rgreenberg1-patch-1/docs/assets/perf_stats.png">
  </picture>
</p>


This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Request Latency [1%, 5%, 10%, 50%, 90%, 95%, 99%] (sec)**: the time it takes from submitting a query to receiving the full response, including the performance of your queueing/batching mechanisms and network latencies
-  **Time to First Token [1%, 5%, 10%, 50%, 90%, 95%, 99%] (ms)**: the time it takes from submitting the query to receiving the first token (if the response is not empty); often abbreviated as TTFT
- **Inter Token Latency [1%, 5%, 10%, 50%, 90% 95%, 99%] (ms)**: the time between consecutive tokens and is also known as time per output token (TPOT)
-  **Output Token Throughput (tokens/sec)**: the total output tokens per second throughput, accounting for all the requests happening simultaneously


### Performance Summary by Benchmark
This section shows the averages for the LLM peformance statistics for the benchmarks that were run. The average Performance Statistics provide an overall summary of the model performance across the key LLM performance metrics. Viewing these summary metrics are integral to ensuring the performance of your inference server for your task on your designated hardware where you are running your inference server. 

<p>
  <picture>
    <img alt="GuideLLM Requests Data by Benchmark" src="https://github.com/neuralmagic/guidellm/blob/rgreenberg1-patch-1/docs/assets/perf_summary.png">
  </picture>
</p>


This table includes:
- **Benchmark:** Synchronous or Asynchronous@X req/sec
- **Request Latency (sec)**: the average time it takes from submitting a query to receiving the full response, including the performance of your queueing/batching mechanisms and network latencies
-  **Time to First Token (ms)**: the average time it takes from submitting the query to receiving the first token (if the response is not empty); often abbreviated as TTFT
- **Inter Token Latency (ms)**: the average time between consecutive tokens and is also known as time per output token (TPOT)
-  **Output Token Throughput (tokens/sec)**: the total average output tokens per second throughput, accounting for all the requests happening simultaneously


## Report a Bug

To report a bug, file an issue on [GitHub Issues](https://github.com/neuralmagic/guidellm/issues). 



