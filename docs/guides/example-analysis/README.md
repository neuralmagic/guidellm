# GuideLLM Example Analysis

This directory contains example analysis script for GuideLLM performance testing.

## Running Benchmarks in Kubernetes

To run comprehensive GuideLLM benchmarks in Kubernetes, follow the instructions in the [k8s/README.md](../k8s/README.md). This will help you:

- Set up the necessary Kubernetes environment
- Configure benchmark parameters
- Execute the benchmarks
- Collect performance data

## Analyzing Results

### Using the Analysis Script

The [analyze_benchmarks.py](./analyze_benchmarks.py) script processes benchmark YAML output and generates visualizations and statistics. To use it:

1. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the GuideLLM benchmark YAML file from the Kubernetes guidellm-job pod is copied to your local environment.

   ```bash
   # From the k8s/README.md instructions
   kubectl cp <pod-name>:/path/to/benchmark.yaml ./llama32-3b.yaml
   ```

3. Run the analysis script (make sure the YAML file is in the same directory):

   ```bash
   python analyze_benchmarks.py
   ```

The script will:

- Process the benchmark YAML file
- Generate visualizations in the `benchmark_plots` directory
- Create a CSV file with processed metrics
- Print summary statistics
