#!/usr/bin/env bash
set -euo pipefail

# Required environment variables
TARGET=${TARGET:-"http://localhost:8000"}
MODEL=${MODEL:-"neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"}
RATE_TYPE=${RATE_TYPE:-"sweep"}
DATA=${DATA:-"prompt_tokens=256,output_tokens=128"}
MAX_REQUESTS=${MAX_REQUESTS:-"100"}
MAX_SECONDS=${MAX_SECONDS:-""}

# Output configuration
OUTPUT_PATH=${OUTPUT_PATH:-"/results/guidellm_benchmark_results"}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-"json"}  # Can be json, yaml, or yml

# Build the command
CMD="guidellm benchmark --target \"${TARGET}\" --model \"${MODEL}\" --rate-type \"${RATE_TYPE}\" --data \"${DATA}\""

# Add optional parameters
if [ ! -z "${MAX_REQUESTS}" ]; then
    CMD="${CMD} --max-requests ${MAX_REQUESTS}"
fi

if [ ! -z "${MAX_SECONDS}" ]; then
    CMD="${CMD} --max-seconds ${MAX_SECONDS}"
fi

# Add output path with appropriate extension
if [ ! -z "${OUTPUT_PATH}" ]; then
    CMD="${CMD} --output-path \"${OUTPUT_PATH}.${OUTPUT_FORMAT}\""
fi

# Execute the command
echo "Running command: ${CMD}"
eval "${CMD}"
