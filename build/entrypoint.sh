#!/usr/bin/env bash
set -euo pipefail

# If we receive any arguments switch to guidellm command
if [ $# -gt 0 ]; then
    echo "Running command: guidellm $*"
    exec guidellm "$@"
fi

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
    CMD="${CMD} --output-path \"${OUTPUT_PATH}\""
fi

# Execute the command
echo "Running command: ${CMD}"
eval "${CMD}"
