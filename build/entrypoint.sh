#!/usr/bin/env bash
set -euo pipefail

# Path to the guidellm binary
guidellm_bin="/opt/guidellm/bin/guidellm"

# If we receive any arguments switch to guidellm command
if [ $# -gt 0 ]; then
    echo "Running command: guidellm $*"
    exec $guidellm_bin "$@"
fi

# NOTE: Bash vec + exec prevent shell escape issues
CMD=("${guidellm_bin}" "benchmark" "--target" "${TARGET}" "--model" "${MODEL}" "--rate-type" "${RATE_TYPE}" "--data" "${DATA}")

if [ -n "${MAX_REQUESTS}" ]; then
    CMD+=("--max-requests" "${MAX_REQUESTS}")
fi

if [ -n "${MAX_SECONDS}" ]; then
    CMD+=("--max-seconds" "${MAX_SECONDS}")
fi

if [ -n "${OUTPUT_PATH}" ]; then
    CMD+=("--output-path" "${OUTPUT_PATH}")
fi

# Execute the command
echo "Running command: ${CMD[*]}"
exec "${CMD[@]}"
