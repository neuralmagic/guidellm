#!/usr/bin/env bash
set -euo pipefail

# Path to the guidellm binary
guidellm_bin="/opt/guidellm/bin/guidellm"

# If we receive any arguments switch to guidellm command
if [ $# -gt 0 ]; then
    echo "Running command: guidellm $*"
    exec $guidellm_bin "$@"
fi

# Get a list of environment variables that start with GUIDELLM_
args="$(printenv | cut -d= -f1 | grep -E '^GUIDELLM_')"

# NOTE: Bash array + exec prevent shell escape issues
CMD=("${guidellm_bin}" "benchmark")

# Parse environment variables for the benchmark command
for var in $args; do
    # Remove GUIDELLM_ prefix
    arg_name="${var#GUIDELLM_}"
    # Convert to lowercase
    arg_name="${arg_name,,}"
    # Replace underscores with dashes
    arg_name="${arg_name//_/-}"

    # Add the argument to the command array if set
    if [ -n "${!var}" ]; then
        CMD+=("--${arg_name}" "${!var}")
    fi
done

# Execute the command
echo "Running command: ${CMD[*]}"
exec "${CMD[@]}"
