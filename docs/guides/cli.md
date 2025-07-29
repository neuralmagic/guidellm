# CLI Reference

This page provides a reference for the `guidellm` command-line interface. For more advanced configuration, including environment variables and `.env` files, see the [Configuration Guide](./configuration.md).

## `guidellm benchmark run`

This command is the primary entrypoint for running benchmarks. It has many options that can be specified on the command line or in a scenario file.

### Scenario Configuration

| Option                      | Description                                                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `--scenario <PATH or NAME>` | The name of a builtin scenario or path to a scenario configuration file. Options specified on the command line will override the scenario file. |

### Target and Backend Configuration

These options configure how `guidellm` connects to the system under test.

| Option                  | Description                                                                                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--target <URL>`        | **Required.** The endpoint of the target system, e.g., `http://localhost:8080`. Can also be set with the `GUIDELLM__OPENAI__BASE_URL` environment variable.                                                   |
| `--backend-type <TYPE>` | The type of backend to use. Defaults to `openai_http`.                                                                                                                                                        |
| `--backend-args <JSON>` | A JSON string for backend-specific arguments. For example: `--backend-args '{"headers": {"Authorization": "Bearer my-token"}, "verify": false}'` to pass custom headers and disable certificate verification. |
| `--model <NAME>`        | The ID of the model to benchmark within the backend.                                                                                                                                                          |

### Data and Request Configuration

These options define the data to be used for benchmarking and how requests will be generated.

| Option                    | Description                                                                                                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--data <SOURCE>`         | The data source. This can be a HuggingFace dataset ID, a path to a local data file, or a synthetic data configuration. See the [Data Formats Guide](./data_formats.md) for more details. |
| `--rate-type <TYPE>`      | The type of request generation strategy to use (e.g., `constant`, `poisson`, `sweep`).                                                                                                   |
| `--rate <NUMBER>`         | The rate of requests per second for `constant` or `poisson` strategies, or the number of steps for a `sweep`.                                                                            |
| `--max-requests <NUMBER>` | The maximum number of requests to run for each benchmark.                                                                                                                                |
| `--max-seconds <NUMBER>`  | The maximum number of seconds to run each benchmark for.                                                                                                                                 |
