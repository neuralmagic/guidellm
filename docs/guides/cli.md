# CLI Reference

This page provides a reference for the `guidellm` command-line interface. For more advanced configuration, including environment variables and `.env` files, see the [Configuration Guide](./configuration.md).

## `guidellm benchmark run`

This command is the primary entrypoint for running benchmarks.

### Target Configuration

These options configure how `guidellm` connects to the system under test.

| Option | Description |
| --- | --- |
| `--target <URL>` | **Required.** The endpoint of the target system, e.g., `http://localhost:8080`. |
| `--target-header <HEADER>` | A header to send with requests to the target. This option can be specified multiple times to send multiple headers. The header should be in the format `"Header-Name: Header-Value"`. For example: `--target-header "Authorization: Bearer my-secret-token"` |
| `--target-skip-ssl-verify` | A flag to disable SSL certificate verification when connecting to the target. This is useful for development environments with self-signed certificates, but should be used with caution in production. |