# Configuration

The `guidellm` application can be configured using command-line arguments, environment variables, or a `.env` file. This page details the file-based and environment variable configuration options.

## Configuration Methods

Settings are loaded with the following priority (highest priority first):

1. Command-line arguments.
2. Environment variables.
3. Values in a `.env` file in the directory where the command is run.
4. Default values.

## Environment Variable Format

All settings can be configured using environment variables. The variables must be prefixed with `GUIDELLM__`, and nested settings are separated by a double underscore `__`.

For example, to set the `api_key` for the `openai` backend, you would use the following environment variable:

```bash
export GUIDELLM__OPENAI__API_KEY="your-api-key"
```

### Target and Backend Configuration

You can configure the connection to the target system using environment variables. This is an alternative to using the `--target-*` command-line flags.

| Environment Variable                  | Description                                                                                                                | Example                                                                   |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `GUIDELLM__OPENAI__BASE_URL`          | The endpoint of the target system. Equivalent to the `--target` CLI option.                                                | `export GUIDELLM__OPENAI__BASE_URL="http://localhost:8080"`               |
| `GUIDELLM__OPENAI__API_KEY`           | The API key to use for bearer token authentication.                                                                        | `export GUIDELLM__OPENAI__API_KEY="your-secret-api-key"`                  |
| `GUIDELLM__OPENAI__BEARER_TOKEN`      | The full bearer token to use for authentication.                                                                           | `export GUIDELLM__OPENAI__BEARER_TOKEN="Bearer your-secret-token"`        |
| `GUIDELLM__OPENAI__HEADERS`           | A JSON string representing a dictionary of headers to send to the target. These headers will override any default headers. | `export GUIDELLM__OPENAI__HEADERS='{"Authorization": "Bearer my-token"}'` |
| `GUIDELLM__OPENAI__ORGANIZATION`      | The OpenAI organization to use for requests.                                                                               | `export GUIDELLM__OPENAI__ORGANIZATION="org-12345"`                       |
| `GUIDELLM__OPENAI__PROJECT`           | The OpenAI project to use for requests.                                                                                    | `export GUIDELLM__OPENAI__PROJECT="proj-67890"`                           |
| `GUIDELLM__OPENAI__VERIFY`            | Set to `false` or `0` to disable certificate verification.                                                                 | `export GUIDELLM__OPENAI__VERIFY=false`                                   |
| `GUIDELLM__OPENAI__MAX_OUTPUT_TOKENS` | The default maximum number of tokens to request for completions.                                                           | `export GUIDELLM__OPENAI__MAX_OUTPUT_TOKENS=2048`                         |

### General HTTP Settings

These settings control the behavior of the underlying HTTP client.

| Environment Variable                 | Description                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| `GUIDELLM__REQUEST_TIMEOUT`          | The timeout in seconds for HTTP requests. Defaults to 300.                      |
| `GUIDELLM__REQUEST_HTTP2`            | Set to `true` or `1` to enable HTTP/2 support. Defaults to true.                |
| `GUIDELLM__REQUEST_FOLLOW_REDIRECTS` | Set to `true` or `1` to allow the client to follow redirects. Defaults to true. |

### Using a `.env` file

You can also place these variables in a `.env` file in your project's root directory:

```dotenv
# .env file
GUIDELLM__OPENAI__BASE_URL="http://localhost:8080"
GUIDELLM__OPENAI__API_KEY="your-api-key"
GUIDELLM__OPENAI__HEADERS='{"Authorization": "Bearer my-token"}'
GUIDELLM__OPENAI__VERIFY=false
```
