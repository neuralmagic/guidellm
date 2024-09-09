# Developer Documentation

## Introduction

Welcome to the developer documentation for `guidellm`, a guidance platform for evaluating large language model deployments. This document aims to provide developers with all the necessary information to contribute to the project effectively.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- `pip` (Python package installer)
- `git` (version control system)

### Clone the repository:

```sh
git clone https://github.com/neuralmagic/guidellm.git
cd guidellm
```

### Install dependencies:

All the dependencies are specified in `pyproject.toml` file. There is an option to install only required dependencies and optional dependencies

Install required dependencies along with optional `dev` dependencies.

```bash
git clone https://github.com/neuralmagic/guidellm.git
cd guidellm
pip install -e .[dev]
```

In case of working with `deepsparse` backend, etc it has some other software limitations. In order to install dependencies for the specific backend, run:

```sh
pip install -e .[deepsparse]
# or pip install -e '.[deepsparse]'
```

In case of working with `vllm` backend, etc it has some other software limitations. In order to install dependencies for the specific backend, run:

```sh
pip install -e .[vllm]
# or pip install -e '.[vllm]'
```

According to the [installation guide](https://docs.vllm.ai/en/v0.4.0.post1/getting_started/installation.html) `vllm` is supported only on **Linux**. It means that running the application and tests will fail.

Workaround with Docker:

```sh
cd guidellm/
docker build -t guidellm:latest .
docker run -v ./:./ guidellm:latest python -m pytest -s -v src/unit/backend/test_vllm.py
```

## Project Structure

The project follows a standard Python project structure:

```plaintext
guidellm/
├── src/
│ └── guidellm/
├── tests/
│ ├── unit/
│ ├── integration/
│ └── e2e/
├── pyproject.toml
├── tox.ini
└── README.md
```

- `pyproject.toml`: Project metadata
- `**src/guidellm/**`: Main source code for the project.
- `**tests/**`: Test cases categorized into unit, integration, and end-to-end tests.

## Development Environment Setup

To set up your development environment, follow these steps:

1. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

2. Ensure all dependencies are installed:

   ```bash
   pip install -e .[dev]
   ```

## Code Quality and Style Guidelines

We use several tools to maintain code quality and consistency:

### Ruff

Ruff is used for linting and formatting checks.

- Configuration is in `pyproject.toml`.
- To run Ruff:
  ```bash
  ruff check src tests
  ```

### Isort

Isort is used for sorting imports.

- Configuration is in `pyproject.toml`.
- To sort imports:
  ```bash
  isort src tests
  ```

### Flake8

Flake8 is used for linting.

- Configuration is in `tox.ini`.
- To run Flake8:
  ```bash
  flake8 src tests --max-line-length 88
  ```

### MyPy

MyPy is used for type checking.

- Configuration is in `pyproject.toml`.
- To run MyPy:
  ```bash
  mypy src/guidellm
  ```

## Testing

We use `pytest` for running tests.

### Running All Tests

To run all tests:

```bash
tox
```

Additionally, all tests are marked with smoke, sanity, or regression tags for better organization. To run tests with a specific tag, use the `-m` flag:

```bash
tox -- -m "smoke or sanity"
```

### Running Unit Tests

The unit tests are located in the `tests/unit` directory. To run the unit tests, use the following command:

```bash
tox -e test-unit
```

### Running Integration Tests

The integration tests are located in the `tests/integration` directory. To run the integration tests, use the following command:

```bash
tox -e test-integration
```

### Running End-to-End Tests

The end-to-end tests are located in the `tests/e2e` directory. To run the end-to-end tests, use the following command:

```bash
tox -e test-e2e
```

### Running unsopported tests

Some of the test might be not supported on your system (_for instance `vllm` is not supported on MacOS yet_). In order to run them on Linux Operating System you might use technologies like **WSL** on Windows, or **Docker** on Windows or MacOS.

In order to run under the Docker just run the command below:

```sh
docker build -t guidellm:latest --shm-size=1024m .
docker run --rm --env-file .env -v ./:/app/guidellm -it guidellm:latest pytest tests/
```

Using `--shm-size=1024m` is recommended due to potential local Docker configuration. Increase or decrease this value depending on your needs.

<br>

## Formatting, Linting, and Type Checking

### Running Quality Checks (Linting)

To run quality checks (ruff, isort, flake8, mypy), use the following command:

```bash
tox -e quality
```

### Running Auto Formatting and Validation

To run formatting checks and fixes (ruff, isort, flake8) for automatic formatting, use the following command:

```bash
tox -e style
```

### Type Checking

To run type checks (MyPy), use the following command:

```bash
tox -e types
```

## Building the Project

To build the project, use the following command:

```bash
tox -e build
```

## Cleaning Up

To clean up build, dist, and cache files, use the following command:

```bash
tox -e clean
```

## Continuous Integration/Continuous Deployment (CI/CD)

Our CI/CD pipeline is configured using GitHub Actions. The configuration files are located in the .github/workflows/ directory. You can run the CI/CD pipeline locally using act or similar tools.

## Contributing

Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to the project.

## Maintaining

Please refer to the MAINTAINERS file for maintenance guidelines and contact information.

## Project configuration

The project configuartion is powered by _[`🔗 pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)_

The project configuration entry point is represented by lazy-loaded `settigns` singleton object ( `src/config/__init__` )

The project is fully configurable with environment variables. With that configuration set you can load parameters to `LoggingSettings()` by using environment variables. Just run `export GUIDELLM__LOGGING__DISABLED=true` or `export GUIDELLM__LOGGING__NESTED=another_value` respectfully. The nesting delimiter is `__`.

### Environment variables

| Name                           | Default value           | Description                                                                                                                                             |
| ------------------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GUIDELLM__LOGGING__DISABLED`  | `False`                 | Determines whether logger for the `guidellm` package is disabled or not                                                                                 |
| `GUIDELLM__LOGGING__LOG_LEVEL` | `INFO`                  | The level of `guidellm` package logging                                                                                                                 |
| `GUIDELLM__LOGGING__LOG_FILE`  | `guidellm.log`          | The name of a log file                                                                                                                                  |
| `GUIDELLM__OPENAI__BASE_URL`   | `http://localhost:8080` | The address to the **OpenAI-compatible** server.<br><br>OpenAI live base url is `https://api.openai.com/v1`                                             |
| `GUIDELLM__OPENAI__API_KEY`    | `invalid`               | Corresponds to the **OpenAI-compatible** server API key.<br><br>If you look for the live key - check [this link](https://platform.openai.com/api-keys). |

<br>

## Project configuration

The project configuartion is powered by _[`🔗 pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)_

The project configuration entrypoint is represented by lazy-loaded `settigns` singleton object ( `src/config/__init__` )

The project is fully configurable with environment variables. All the default values and

```py
class Nested(BaseModel):
    nested: str = "default value"

class LoggingSettings(BaseModel):
    # ...

    disabled: bool = False
    nested: Nested = Nested()


class Settings(BaseSettings):
    """The entrypoint to settings."""

    # ...
    logging: LoggingSettings = LoggingSettings()


settings = Settings()
```

With that configuration set you can load parameters to `LoggingSettings()` by using environment variables. Just run `export GUIDELLM__LOGGING__DISABLED=true` or `export GUIDELLM__LOGGING__NESTED=another_value` respectfully. The nesting delimiter is `__`

## Contact and Support

If you need help or have any questions, please open an issue on GitHub or contact us at support@neuralmagic.com.
