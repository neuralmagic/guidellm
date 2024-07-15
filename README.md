# guidellm

# Project configuration

The project is configured with environment variables. Check the example in `.env.example`.

```sh
# Create .env file and update the configuration
cp .env.example  .env

# Export all variables
set -o allexport; source .env; set +o allexport
```

## Environment Variables

| Variable        | Default Value         | Description                                                                                                                                                    |
| --------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OPENAI_BASE_URL | http://127.0.0.1:8080 | The host where the `openai` library will make requests to. For running integration tests it is required to have the external OpenAI compatible server running. |
| OPENAI_API_KEY  | invalid               | [OpenAI Platform](https://platform.openai.com/api-keys) to create a new API key. This value is not used for tests.                                             |

</br>

# Code Quality

The variety of tools are available for checking the code quality.

All the code quality tools configuration is placed into next files: `pyproject.toml`, `Makefile`, `tox.ini`

**To provide the code quality next tools are used:**

- `pytest` as a testing framework
- `ruff` as a linter
- `black` & `isort` as formatters
- `mypy` as a static type checker
- `tox` as a automation tool
- `make` as a automation tool (works only on Unix by default)

**Checking code quality using CLI**

All the tools could be run right from the CLI.

Recommended command template: `python -m pytest test.integration`


**Checking code quality using Makefile**

Using `Makefile` you can run almost all common scripts to check the code quality, install dependencies, and to provide auto fixes. All the commands from the `Makefile` are valid to be used in the CLI.

Here are some commands

```sh
# install dev dependencies
make install.dev

# run unit tests
make test.unit

# run quality checkers
make quality

# run autofixes
make style
```

**Checking code quality using tox**

The `tox` is an automation tool for running code quality checkers for selected Python versions. The configuration is placed in the `tox.ini`.

To run the automation just run: `python -m tox`
