from typing import Optional
from enum import Enum

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = [
    "settings",
    "Settings",
    "Environment",
    "LoggingSettings",
    "OpenAISettings",
    "ReportGenerationSettings",
]


class Environment(str, Enum):
    """
    Enum for the supported environments
    """

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


ENV_REPORT_MAPPING = {
    Environment.PROD: "https://guidellm.neuralmagic.com/local-report/index.html",
    Environment.STAGING: "https://staging.guidellm.neuralmagic.com/local-report/index.html",
    Environment.DEV: "https://dev.guidellm.neuralmagic.com/local-report/index.html",
    Environment.LOCAL: "tests/dummy/report.html",
}


class LoggingSettings(BaseModel):
    """
    Logging settings for the application
    """

    disabled: bool = False
    clear_loggers: bool = True
    console_log_level: str = "INFO"
    log_file: Optional[str] = None
    log_file_level: Optional[str] = None


class OpenAISettings(BaseModel):
    """
    OpenAI settings for the application to connect to the API
    for OpenAI server based pathways
    """

    # OpenAI API key.
    api_key: str = "invalid"

    # OpenAI-compatible server URL
    # NOTE: The default value is default address of llama.cpp web server
    base_url: str = "http://localhost:8080"


class ReportGenerationSettings(BaseModel):
    source: str = ""


class Settings(BaseSettings):
    """
    All the settings are powered by pydantic_settings and could be
    populated from the .env file.

    The format to populate the settings is next

    ```sh
    export GUIDELLM__LOGGING__DISABLED=true
    export GUIDELLM__OPENAI__API_KEY=******
    ```

    """

    model_config = SettingsConfigDict(
        env_prefix="GUIDELLM__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    env: Environment = Environment.PROD
    logging: LoggingSettings = LoggingSettings()
    openai: OpenAISettings = OpenAISettings()
    report_generation: ReportGenerationSettings = ReportGenerationSettings()

    @model_validator(mode="before")
    def set_default_source(cls, values):
        env = values.get("env", Environment.PROD)  # type: Environment

        if not values.get("report_generation"):
            values["report_generation"] = ReportGenerationSettings()

        if isinstance(values["report_generation"], dict) and not values[
            "report_generation"
        ].get("source"):
            values["report_generation"]["source"] = ENV_REPORT_MAPPING.get(env)
        elif (
            isinstance(values["report_generation"], ReportGenerationSettings)
            and not values["report_generation"].source
        ):
            values["report_generation"].source = ENV_REPORT_MAPPING[env]

        return values


settings = Settings()
