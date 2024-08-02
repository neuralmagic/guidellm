from enum import Enum
from typing import Optional

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

    max_gen_tokens: int = 4096


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
        extra="ignore",
        validate_default=True,
        env_file=".env",
    )

    env: Environment = Environment.PROD
    request_timeout: int = 30

    logging: LoggingSettings = LoggingSettings()
    openai: OpenAISettings = OpenAISettings()
    report_generation: ReportGenerationSettings = ReportGenerationSettings()

    @model_validator(mode="after")
    @classmethod
    def set_default_source(cls, values):
        if not values.report_generation.source:
            values.report_generation.source = ENV_REPORT_MAPPING.get(values.env)

        return values


settings = Settings()
