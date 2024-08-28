from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DatasetSettings",
    "EmulatedDataSettings",
    "Environment",
    "LoggingSettings",
    "OpenAISettings",
    "ReportGenerationSettings",
    "Settings",
    "reload_settings",
    "settings",
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
    console_log_level: str = "WARNING"
    log_file: Optional[str] = None
    log_file_level: Optional[str] = None


class DatasetSettings(BaseModel):
    """
    Dataset settings for the application
    """

    preferred_data_columns: List[str] = Field(
        default_factory=lambda: [
            "prompt",
            "instruction",
            "input",
            "inputs",
            "question",
            "context",
            "text",
            "content",
            "body",
            "data",
        ]
    )
    preferred_data_splits: List[str] = Field(
        default_factory=lambda: ["test", "tst", "validation", "val", "train"]
    )
    default_tokenizer: str = "neuralmagic/Meta-Llama-3.1-8B-FP8"


class EmulatedDataSettings(BaseModel):
    """
    Emulated data settings for the application to use
    """

    source: str = "https://www.gutenberg.org/files/1342/1342-0.txt"
    filter_start: str = "It is a truth universally acknowledged, that a"
    filter_end: str = "CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO."
    clean_text_args: Dict[str, bool] = Field(
        default_factory=lambda: {
            "fix_encoding": True,
            "clean_whitespace": True,
            "remove_empty_lines": True,
            "force_new_line_punctuation": True,
        }
    )


class OpenAISettings(BaseModel):
    """
    OpenAI settings for the application to connect to the API
    for OpenAI server based pathways
    """

    # OpenAI API key.
    api_key: str = "invalid_token"

    # OpenAI-compatible server URL
    # NOTE: The default value is default address of llama.cpp web server
    base_url: str = "http://localhost:8000/v1"

    max_gen_tokens: int = 4096


class DeepsparseSettings(BaseModel):
    """
    Deepsparse settings for the Python API library
    """

    model: Optional[str] = None


class ReportGenerationSettings(BaseModel):
    """
    Report generation settings for the application
    """

    source: str = ""
    report_html_match: str = "window.report_data = {};"
    report_html_placeholder: str = "{}"


class Settings(BaseSettings):
    """
    All the settings are powered by pydantic_settings and could be
    populated from the .env file.

    The format to populate the settings is next

    ```sh
    export GUIDELLM__LOGGING__DISABLED=true
    export GUIDELLM__OPENAI__API_KEY=******
    export GUIDELLM__DEEPSPARSE__MODEL=******
    ```
    """

    model_config = SettingsConfigDict(
        env_prefix="GUIDELLM__",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
        env_file=".env",
    )

    # general settings
    env: Environment = Environment.PROD
    request_timeout: int = 30
    max_concurrency: int = 512
    num_sweep_profiles: int = 9
    logging: LoggingSettings = LoggingSettings()

    # Data settings
    dataset: DatasetSettings = DatasetSettings()
    emulated_data: EmulatedDataSettings = EmulatedDataSettings()

    # Request settings
    openai: OpenAISettings = OpenAISettings()
    deepsprase: DeepsparseSettings = DeepsparseSettings()
    report_generation: ReportGenerationSettings = ReportGenerationSettings()

    @model_validator(mode="after")
    @classmethod
    def set_default_source(cls, values):
        if not values.report_generation.source:
            values.report_generation.source = ENV_REPORT_MAPPING.get(values.env)

        return values


settings = Settings()


def reload_settings():
    """
    Reload the settings from the environment variables
    """
    new_settings = Settings()
    settings.__dict__.update(new_settings.__dict__)
