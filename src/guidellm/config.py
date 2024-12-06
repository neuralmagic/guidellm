import json
from enum import Enum
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DatasetSettings",
    "EmulatedDataSettings",
    "Environment",
    "LoggingSettings",
    "OpenAISettings",
    "print_config",
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


class AiohttpSettings(OpenAISettings):
    pass

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
    aiohttp: AiohttpSettings = AiohttpSettings()

    # Report settings
    report_generation: ReportGenerationSettings = ReportGenerationSettings()

    @model_validator(mode="after")
    @classmethod
    def set_default_source(cls, values):
        if not values.report_generation.source:
            values.report_generation.source = ENV_REPORT_MAPPING.get(values.env)

        return values

    def generate_env_file(self) -> str:
        """
        Generate the .env file from the current settings
        """
        return Settings._recursive_generate_env(
            self,
            self.model_config["env_prefix"],  # type: ignore  # noqa: PGH003
            self.model_config["env_nested_delimiter"],  # type: ignore  # noqa: PGH003
        )

    @staticmethod
    def _recursive_generate_env(model: BaseModel, prefix: str, delimiter: str) -> str:
        env_file = ""
        add_models = []
        for key, value in model.model_dump().items():
            if isinstance(value, BaseModel):
                # add nested properties to be processed after the current level
                add_models.append((key, value))
                continue

            dict_values = (
                {
                    f"{prefix}{key.upper()}{delimiter}{sub_key.upper()}": sub_value
                    for sub_key, sub_value in value.items()
                }
                if isinstance(value, dict)
                else {f"{prefix}{key.upper()}": value}
            )

            for tag, sub_value in dict_values.items():
                if isinstance(sub_value, Sequence) and not isinstance(sub_value, str):
                    value_str = ",".join(f'"{item}"' for item in sub_value)
                    env_file += f"{tag}=[{value_str}]\n"
                elif isinstance(sub_value, Dict):
                    value_str = json.dumps(sub_value)
                    env_file += f"{tag}={value_str}\n"
                elif not sub_value:
                    env_file += f"{tag}=\n"
                else:
                    env_file += f'{tag}="{sub_value}"\n'

        for key, value in add_models:
            env_file += Settings._recursive_generate_env(
                value, f"{prefix}{key.upper()}{delimiter}", delimiter
            )
        return env_file


settings = Settings()


def reload_settings():
    """
    Reload the settings from the environment variables
    """
    new_settings = Settings()
    settings.__dict__.update(new_settings.__dict__)


def print_config():
    """
    Print the current configuration settings
    """
    print(f"Settings: \n{settings.generate_env_file()}")  # noqa: T201


if __name__ == "__main__":
    print_config()
